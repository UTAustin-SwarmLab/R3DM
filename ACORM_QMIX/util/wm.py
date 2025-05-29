import torch 
from util import networks
from torch import nn
from util import tools

to_np = lambda x: x.detach().cpu().numpy()
class DreamerWorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config, role_config = None):
        super(DreamerWorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        
        shapes = obs_space
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        
        # Role config contains "role_embed_size" to indicate the WM takes in role_inputs
        self.role_config = role_config
        if self._config.rssm_role:
            self.dynamics = networks.RSSM(
                config.dyn_stoch,
                config.dyn_deter,
                config.dyn_hidden,
                config.dyn_rec_depth,
                config.dyn_discrete,
                config.act,
                config.norm,
                config.dyn_mean_act,
                config.dyn_std_act,
                config.dyn_min_std,
                config.unimix_ratio,
                config.initial,
                config.num_actions,
                self.embed_size,
                config.device,
                role_config
            )
        else:
            self.dynamics = networks.RSSM(
                config.dyn_stoch,
                config.dyn_deter,
                config.dyn_hidden,
                config.dyn_rec_depth,
                config.dyn_discrete,
                config.act,
                config.norm,
                config.dyn_mean_act,
                config.dyn_std_act,
                config.dyn_min_std,
                config.unimix_ratio,
                config.initial,
                config.num_actions,
                self.embed_size,
                config.device
            )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        
        if role_config is not None and config.decode_role:
            feat_size += role_config['role_embed_size']
            
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict()

    def predict(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(
            embed, data["action"], data["is_first"]
        )
        return self.heads["decoder"](self.dynamics.get_feat(states))['vector_obs']
    
    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                # TODO : Include role information
                role_embed_dyn = None if (self.role_config is None or not self._config.rssm_role) else data['role_embed']
                role_embed_dec = None if (self.role_config is None or not self._config.decode_role) else data['role_embed']
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"], role_embed_dyn
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    if self._config.decode_role:
                        feat = torch.cat([feat, role_embed_dec.detach()], dim=-1) if role_embed_dec is not None else feat
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                model_loss = model_loss * data["mask"]
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss * data["mask"]) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss * data["mask"])
        metrics["rep_loss"] = to_np(rep_loss * data["mask"])
        metrics["kl"] = to_np(torch.mean(kl_value * data["mask"]))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy() * data["mask"])
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy() * data["mask"])
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        # We dont have images
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def predict_test(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        # TODO: Remove the image portion
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["vector_obs"].mode()[
            :6
        ]
        #reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["vector_obs"].mode()
        #reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2) 

class SimpleWorldModel(nn.Module):
    
    def __init__(self,obs_space, act_space, step, config, role_config = None) -> None:
        super().__init__()
        self._step = step
        self.obs_space = obs_space
        
        
        
    