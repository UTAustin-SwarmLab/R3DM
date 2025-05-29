import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.acorm import ACORM_Agent
from util.net import *
from util.attention import MultiHeadAttention
# from algorithm.vdn_qmix import QMIX_Net
import numpy as np
import copy
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
# from kmeans_pytorch import kmeans
from algorithm.acorm import RECL_MIX, RECL_NET
from util import wm

to_np = lambda x: x.detach().cpu().numpy()
class R3DM_Agent(ACORM_Agent):
    def __init__(self, args, dyn_model_config, env_info):
        super(R3DM_Agent, self).__init__(args)
        self.args = args
        
        #TODO: Create 2 word models, one wtih role embeddings and one without role embeddings
        obs_space = {'vector_obs':[env_info['obs_shape']]}
        act_space = {'vector_act':[env_info['n_actions']]}
        dyn_model_config.num_actions = env_info['n_actions']
        dyn_model_config.device = str(self.device)
        dyn_model_config.encoder.device = str(self.device)
        dyn_model_config.decoder.device = str(self.device)
        dyn_model_config.num_actions = env_info['n_actions']
        
        role_config = {'role_embed_size':args.role_embedding_dim}
        self.dyn_model_config = dyn_model_config
        
        self.wm_role = wm.DreamerWorldModel(obs_space, act_space, 5, dyn_model_config, role_config=role_config)
        self.wm_role.to(self.device)
        self.wm = wm.DreamerWorldModel(obs_space, act_space, 5, dyn_model_config)
        self.wm.to(self.device)
        
        self.target_wm_role = wm.DreamerWorldModel(obs_space, act_space, 5, dyn_model_config, role_config=role_config)
        self.target_wm_role.to(self.device)
        self.target_wm = wm.DreamerWorldModel(obs_space, act_space, 5, dyn_model_config)
        self.target_wm.to(self.device) 
        
    def get_role_embedding(self, obs_n, last_a):
        recl_obs = torch.tensor(np.array(obs_n), dtype=torch.float32).to(self.device)
        recl_last_a = torch.tensor(np.array(last_a), dtype=torch.float32).to(self.device)
        role_embedding = self.RECL(recl_obs, recl_last_a, detach=True)
        return role_embedding
    
    def choose_action(self, obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
            else:
                inputs = copy.deepcopy(obs_n)
                if self.add_last_action:
                    inputs = np.hstack((inputs, last_onehot_a_n))
                inputs = np.hstack((inputs, role_embedding.to('cpu')))
                inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs = inputs.to(self.device)

                q_value = self.eval_Q_net(inputs)
                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                q_value = q_value.to('cpu')
                q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions

                a_n = q_value.argmax(dim=-1).numpy()
        return a_n
            
    def get_inputs(self, batch):
        inputs = copy.deepcopy(batch['obs_n'])
        if self.add_last_action:
            inputs = np.concatenate((inputs, batch['last_onehot_a_n']),axis=-1)
        inputs = torch.tensor(inputs, dtype=torch.float32)

        inputs = inputs.to(self.device)
        batch_o = batch['obs_n'].to(self.device)
        batch_s = batch['s'].to(self.device)
        batch_r = batch['r'].to(self.device)
        batch_a = batch['a_n'].to(self.device)
        batch_last_a = batch['last_onehot_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        batch_dw = batch['dw'].to(self.device)
        batch_avail_a_n = batch['avail_a_n']
        return inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw
    
    def train(self, replay_buffer, total_steps):
        self.train_step += 1
        batch, max_episode_len = replay_buffer.sample(self.batch_size)  # Get training data
        inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a,\
        batch_avail_a_n, batch_active, batch_dw= self.get_inputs(batch)

        # Train the world models
        wm_metrics = None
        for _ in range(self.dyn_model_config.iterations):
            m = self.train_world_models(replay_buffer)
            if wm_metrics is None:
                wm_metrics = {k:[v] for k,v in m.items()}
            else:
                wm_metrics = {k:v + [m[k]] for k,v in wm_metrics.items()}
            self.soft_update_params(self.wm, self.target_wm, self.tau)
            self.soft_update_params(self.wm_role, self.target_wm_role, self.tau)
        
        metrics = {k:np.mean(v) for k,v in wm_metrics.items()}

        if self.train_recl_freq !=0 and self.train_step % self.train_recl_freq == 0:
            loss_dict = self.update_recl(inputs, batch_o, batch_last_a, batch_active, max_episode_len)
            self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)
            for k,v in loss_dict.items():
                metrics[f'recl/{k}'] = v
        else:
            metrics['train/recl_loss'] = 0.0
            
        loss, intr_reward_dict = self.update_qmix(inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw, max_episode_len, total_steps)
        metrics['train/mix_loss'] = loss.item()
        for k,v in intr_reward_dict.items():
            metrics[f'rewards/{k}'] = v
        metrics[f'train/batch_reward'] = batch_r.squeeze().sum(dim=-1).mean()
        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            # Softly update the target networks
            self.soft_update_params(self.eval_Q_net, self.target_Q_net, self.tau)
            self.soft_update_params(self.eval_mix_net, self.target_mix_net, self.tau)
        self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.tau)
    
        if self.use_lr_decay:
                self.qmix_lr_decay.step()
                self.role_lr_decay.step()
        
        return metrics

    def train_world_models(self, replay_buffer, soft_update=True):
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        # Compute role embeddings and detach them here
        
        inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a,\
        batch_avail_a_n, batch_active, batch_dw = self.get_inputs(batch)
        with torch.no_grad():
            if self.args.agent_net == 'Agent_Embedding':
                _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=True) # shape=(batch_size, (max_episode_len+1),N, role_embed_dim)
            else:
                _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=True) # shape=(batch_size, (max_episode_len+1),N, role_embed_dim)
            
        batch, length, num_agents, dim = batch_o.shape
        is_first = torch.zeros((batch, length, num_agents))
        is_first[:, 0, :] = 1.0
        is_first = is_first.to(self.device)
        
        # Reshape batch_o to batch*num_agents, length, dim
        batch_o = batch_o.permute(0,2,1,3).reshape(-1, length, dim)
        is_first = is_first.permute(0,2,1).reshape(-1, length)
        batch_last_a = batch_last_a.permute(0,2,1,3).reshape(-1, length, self.args.action_dim)
        role_embeddings = role_embeddings.permute(0,2,1,3).reshape(-1, length, self.args.role_embedding_dim)
        
        # role embeddings concat 0 across dim =1 and get upto second last element
        role_embeddings = torch.concat((torch.zeros_like(role_embeddings[:,0:1]), role_embeddings), dim=1)
        role_embeddings = role_embeddings[:, :-1]
        
        batch_active = batch_active.expand(-1, -1, num_agents)
        batch_active = torch.cat([batch_active, batch_active[:,-1:,:]], dim=1)
        wm_role_datadict = {
            'vector_obs': batch_o,
            'action': batch_last_a,
            'is_first' : is_first,
            'role_embed' : role_embeddings.detach(),
            'mask': batch_active.permute(0,2,1).reshape(-1, length)
            }
        _, _, metrics_role = self.wm_role._train(wm_role_datadict)
        
        wm_datadict = {
            'vector_obs': batch_o,
            'action': batch_last_a,
            'is_first' : is_first,
            'mask': batch_active.permute(0,2,1).reshape(-1, length)
        }
        
        _, _, metrics = self.wm._train(wm_datadict)
        # Combine the metrics and role in front of metrics_role keys
        metrics = {f'wm/{k}':np.mean(v) for k,v in metrics.items()}
        metrics.update({f'wm_role/{k}':np.mean(v) for k,v in metrics_role.items()})
        
        if soft_update:
            self.soft_update_params(self.wm, self.target_wm, self.dyn_model_config.dyna_tau)
            self.soft_update_params(self.wm_role, self.target_wm_role, self.dyn_model_config.dyna_tau)
        else:
            self.target_wm.load_state_dict(self.wm.state_dict())
            self.target_wm_role.load_state_dict(self.wm_role.state_dict())
            
        return metrics

    
    def pretrain_recl(self, replay_buffer):
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw = self.get_inputs(batch)
        loss_dict = self.update_recl(inputs, batch_o, batch_last_a, batch_active, max_episode_len )
        self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)
        return loss_dict['total_loss']
           
    def update_qmix(self, inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw, max_episode_len, total_steps):
        self.eval_Q_net.rnn_hidden = None
        self.target_Q_net.rnn_hidden = None
        # Detach agent_embeddings. detached embeddings
        if self.args.agent_net == 'Agent_Embedding':
            _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=False) # shape=(batch_size, (max_episode_len+1),N, role_embed_dim)
        else:
            _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=False) # shape=(batch_size, (max_episode_len+1),N, role_embed_dim)
        inputs_copy = inputs.clone()
        inputs = torch.cat([inputs, role_embeddings], dim=-1)
        q_evals, q_targets = [], []

        self.eval_mix_net.state_gru_hidden = None
        # self.target_mix_net.state_gru_hidden = None
        fc_batch_s = F.relu(self.eval_mix_net.state_fc(batch_s.reshape(-1, self.state_dim))).reshape(-1, max_episode_len+1, self.state_dim)    # shape(batch*max_len+1, state_dim)
        state_gru_outs = []
        for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
            q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.QMIX_input_dim))  # q_eval.shape=(batch_size*N,action_dim)
            q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.QMIX_input_dim))
            q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))  # q_eval.shape=(batch_size,N,action_dim)
            q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(fc_batch_s[:, t].reshape(-1,self.state_dim), self.eval_mix_net.state_gru_hidden)   # shape=(batch, N*state_embed_dim)
            state_gru_outs.append(self.eval_mix_net.state_gru_hidden)

        #     role_eval = self.eval_mix_net.role_gru_forward(role_embeddings[:,t].reshape(-1, self.N*self.role_embedding_dim))   # shape=(batch_size, N*role_embed_dim)
        #     role_target = self.target_mix_net.role_gru_forward(role_embeddings[:,t].reshape(-1, self.N*self.role_embedding_dim))   # shape=(batch_size, N*role_embed_dim)
        #     role_evals.append(role_eval)
        #     role_targets.append(role_target)
        self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(fc_batch_s[:, max_episode_len].reshape(-1,self.state_dim), self.eval_mix_net.state_gru_hidden)
        state_gru_outs.append(self.eval_mix_net.state_gru_hidden)
        # role_targets.append(self.target_mix_net.role_gru_forward(role_embeddings[:,max_episode_len].reshape(-1, self.N*self.role_embedding_dim)))

        # Stack them according to the time (dim=1)
        # role_evals = torch.stack(role_evals, dim=1)     # shape=(batch_size, max_len+1, N*role_dim)
        # role_targets = torch.stack(role_targets, dim=1) 
        state_gru_outs = torch.stack(state_gru_outs, dim=1).reshape(-1, self.N, self.args.state_embed_dim) # shape=(batch*max_len+1, N,state_embed_dim)
        q_evals = torch.stack(q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
        q_targets = torch.stack(q_targets, dim=1)

        # Take argmax from the current entwork to compute the values from the target network
        with torch.no_grad():
            q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.QMIX_input_dim)).reshape(self.batch_size, 1, self.N, -1)
            q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
            q_evals_next[batch_avail_a_n[:, 1:] == 0] = -999999
            a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
            q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
        q_evals = torch.gather(q_evals, dim=-1, index=batch_a.unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)
        
        role_embeddings_rs = role_embeddings.reshape(-1, self.N, self.role_embedding_dim) # shape=((batch_size * max_episode_len+1), N, role_embed_dim)
        # eval_state_qv = self.eval_mix_net.state_fc2(batch_s.reshape(-1, self.state_dim)).reshape(-1, self.N, self.eval_mix_net.dim_k) # shape=(batch*max_len, N, state_dim//N)
        # target_state_qv = self.target_mix_net.state_fc2(batch_s.reshape(-1, self.state_dim)).reshape(-1, self.N, self.eval_mix_net.dim_k)
        # agent_embeddings = agent_embeddings.reshape(-1, self.N, self.agent_embedding_dim)
        att_eval = self.eval_mix_net.attention_net(state_gru_outs, role_embeddings_rs, role_embeddings_rs).reshape(-1, max_episode_len+1, self.N*self.att_out_dim) # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)
        with torch.no_grad():
            att_target = self.target_mix_net.attention_net(state_gru_outs, role_embeddings_rs, role_embeddings_rs).reshape(-1, max_episode_len+1, self.N*self.att_out_dim) # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)
        
        # eval_batch_s = self.eval_mix_net.state_fc1(batch_s.reshape(-1, self.state_dim)).reshape(-1, max_episode_len+1, self.state_dim)
        # traget_batch_s = self.target_mix_net.state_fc1(batch_s.reshape(-1, self.state_dim)).reshape(-1, max_episode_len+1, self.state_dim)

        q_total_eval = self.eval_mix_net(q_evals, fc_batch_s[:, :-1], att_eval[:, :-1])
        q_total_target = self.target_mix_net(q_targets, fc_batch_s[:, 1:], att_target[:, 1:])
        
        
        # TODO: Compute the intrinsic reward
        if self.args.intrinsic_reward:
            with torch.no_grad():
                intrinsic_reward, rewards_dict = self.compute_wm_reward(inputs_copy, batch_o, batch_last_a, role_embeddings.detach(), max_episode_len, total_steps)
            batch_r_mod = batch_r + self.dyn_model_config.intr_weight*intrinsic_reward.unsqueeze(-1).detach()
        else:
            rewards_dict = {}
            batch_r_mod = batch_r.clone()
       
        #rewards_dict['final_intrinsic_reward_mean'] = self.dyn_model_config.intr_weight*intrinsic_reward.unsqueeze(-1).detach().mean()
        targets = batch_r_mod + self.gamma * (1 - batch_dw) * q_total_target
        td_error = (q_total_eval - targets.detach())    # targets.detach() to cut the backward
        mask_td_error = td_error * batch_active
        loss = (mask_td_error ** 2).sum() / batch_active.sum()
        self.optimizer.zero_grad()
        self.role_embedding_optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.role_parameters, 10)
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)        
        self.optimizer.step()
        self.role_embedding_optimizer.step()
        
        return loss, rewards_dict
        # Add losses and metrics to the metrics dictionary
    
    
    def update_recl(self, inputs, batch_o, batch_last_a, batch_active, max_episode_len):
        """
        N = agent_num
        batch_o.shape = (batch_size, max_episode_len + 1, N,  obs_dim)
        batch_a.shape = (batch_size, max_episode_len, N,  action_dim)
        batch_active = (batch_size, max_episode_len, 1)
        """
        self.RECL.agent_embedding_net.rnn_hidden = None
        loss = 0
        labels = np.zeros((batch_o.shape[0], self.N))  # (batch_size, N)
        role_embeddings = []
        for t in range(max_episode_len):    # t = 0,1,2...(max_episode_len-1)
            with torch.no_grad():
                agent_embedding = self.RECL.agent_embedding_forward(batch_o[:, t].reshape(-1, self.obs_dim),
                                            batch_last_a[:, t].reshape(-1, self.action_dim),
                                            detach=True)  # agent_embedding.shape=(batch_size*N, agent_embed_dim)
            role_embedding_qury = self.RECL.role_embedding_forward(agent_embedding,
                                                                   detach=False,
                                                                   ema=False).reshape(-1,self.N, self.role_embedding_dim)   # shape=(batch_size, N, role_embed_dim)
            role_embedding_key = self.RECL.role_embedding_forward(agent_embedding,
                                                                  detach=True,
                                                                  ema=True).reshape(-1,self.N, self.role_embedding_dim)
            role_embeddings.append(role_embedding_qury)
                                                                      
            logits = torch.bmm(role_embedding_qury, self.RECL.W.squeeze(0).expand((role_embedding_qury.shape[0],self.role_embedding_dim,self.role_embedding_dim)))
            logits = torch.bmm(logits, role_embedding_key.transpose(1,2))   # (batch_size, N, N)
            logits = logits - torch.max(logits, dim=-1)[0][:,:,None]
            exp_logits = torch.exp(logits) # (batch_size, N, 1)
            agent_embedding = agent_embedding.reshape(batch_o.shape[0],self.N, -1).to('cpu')  # shape=(batch_size,N, agent_embed_dim)

            for idx in range(agent_embedding.shape[0]): # idx = 0,1,2...(batch_size-1)
                if batch_active[idx, t] > 0.5:
                    if t % self.multi_steps == 0:
                        clusters_labels = KMeans(n_clusters=self.cluster_num).fit(agent_embedding[idx]).labels_ # (1,N)
                        labels[idx] = copy.deepcopy(clusters_labels)
                    else:
                        clusters_labels = copy.deepcopy(labels[idx])
                    # clusters_labels, _ = kmeans(X=agent_embedding[idx],num_clusters=self.cluster_num)
                    for j in range(self.cluster_num):   # j = 0,1,...(cluster_num -1)
                        label_pos = [idx for idx, value in enumerate(clusters_labels) if value==j]
                        # label_neg = [idx for idx, value in enumerate(clusters_labels) if value!=j]
                        for anchor in label_pos:
                            loss += -torch.log(exp_logits[idx, anchor, label_pos].sum()/exp_logits[idx, anchor].sum())
        loss /= (self.batch_size * max_episode_len * self.N)

        loss_dict = {}
        if not self.args.intrinsic_reward:

            role_embeddings = torch.stack(role_embeddings, dim=1)  # role_embeddings.shape=(batch_size, max_episode_len, N, role_embedding_dim)
            dynamic_loss, loss_dict = self.compute_wm_role_gradient(
                inputs,
                batch_o,
                batch_last_a,
                role_embeddings, 
                max_episode_len, 
            )
            loss_dict['recl_loss']  = loss.item()
            loss = loss + dynamic_loss
            
        self.RECL_optimizer.zero_grad()
        loss.backward()
        self.RECL_optimizer.step()
        loss_dict['total_loss'] = loss.item()
        return loss_dict       
    
    def compute_wm_role_gradient(self, inputs, batch_o, batch_last_a, role_embeddings, max_episode_len):         # expand the inputs shape to accomodate 
        # Inputs shape = (batch_size, max_episode_len, N, input_dim)
        # role_embeddings shape = (batch_size, max_episode_len, N, role_embedding_dim)
        # Tile inputs along new dim =-2 
        
        # I want to create a tensor that combines all N X N combinatons of inputs and role_mebeddings
        # This will allow me to compute the intrinsic reward for each agent
        
        self.eval_Q_net.rnn_hidden = None
        q_evals_agent = []
        q_evals_average = []
        for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
            inputs_expand = inputs[:, t].unsqueeze(1).expand(-1, self.N, -1, -1)
            role_embeddings_expand = role_embeddings[:, t].unsqueeze(2).expand(-1, -1, self.N, -1)
            inputs_expand = torch.cat([inputs_expand, role_embeddings_expand], dim=-1)
            q_eval = self.eval_Q_net(inputs_expand.reshape(-1, self.QMIX_input_dim))  # q_eval.shape=(batch_size*N*N,action_dim)
            q_eval = q_eval.reshape(self.batch_size, self.N, self.N, -1)  # q_eval.shape=(batch_size,N,N,action_dim)
            
            #Get diagonal elements
            q_eval_agent = q_eval.diagonal(dim1=1,dim2=2).permute(0,2,1)
            q_evals_agent.append(q_eval_agent)
            q_evals_average.append(q_eval)
        
        
        q_evals_agent = torch.stack(q_evals_agent, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
        q_evals_average = torch.stack(q_evals_average, dim=1) # batch_size,max_episode_len,N,N,action_dim)
        
        # Compute KL divergence b/w softmax of q_evals_agent and q_evals_average with efficient implementation
        q_evals_agent_logit = torch.softmax(self.dyn_model_config.intr_beta1 * q_evals_agent, dim=-1)
        q_evals_average_logit = torch.softmax(q_evals_average, dim=-1)
        q_evals_average_logit = q_evals_average_logit.mean(dim=-3)
        # q_evals_agent_logit = self.dyn_model_config.intr_beta1 * (q_evals_agent - q_evals_agent.max(dim=-1, keepdim=True)[0])
        # q_evals_average_logit = q_evals_average - q_evals_average.max(dim=-1, keepdim=True)[0]
        # q_evals_agent_logit = torch.exp(q_evals_agent_logit - torch.logsumexp(q_evals_agent_logit, dim=-1, keepdim=True))
        # q_evals_average_logit = torch.exp(q_evals_average_logit - torch.logsumexp(q_evals_average_logit, dim=-1, keepdim=True))
        # q_evals_average_logit = q_evals_average_logit.mean(dim=-2).detach()
        
        policy_intrinsic_loss = self.dyn_model_config.intr_beta2 * (q_evals_agent_logit * (q_evals_agent_logit / q_evals_average_logit).log()).sum(dim=-1) 
        policy_intrinsic_loss = policy_intrinsic_loss.mean() # shape=(batch_size, max_episode_len)
    
       
        # Compute reward from log prior from dynamics model
        B, length, N, dim = batch_o.shape
        data_dict = {
            'vector_obs':  batch_o.permute(0,2,1,3).reshape(-1, length, dim).clone()
        }
        embed = self.target_wm.encoder(data_dict)
        embed_role = self.target_wm_role.encoder(data_dict)
        state_t = None
        state_role_t = None
        
        pred_loss = 0.0
        for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
            # Predict
            batch_last_a_t = batch_last_a[:, t].reshape(-1, self.action_dim)
            batch_a_t = batch_last_a[:, t+1].reshape(-1, self.action_dim)
            # role_embeddings_t = role_embeddings[:, t].reshape(-1, self.role_embedding_dim)
            if t==0:
                role_embeddings_t = torch.zeros_like(role_embeddings[:, t].reshape(-1, self.role_embedding_dim))
                is_first = torch.ones((embed.shape[0]))
            else:
                role_embeddings_t = role_embeddings[:, t-1].reshape(-1, self.role_embedding_dim)
                is_first = torch.zeros((embed.shape[0]))
            
            state_t, _ = self.target_wm.dynamics.obs_step(
                 state_t, batch_last_a_t, embed[:,t], is_first
            )
            
            if self.dyn_model_config.rssm_role:
                state_role_t,_ = self.target_wm_role.dynamics.obs_step(
                    state_role_t, batch_last_a_t, embed_role[:,t], is_first, role_embeddings_t
                )
            else:
                state_role_t,_ = self.target_wm_role.dynamics.obs_step(
                    state_role_t, batch_last_a_t, embed_role[:,t], is_first
                )
                
            state_pred_t = self.target_wm.dynamics.img_step(state_t, batch_a_t)
            if self.dyn_model_config.rssm_role:
                state_pred_role_t = self.target_wm_role.dynamics.img_step(state_role_t,
                                                                batch_a_t, 
                                                               role_embeddings[:, t].reshape(-1, self.role_embedding_dim))
            else:
                state_pred_role_t = self.target_wm_role.dynamics.img_step(state_role_t,
                                                                batch_a_t)
            
            pred_obs_t = self.wm.heads["decoder"](self.target_wm.dynamics.get_feat(state_pred_t))['vector_obs']
            # TODO: Include the role embeddings
            state_pred_role_feat_t = self.target_wm_role.dynamics.get_feat(state_pred_role_t)
            if self.dyn_model_config.decode_role:
                state_pred_role_feat_t = torch.cat([state_pred_role_feat_t, role_embeddings[:, t].reshape(-1, self.role_embedding_dim)], dim=-1)    
            pred_obs_role_t = self.target_wm_role.heads["decoder"](state_pred_role_feat_t)['vector_obs']
            obs_dim = batch_o.shape[-1]
           
            loss = \
                (-pred_obs_t.log_prob(batch_o[:, t+1].reshape(-1, obs_dim), sum_dim = 1) \
                    + self.dyn_model_config.intr_beta1 * pred_obs_role_t.log_prob(batch_o[:, t+1].reshape(-1, obs_dim ), sum_dim = 1))

            pred_loss += loss * self.dyn_model_config.intr_beta3
        pred_loss = pred_loss.sum()
        pred_loss = pred_loss/(self.batch_size * max_episode_len * self.N)
        total_loss = pred_loss + policy_intrinsic_loss
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "pred_loss": pred_loss.item(),
            "policy_intrinsic_loss": policy_intrinsic_loss.item()
        }
        return total_loss, loss_dict
    
    def compute_wm_reward(self, inputs, batch_o, batch_last_a, role_embeddings, max_episode_len, total_steps):         # expand the inputs shape to accomodate 
        # Inputs shape = (batch_size, max_episode_len, N, input_dim)
        # role_embeddings shape = (batch_size, max_episode_len, N, role_embedding_dim)
        # Tile inputs along new dim =-2 
        
        # I want to create a tensor that combines all N X N combinatons of inputs and role_mebeddings
        # This will allow me to compute the intrinsic reward for each agent
        
        self.eval_Q_net.rnn_hidden = None
        q_evals_agent = []
        q_evals_average = []
        for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
            inputs_expand = inputs[:, t].unsqueeze(1).expand(-1, self.N, -1, -1)
            role_embeddings_expand = role_embeddings[:, t].unsqueeze(2).expand(-1, -1, self.N, -1)
            inputs_expand = torch.cat([inputs_expand, role_embeddings_expand], dim=-1)
            q_eval = self.eval_Q_net(inputs_expand.reshape(-1, self.QMIX_input_dim))  # q_eval.shape=(batch_size*N*N,action_dim)
            q_eval = q_eval.reshape(self.batch_size, self.N, self.N, -1)  # q_eval.shape=(batch_size,N,N,action_dim)
            
            #Get diagonal elements
            q_eval_agent = q_eval.diagonal(dim1=1,dim2=2).permute(0,2,1)
            q_evals_agent.append(q_eval_agent)
            q_evals_average.append(q_eval)
        
        
        q_evals_agent = torch.stack(q_evals_agent, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
        q_evals_average = torch.stack(q_evals_average, dim=1) # batch_size,max_episode_len,N,N,action_dim)
        
        # Compute KL divergence b/w softmax of q_evals_agent and q_evals_average with efficient implementation
        
        q_evals_agent_logit = torch.softmax(self.dyn_model_config.intr_beta1 * q_evals_agent, dim=-1)
        q_evals_average_logit = torch.softmax(q_evals_average, dim=-1)
        q_evals_average_logit = q_evals_average_logit.mean(dim=-3)
        # q_evals_agent_logit = self.dyn_model_config.intr_beta1 * (q_evals_agent - q_evals_agent.max(dim=-1, keepdim=True)[0])
        # q_evals_average_logit = q_evals_average - q_evals_average.max(dim=-1, keepdim=True)[0]
        # q_evals_agent_logit = torch.exp(q_evals_agent_logit - torch.logsumexp(q_evals_agent_logit, dim=-1, keepdim=True))
        #q_evals_average_logit = torch.exp(q_evals_average_logit - torch.logsumexp(q_evals_average_logit, dim=-1, keepdim=True))
        
        
        policy_intrinsic_reward = self.dyn_model_config.intr_beta2 * (q_evals_agent_logit * (q_evals_agent_logit / q_evals_average_logit).log()).sum(dim=-1) 
        policy_intrinsic_reward = policy_intrinsic_reward.mean(dim=-1) # shape=(batch_size, max_episode_len)
    
       
        # Compute reward from log prior from dynamics model
        B, length, N, dim = batch_o.shape
        data_dict = {
            'vector_obs':  batch_o.permute(0,2,1,3).reshape(-1, length, dim).clone()
        }
        embed = self.target_wm.encoder(data_dict)
        embed_role = self.target_wm_role.encoder(data_dict)
        state_t = None
        state_role_t = None
        
        dreamer_intrinsic = torch.zeros((B, max_episode_len)).to(self.device)
        obs_intrinsic = torch.zeros((B, max_episode_len)).to(self.device)
        for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
            # Predict
            batch_last_a_t = batch_last_a[:, t].reshape(-1, self.action_dim)
            batch_a_t = batch_last_a[:, t+1].reshape(-1, self.action_dim)
            # role_embeddings_t = role_embeddings[:, t].reshape(-1, self.role_embedding_dim)
            if t==0:
                role_embeddings_t = torch.zeros_like(role_embeddings[:, t].reshape(-1, self.role_embedding_dim))
                is_first = torch.ones((embed.shape[0]))
            else:
                role_embeddings_t = role_embeddings[:, t-1].reshape(-1, self.role_embedding_dim)
                is_first = torch.zeros((embed.shape[0]))
            
            state_t, _ = self.target_wm.dynamics.obs_step(
                 state_t, batch_last_a_t, embed[:,t], is_first
            )
            
            if self.dyn_model_config.rssm_role:
                state_role_t,_ = self.target_wm_role.dynamics.obs_step(
                    state_role_t, batch_last_a_t, embed_role[:,t], is_first, role_embeddings_t
                )
            else:
                state_role_t,_ = self.target_wm_role.dynamics.obs_step(
                    state_role_t, batch_last_a_t, embed_role[:,t], is_first
                )
                
            # state_pred_t = self.target_wm.dynamics.img_step(state_t, batch_a_t)
            # if self.dyn_model_config.rssm_role:
            #     state_pred_role_t = self.target_wm_role.dynamics.img_step(state_role_t,
            #                                                     batch_a_t, 
            #                                                    role_embeddings[:, t].reshape(-1, self.role_embedding_dim))
            # else:
            #     state_pred_role_t = self.target_wm_role.dynamics.img_step(state_role_t,
            #                                                     batch_a_t)

            preds = self.imagine(state_t, state_role_t, batch_last_a, role_embeddings, t, max_episode_len)

            state_pred_ts = preds['state']
            state_pred_role_ts = preds['state_role']
            pred_intrinsic_rew = 0.
            for state_pred_t, state_pred_role_t in zip(state_pred_ts, state_pred_role_ts):
                
                pred_intrinsic_rew += \
                    ( - self.target_wm.dynamics.get_dist(state_pred_t).log_prob(state_pred_t['stoch']) + \
                        self.dyn_model_config.intr_beta1 * self.target_wm_role.dynamics.get_dist(state_pred_role_t).log_prob(state_pred_role_t['stoch']))
            assert len(state_pred_ts) <= self.dyn_model_config.reward_imagine_steps
            pred_intrinsic_rew /= len(state_pred_ts)
            
            
            pred_intrinsic_rew = self.dyn_model_config.intr_beta3*pred_intrinsic_rew
            dreamer_intrinsic[:,t] = pred_intrinsic_rew.clone().reshape(B, N).mean(dim=-1)
            
            # pred_obs_t = self.target_wm.heads["decoder"](self.target_wm.dynamics.get_feat(state_pred_t))['vector_obs']
            # # TODO: Include the role embeddings
            # state_pred_role_feat_t = self.target_wm_role.dynamics.get_feat(state_pred_role_t)
            # if self.dyn_model_config.decode_role:
            #     state_pred_role_feat_t = torch.cat([state_pred_role_feat_t, role_embeddings[:, t].reshape(-1, self.role_embedding_dim)], dim=-1)    
            # pred_obs_role_t = self.target_wm_role.heads["decoder"](state_pred_role_feat_t)['vector_obs']
            
            obs_dim = batch_o.shape[-1]
            pred_intrinsic_rew = 0.
            for j, (pred_obs_t, pred_obs_role_t) in enumerate(zip(preds['obs'], preds['obs_role'])):
                
                pred_intrinsic_rew += \
                    (-pred_obs_t.log_prob(batch_o[:, t+j+1].reshape(-1, obs_dim), sum_dim = 1) \
                        + self.dyn_model_config.intr_beta1 * pred_obs_role_t.log_prob(batch_o[:, t+j+1].reshape(-1, obs_dim ), sum_dim = 1))
            assert len(preds['obs']) <= self.dyn_model_config.reward_imagine_steps
            pred_intrinsic_rew /= len(preds['obs'])
            obs_intrinsic[:,t] = pred_intrinsic_rew.clone().reshape(B, N).mean(dim=-1)
            
        prediction_intrinsic_reward = dreamer_intrinsic + obs_intrinsic 
        intrinsic_reward  = policy_intrinsic_reward + prediction_intrinsic_reward
        anneal_factor = 1.0
        if self.dyn_model_config.intrinsic_anneal:
            if total_steps < self.dyn_model_config.intrinsic_anneal_start:
                anneal_factor = 1.0
            else:
                anneal_factor = max(0.0, 1.0 - (total_steps - self.dyn_model_config.intrinsic_anneal_start) / self.dyn_model_config.intrinsic_anneal_steps)

        dreamer_intrinsic = dreamer_intrinsic *  anneal_factor
        policy_intrinsic_reward = policy_intrinsic_reward * anneal_factor
        obs_intrinsic = obs_intrinsic * anneal_factor
        intrinsic_reward = intrinsic_reward * anneal_factor

        rewards = {
            "dreamer_intrinsic_mean": to_np(dreamer_intrinsic.mean()),
            "obs_intrinsic_mean": to_np(obs_intrinsic.mean()),
            "policy_intrinsic_mean": to_np(policy_intrinsic_reward.mean()),
            "dreamer_intrinsic_max": to_np(dreamer_intrinsic.max()),
            "obs_intrinsic_max": to_np(obs_intrinsic.max()),
            "policy_intrinsic_min": to_np(policy_intrinsic_reward.min()),
            "obs_intrinsic_min": to_np(obs_intrinsic.min()),
            "dreamer_intrinsic_min": to_np(dreamer_intrinsic.min()),
            "dreamer_intrinsic_episode": to_np(dreamer_intrinsic.sum(dim = 1).mean()),
            "obs_intrinsic_episode": to_np(obs_intrinsic.sum(dim=1).mean()),
            "policy_intrinsic_episode": to_np(policy_intrinsic_reward.sum(dim=1).mean()),
            "intrinsic_reward_episode": to_np(intrinsic_reward.sum(dim=1).mean()),
            "intrinisc_reward_mean": to_np(intrinsic_reward.mean()),
            "intrinsic_reward_max": to_np(intrinsic_reward.max()),
            "intrinsic_reward_min": to_np(intrinsic_reward.min()),
            "anneal_factor": anneal_factor
        }
        
        return intrinsic_reward, rewards
        
    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def imagine(self, state_t, state_role_t, batch_act, role_embeddings, t, max_episode_len):
    
        preds = {
            'state':[],
            'state_role':[],
            'obs':[],
            'obs_role':[]
        }
        batch_a_t = batch_act[:, t+1].reshape(-1, self.action_dim)
        
        def singlestep(state, state_role, batch_a, role_embed_int):
            state = self.target_wm.dynamics.img_step(state, batch_a)
            if self.dyn_model_config.rssm_role:
                state_role = self.target_wm_role.dynamics.img_step(state_role,
                                                                batch_a, 
                                                               role_embed_int)
            else:
                state_role= self.target_wm_role.dynamics.img_step(state_role,
                                                                batch_a)
            obs = self.target_wm.heads["decoder"](self.target_wm.dynamics.get_feat(state))['vector_obs']
            
            state_role_feat = self.target_wm_role.dynamics.get_feat(state_role)
            if self.dyn_model_config.decode_role:
                state_role_feat = torch.cat([state_role_feat, role_embed_int], dim=-1)  
            obs_role = self.target_wm_role.heads["decoder"](state_role_feat)['vector_obs']
            
            return state, state_role, obs, obs_role
        # Single Step prediction 

        state_pred_t, state_pred_role_t = copy.deepcopy(state_t), copy.deepcopy(state_role_t)
            
        for j in range(0, self.dyn_model_config.reward_imagine_steps):
            
            # Predict action for new observation based 
            if t+j+1 > max_episode_len:
                break
            
            action = batch_act[:, t+j+1].reshape(-1, self.action_dim)
            role_embed = role_embeddings[:, t+j].reshape(-1, self.role_embedding_dim)

            state_pred_t, state_pred_role_t, obs, obs_role = singlestep(state_pred_t, state_pred_role_t, action, role_embed)

            preds['state'].append(copy.deepcopy(state_pred_t))
            preds['state_role'].append(copy.deepcopy(state_pred_role_t))
            preds['obs'].append(copy.deepcopy(obs))
            preds['obs_role'].append(copy.deepcopy(obs_role))
            
        return preds