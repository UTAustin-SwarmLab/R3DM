import torch
import numpy as np
from algorithm.vdn_qmix import VDN_QMIX
from algorithm.acorm import ACORM_Agent
from util.replay_buffer import ReplayBuffer

import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from algorithm.r3dm import R3DM_Agent
from util.tools import recursive_dict_update
import omegaconf
import os
from util.logger import Logger
class Runner:
    def __init__(self, args):
        self.args = args
        self.env_name = self.args.env_name
        self.seed = self.args.seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.embed = None
        # Create env
        self.text_env = False
        if 'text' in args.algorithm or 'TEXT' in args.algorithm:  
            from text_env import StarCraft2EnvText 
            self.env = StarCraft2EnvText(map_name=self.env_name, seed=self.seed)
            self.embed = TextEmbed2(args)
            self.text_env = True
        elif 'protoss' in self.env_name or 'terran' in self.env_name or 'zerg' in self.env_name:
            from env.smacv2.smacv2_env import SMACv2Env
            print("Using SMACv2Env")
            self.env = SMACv2Env(args = {'map_name': self.env_name})
            self.env.seed(self.seed)
        else:
            from smac.env import StarCraft2Env
            self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed)
            
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        self.args.obs_text_dim = 4096
        if 'text' in args.algorithm or 'TEXT' in args.algorithm:
            if args.agent_net == 'Agent_Embedding':
                args.agent_embedding_dim = self.args.obs_text_dim
                self.args.agent_embedding_dim = self.args.obs_text_dim
        
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        self.save_path =  args.save_path
        self.model_path = args.model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        if self.args.max_history is 0:
            self.args.max_history = None
    
        # Create N agents
        self.logger = Logger(args)
        if args.algorithm in ['QMIX', 'VDN']:
            path = os.path.abspath(os.path.dirname(__file__))
            extra_args = omegaconf.OmegaConf.load(os.path.join(path,"configs/qmix.yaml"))
            self.args.__dict__.update(extra_args)
            self.agent_n = VDN_QMIX(self.args)
        elif args.algorithm == 'ACORM':
            if args.agent_net != 'Agent_Embedding':
                raise NotImplementedError
            self.agent_n = ACORM_Agent(self.args)
        elif args.algorithm == 'ACORM_TEXT':
            self.agent_n = ACORM_TEXT_Agent(self.args)
        elif args.algorithm == 'ACORM_TEXT_INFER':
            self.agent_n = ACORM_TEXT_INFER_Agent(self.args)
        elif args.algorithm == 'R3DM':
            if args.agent_net != 'Agent_Embedding':
                raise NotImplementedError
            path = os.path.abspath(os.path.dirname(__file__))
            self.dyn_config = omegaconf.OmegaConf.load(os.path.join(path,"configs/dyn_config.yaml"))
            # Load environment specific intrinsic rewards and add that to dyn_config
            
            intrinsics = omegaconf.OmegaConf.load(os.path.join(path,"configs/{}.yaml".format(self.env_name)))
            # Update the dyn_config with the environment specific intrinsic rewards
            if self.args.intrinsic_reward:
                self.dyn_config = recursive_dict_update(self.dyn_config, intrinsics.intrinsic)
            else:
                self.dyn_config = recursive_dict_update(self.dyn_config, intrinsics.backprop)
            # define recursive_dict_update

            self.agent_n =  R3DM_Agent(self.args, self.dyn_config, self.env_info)
            self.logger = Logger(args, self.dyn_config)
            
        self.replay_buffer = ReplayBuffer(self.args, self.args.buffer_size)

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.evaluate_reward = []
        self.total_steps = 0
        self.agent_embed_pretrain_epoch, self.recl_pretrain_epoch = 0, 0
        self.pretrain_agent_embed_loss, self.pretrain_recl_loss = [], []
        if 'ACORM' in args.algorithm or 'R3DM' in args.algorithm:
            if args.algorithm =='ACORM_TEXT':
                self.args.agent_embed_pretrain_epochs = 0 
            else:
                self.args.agent_embed_pretrain_epochs = 120 #120
            self.args.recl_pretrain_epochs = 100 #100
        else:
            self.args.agent_embed_pretrain_epochs = 0 
            self.args.recl_pretrain_epochs = 0

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode
            
            if self.agent_embed_pretrain_epoch < self.args.agent_embed_pretrain_epochs:
                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent_embed_pretrain_epoch += 1
                    agent_embedding_loss = self.agent_n.pretrain_agent_embedding(self.replay_buffer)
                    self.pretrain_agent_embed_loss.append(agent_embedding_loss.item())
                    self.logger.log(
                        {
                         'pretrain/agent_embed_loss': agent_embedding_loss.item()
                         },
                        step = self.agent_embed_pretrain_epoch
                    )
                    if self.args.algorithm == "R3DM":
                        for _ in range(self.dyn_config.iterations):
                            m = self.agent_n.train_world_models(replay_buffer=self.replay_buffer, soft_update=False)
                            self.logger.log(m, self.agent_embed_pretrain_epoch)
                    
            else:
                if self.recl_pretrain_epoch < self.args.recl_pretrain_epochs:
                    self.recl_pretrain_epoch += 1
                    recl_loss = self.agent_n.pretrain_recl(self.replay_buffer)
                    self.pretrain_recl_loss.append(recl_loss)
                    if self.args.algorithm == "R3DM":
                        for _ in range(self.dyn_config.iterations):
                            self.agent_n.train_world_models(replay_buffer=self.replay_buffer)
                            self.logger.log(m, self.agent_embed_pretrain_epoch + self.recl_pretrain_epoch)

                    self.logger.log(
                        {'pretrain/recl_loss': recl_loss,
                         },
                        step = self.agent_embed_pretrain_epoch + self.recl_pretrain_epoch
                    )
                else:                                                          
                    self.total_steps += episode_steps
                    if self.replay_buffer.current_size >= self.args.batch_size:
                        metrics = self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                        self.logger.log(metrics, self.agent_embed_pretrain_epoch + self.recl_pretrain_epoch + self.total_steps)
                                
            print("total_steps:{} \t episode_steps:{}".format(self.total_steps, episode_steps))   
        self.evaluate_policy()
         # save model
        model_path = f'{self.model_path}/{self.env_name}_seed{self.seed}_'
        torch.save(self.agent_n.eval_Q_net, model_path + 'q_net.pth')
        torch.save(self.agent_n.RECL.role_embedding_net, model_path + 'role_net.pth')
        torch.save(self.agent_n.RECL.agent_embedding_net, model_path+'agent_embed_net.pth')
        torch.save(self.agent_n.eval_mix_net.attention_net, model_path+'attention_net.pth')
        torch.save(self.agent_n.eval_mix_net, model_path+'mix_net.pth')
        if self.embed is not None:   
            self.embed.cleanup()
        self.env.close()
        self.logger.cleanup()

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        for eval in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True, eval_id = eval)
            
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        self.evaluate_reward.append(evaluate_reward)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        
        if self.args.tb_plot or self.args.wandb:
            self.logger.log(
                {
                    'win_rate': win_rate,
                    'eval_reward': evaluate_reward,
                    'step': self.total_steps + self.agent_embed_pretrain_epoch + self.recl_pretrain_epoch 
                }, 
                self.total_steps + self.agent_embed_pretrain_epoch + self.recl_pretrain_epoch 
            )
        if self.args.sns_plot:
            # # plot curve
            sns.set_style('whitegrid')
            plt.figure()
            x_step = np.array(range(len(self.win_rates)))
            ax = sns.lineplot(x=x_step, y=np.array(self.win_rates).flatten(), label=self.args.algorithm)
            plt.ylabel('win_rates', fontsize=14)
            plt.xlabel(f'step*{self.args.evaluate_freq}', fontsize=14)
            plt.title(f'{self.args.algorithm} on {self.env_name}')
            plt.savefig(f'{self.save_path}/{self.env_name}_seed{self.seed}.jpg')

            # Save the win rates
            np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}.npy', np.array(self.win_rates))
            np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}_return.npy', np.array(self.evaluate_reward))
        
    def run_episode_smac(self, evaluate=False, eval_id = None):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        
        self.agent_n.eval_Q_net.rnn_hidden = None
        if self.args.algorithm == 'ACORM' or self.args.algorithm == 'ACORM_TEXT_INFER' or self.args.algorithm == 'R3DM':
            self.agent_n.RECL.agent_embedding_net.rnn_hidden = None
        

        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim)) # Last actions of N agents(one-hot)
        a_n = np.zeros(self.args.N)  # Actions of N agents
        if self.text_env:
            all_text_observations = []
            observation_history = []
        if self.env.render:
            frames = []
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
                
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            
            merged_observation = None
            if self.text_env:
                obs_text = self.env.get_obs_text(short_text=self.args.short_text)
                act_text = self.env.convert_action_text(a_n)
                observation_history, merged_observation = self.merge_all_observation(observation_history, obs_text, act_text, episode_step, max_history=self.args.max_history)
                    
                all_text_observations.append(merged_observation)
                
                
            if self.args.algorithm == 'ACORM' or self.args.algorithm == 'ACORM_TEXT_INFER' or self.args.algorithm == 'R3DM':
                role_embedding = self.agent_n.get_role_embedding(obs_n, last_onehot_a_n)
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon)
            elif self.args.algorithm == 'ACORM_TEXT':
                agent_embedding = self.embed.forward(merged_observation).numpy()
                role_embedding = self.agent_n.get_role_embedding(obs_n, last_onehot_a_n, agent_embedding)
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon)
            else:
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)

            r, done, info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r
            if self.args.render and eval_id is not None and eval_id==0:
                frame = self.env.render(mode='rgb_array')
                frames.append(frame)
                
            if not evaluate:
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                if self.args.algorithm == 'ACORM_TEXT':
                    self.replay_buffer.store_transition(episode_step, obs_n, s,
                                                        avail_a_n, last_onehot_a_n,
                                                        a_n, r, dw, obs_n_text=agent_embedding)
                else:
                    self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
                # obs_a_n_buffer[episode_step] = obs_n
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break
        
        if self.args.render and eval_id is not None and eval_id==0:
                self.logger.log_video(frames, step=self.total_steps + self.agent_embed_pretrain_epoch + self.recl_pretrain_epoch )
        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            if self.text_env:
                obs_text = self.env.get_obs_text(short_text=self.args.short_text)
                act_text = self.env.convert_action_text([0]*self.args.N)
                observation_history, merged_observation = self.merge_all_observation(observation_history, obs_text, act_text, episode_step, max_history=self.args.max_history)

                all_text_observations.append(merged_observation)
                
            if self.args.algorithm == 'ACORM_TEXT':
                agent_embedding = self.embed.forward(merged_observation).numpy()  
            elif self.args.algorithm == 'ACORM_TEXT_INFER':
                num_steps = len(all_text_observations)
                agents = len(all_text_observations[0])
                
                all_text_observations = [item for sublist in all_text_observations for item in sublist]
                instruction = self.env.instruction_text(short_text=self.args.short_text)
                for i in range(len(all_text_observations)): 
                    all_text_observations[i] = [instruction, all_text_observations[i]]
                    
                obs_text_embed = self.embed.forward(all_text_observations).numpy()
                
                obs_text_embed = np.reshape(obs_text_embed, (num_steps, agents, -1))
                self.replay_buffer.store_obs_n_text(obs_text_embed)
            
            if self.args.algorithm == 'ACORM_TEXT':
                self.replay_buffer.store_last_step(episode_step + 1, obs_n, 
                                               s, avail_a_n,
                                               last_onehot_a_n, 
                                               agent_embedding)
            else:
                self.replay_buffer.store_last_step(episode_step + 1, obs_n, 
                                                s, avail_a_n,
                                                last_onehot_a_n)
            

        return win_tag, episode_reward, episode_step+1
    
    def merge_all_observation(self, previous_obs, current_obs, action, timestep=0, max_history=None):
        observation = []
        text_step = f"This is time step {timestep}. "

        if len(previous_obs) == 0:
            previous_obs = []
            for curr, act in zip(current_obs, action):
                observation.append(text_step + curr + " " + act +"\n")
                previous_obs.append([text_step + curr + " " + act + "\n"])
            return previous_obs, observation
        else:
            observation = []
            
            for j, (prev, curr, act) in enumerate(zip(previous_obs, current_obs, action)):
                text = text_step + curr + " " + act + "\n"
                if max_history is not None and len(previous_obs[j]) > max_history:
                    previous_obs[j] = previous_obs[j][1:]
                previous_obs[j].append(text)
                
                combined_text = ""
                for obs in previous_obs[j]:
                    combined_text += obs
                observation.append(combined_text)
                  
                
            return previous_obs,observation
