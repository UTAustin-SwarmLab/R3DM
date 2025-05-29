import wandb
from tensorboardX import SummaryWriter
import numpy as np
import datetime

class Logger:
    
    def __init__(self, args, extra_args =  None) -> None:
        time_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.args = args
        if self.args.tb_plot:
            
            if args.algorithm == 'ACORM_WM' and extra_args is not None:
                self.writer = SummaryWriter(log_dir='./result/tb_logs/{}/{}/{}_seed_{}_{}_{}_[{},{},{},{}]'.format(self.args.algorithm, 
                                                                                                    self.args.env_name, self.args.env_name, self.args.seed,time_path, args.agent_net,
                                                                                                extra_args.intr_beta1, extra_args.intr_beta2, extra_args.intr_beta3, extra_args.intr_weight))
            else:
                self.writer = SummaryWriter(log_dir='./result/tb_logs/{}/{}/{}_seed_{}_{}_{}'.format(self.args.algorithm, 
                                                                                                self.args.env_name, self.args.env_name, self.args.seed,time_path, args.agent_net))           
            
            
        if self.args.wandb:
            wandb.init(project="acorm", 
                       group = '{}_{}{}'.format(self.args.env_name, self.args.algorithm, self.args.tag),
                       name = 'seed{}_date{}'.format(self.args.seed, time_path),
                       )
            # Add extra arguments to existing args under separate header
            wandb.config.update(vars(args))
            
            if extra_args is not None:
                wandb.config.update(vars(extra_args)['_content'], allow_val_change=True)
            
        
        self.step = 0
        self.episode = 0
    
    
    def log(self, log, step):
        if self.args.tb_plot:

            if isinstance(log, dict):
                for key, value in log.items():
                    self.writer.add_scalar(key, value, step)
            else:
                raise ValueError("log should be a dictionary")

        if self.args.wandb:
            wandb.log(log, step)
    
    def log_video(self, video, step):
        if self.args.wandb:
            video = [np.transpose(np.array(f), (2, 0, 1)) for f in video]
            video = np.stack(video, axis = 0)
            print
            wandb.log({"video": wandb.Video(video)}, step=step)
            
    def cleanup(self):
        if self.args.tb_plot:
            self.writer.close()
        if self.args.wandb:
            wandb.finish()