from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
import gym
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
import os
from clearml import Task

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Mentor Group D/Group 2', # NB: Replace YourName with your own name
                    task_name='Experiment2_235030')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=4096)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--discount_factor", type=float, default=0.99)
args = parser.parse_args()

# Initialize WandB
os.environ["WANDB_IGNORE_SYMLINKS"] = "1"  # Avoid symlink issues on Windows
run = wandb.init(project='sb3_pendulum_demo',sync_tensorboard=True)

#env = gym.make('Pendulum-v1', g=9.81)
env = OT2Env(render=False, max_steps=1000)

model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)

wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# variable for how often to save the model
time_steps = 100000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{time_steps*(i+1)}")