from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import time
from ot2_gym_wrapper_2 import OT2Env
import os 
from clearml import Task
import argparse
import wandb

task = Task.init(
    project_name='Mentor Group D/Group 2',
    task_name='Experiment_3_235030'
)
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name='default')

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_steps', type=int, default=4096)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--timesteps', type=int, default=1000000)
parser.add_argument("--discount_factor", type=float, default=0.99)
args = parser.parse_args()

os.environ['WANDB_API_KEY']='1'
run = wandb.init(project='Experiment3', sync_tensorboard=True)



env = OT2Env(render=False, max_steps=1000)

model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            discount_factor = args.discount_factor,
            tensorboard_log=f"runs/{run.id}",)

wandb_callback = WandbCallback(
    model_save_freq=10000,
    model_save_path=f'models/{run.id}',
    verbose=2
)

time_steps = 100000


time_steps = 100000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{time_steps*(i+1)}")

# Final save
model.save(f"models/{run.id}/final_model")
env.close()

# End the Wandb run
wandb.finish()