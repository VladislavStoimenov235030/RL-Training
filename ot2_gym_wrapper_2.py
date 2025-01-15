import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # Action space: 3 continuous values representing movements along x, y, z within [-2, 2]
        self.action_space = spaces.Box(low=-1.2, high=1.2, shape=(3,), dtype=np.float32)

        # Observation space: 6 continuous values representing pipette position and goal position
        self.observation_space = spaces.Box(low=np.array([-1.927, -1.9105, -1.8805, -1.927, -1.9105, -1.8805]),
                                            high=np.array([2.073, 2.0895, 2.1195, 2.073, 2.0895, 2.1195]),
                                            dtype=np.float32)

        # Keep track of the number of steps
        self.steps = 0
        self.previous_distance = None

    def reset(self, seed=None):
        # Being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position for the agent within the working envelope
        self.goal_position = np.random.uniform(low=[-1.927, -1.9105, -1.8805],
                                               high=[2.073, 2.0895, 2.1195], size=3)

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        # Process the observation: get pipette position and append the goal position
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Reset the number of steps
        self.steps = 0

        # Initialize the previous distance for directional reward
        self.previous_distance = np.linalg.norm(pipette_position - self.goal_position)

        return observation, {}
        
        
    def step(self, action):
        # Execute one time step within the environment
        # Append 0 for the drop action since we only control position
        action = np.append(action, 0.0)

        # Call the environment step function
        observation = self.sim.run([action])  # Pass the action as a list

        # Process the observation: get pipette position and append the goal position
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Calculate the distance to the goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        
        reward = -distance_to_goal * 0.5

        # Check if the task is complete (distance below a threshold)
        if distance_to_goal < 0.001:  # Threshold based on pipette tip size
            terminated = True
            reward += 50  # Positive reward for completing the task
        else:
            terminated = False

        # Check if the episode should be truncated (max steps exceeded)
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {}  # No additional information

        # Increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()