import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
from modified_gym_envs.ModifiedHalfCheetah import HalfCheetahEnv
import os

class DynamicFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, initial_map="4x4", dynamic_map="8x8", switch_after=1000, **kwargs):
        """
        Custom FrozenLake environment that changes its map layout after being called a fixed number of times.
        :param initial_map: The initial map layout (can be a gym map name like '4x4' or '8x8').
        :param dynamic_map: The map layout to switch to after `switch_after` steps.
        :param switch_after: Number of steps/episodes after which the map changes.
        """
        # Load the initial map
        super().__init__(map_name=initial_map, **kwargs)
        self.initial_map = initial_map
        self.dynamic_map = dynamic_map
        self.switch_after = switch_after
        self.steps_taken = 0
        self.current_map = initial_map
        self.init_kwargs = kwargs
    
    def step(self, action):
        # Call the step function of the parent class
        obs, reward, done, truncated, info = super().step(action)
        

        # Increment the steps counter
        self.steps_taken += 1

        # Check if we should switch the map layout
        if self.steps_taken >= self.switch_after and self.current_map != self.dynamic_map:
            print(f"Switching environment map after {self.steps_taken} steps.")
            self.switch_map_layout(self.dynamic_map)

        return obs, reward, done, truncated, info

    def reset(self,
        *,
        seed= None,
        options = None,):
        # Reset the environment as usual, increment episode count if needed
        if self.steps_taken >= self.switch_after and self.current_map != self.dynamic_map:
            print(f"Switching environment map after {self.steps_taken} steps.")
            self.switch_map_layout(self.dynamic_map)

        return super().reset(seed=seed, options=options)

    def switch_map_layout(self, new_map):
        """
        Switch the map layout to a new one.
        :param new_map: The map to switch to (e.g., '8x8').
        """
        if new_map == "4x4":
            new_desc = [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
            ]
        elif new_map == "8x8":
            new_desc = [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFHFFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG"
            ]
        else:
            raise ValueError("Unsupported map layout")
        
        # Switch the map layout and reset environment state
        # super().__init__(desc=new_desc, map_name=new_map, **self.init_kwargs)
        self.desc = new_desc
        self.desc = np.asarray(self.desc, dtype='c')
        self.nrow, self.ncol = self.desc.shape
        self.current_map = new_map
        self.initial_state_distrib = np.array(self.desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        print(self.desc)

class DynamicHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, switch_after=10000, render_mode=None, xml_file=None):


        self.switch_after = switch_after
        self.switches = 0
        self.steps_taken = 0
        self.xml_files = ["half_cheetah.xml", xml_file]
        super().__init__(
            xml_file=self.xml_files[0],
            render_mode=render_mode
        )

    def step(self, action):
        # Take a step in the current environment
        obs, reward, done, truncated, info = super().step(action)

        # Increment the steps counter
        self.steps_taken += 1

        # Check if we should switch the dynamics
        if self.steps_taken % self.switch_after == 0:
            print(f"Switching environment dynamics after {self.steps_taken} steps.")
            self.switch_dynamics()

        return obs, reward, done, truncated, info

    def switch_dynamics(self):
        """
        Switch the environment dynamics.
        """
        self.switches += 1
        self.xml_file = self.xml_files[self.switches % len(self.xml_files)]
        super().__init__(
            xml_file=self.xml_files[self.switches % len(self.xml_files)],
            render_mode=self.render_mode
        )
        super().reset()

    def __str__(self):
        return 'modified_half_cheetah environment'

    __credits__ = ["Rushiv Arora"]



class DynamicEnv(gym.Env):
    def __init__(self, env_list, switch_after=1000, **kwargs):
        """
        Custom Mujoco environment that changes its dynamics after being called a fixed number of times.
        :param env_list: List of Mujoco environments.
        :param switch_after: Number of steps/episodes after which the dynamics change.
        """
        super(DynamicEnv, self).__init__()
        self.env_list = env_list
        self.current_env_index = 0
        self.current_env = self.env_list[self.current_env_index](**kwargs)
        self.switch_after = switch_after
        self.steps_taken = 0

        # Set the observation and action spaces to match the current environment
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space

    def step(self, action):
        # Take a step in the current environment
        obs, reward, done, truncated, info = self.current_env.step(action)

        # Increment the steps counter
        self.steps_taken += 1

        # Check if we should switch the dynamics
        if self.steps_taken % self.switch_after == 0:
            print(f"Switching environment dynamics after {self.steps_taken} steps.")
            self.switch_dynamics()

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        # Reset the environment as usual, increment episode count if needed
        if self.steps_taken >= self.switch_after:
            print(f"Switching environment dynamics after {self.steps_taken} steps.")
            self.switch_dynamics()

        return self.current_env.reset()

    def switch_dynamics(self):
        """
        Switch the environment dynamics.
        """
        self.current_env_index = (self.current_env_index + 1) % len(self.env_list)
        self.current_env = self.env_list[self.current_env_index]()
        self.current_env.reset()

        # Update the observation and action spaces to match the new environment
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space

    def render(self, mode='human'):
        return self.current_env.render(mode)

    def close(self):
        return self.current_env.close()