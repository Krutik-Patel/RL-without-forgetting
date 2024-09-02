from gym.envs.toy_text import frozen_lake
import numpy as np
from gym import spaces

class FrozenLakeEnv(frozen_lake.FrozenLakeEnv):
    def __init__(
        self, 
        render_mode = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
    ):
        super().__init__(desc=desc, render_mode=render_mode, map_name=map_name, is_slippery=is_slippery)
        self.desc = desc
        # self.observation_space = self.observation_space
        self.state_without_agent = self.get_state_without_agent(desc)
        self.observation_space = spaces.Box(low=-1, high=100, shape=(self.observation_space.n,), dtype=np.int32)

    def get_state_without_agent(self, desc) -> np.ndarray:
        # the final chest is encoded as 100, holes are encoded as -1 and frozen lake is encoded as 0, the players position is encoded as 1
        # if it doens't work, maybe use mask for the agent position
        state = np.zeros(self.observation_space.n, dtype=np.int32)
        for i in range(len(desc)):
            for j in range(len(desc[0])):
                if desc[i][j] == 'F': state[i*len(desc[0]) + j] = 0
                elif desc[i][j] == 'H': state[i*len(desc[0]) + j] = -1
                elif desc[i][j] == 'G': state[i*len(desc[0]) + j] = 100
        return state

    def gen_state_for_agent_pos(self, agent_pos) -> np.ndarray:
        # the final chest is encoded as 100, holes are encoded as -1 and frozen lake is encoded as 0, the players position is encoded as 1
        # if it doens't work, maybe use mask for the agent position
        state = self.state_without_agent.copy()
        state[agent_pos] = 1
        return state

    def reset(self, *, seed = None, options = None):
        init_state, probs = super().reset(seed=seed, options=options)
        return self.gen_state_for_agent_pos(init_state), probs

    def step(self, action):
        step, reward, t, done, probs = super().step(action)
        new_state = self.gen_state_for_agent_pos(step)
        return new_state, reward, t, done, probs

    