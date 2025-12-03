from collections import deque

import numpy as np
import cv2
from core.game import Game, Action


class CarRacingWrapper(Game):
    def __init__(self, env, k: int, discount: float):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        super().__init__(env, env.action_space.n, discount)
        self.k = k
        self.frames = deque([], maxlen=k)

    def legal_actions(self):
        return [Action(_) for _ in range(self.env.action_space.n)]

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        obs = self.preprocess(obs)
        done = terminal or truncated
        self.rewards.append(reward)
        self.history.append(action)
        self.obs_history.append(obs)

        # obs_history gets init with k obs
        # from there each time we step we add 1 to obs_history, rewards
        # so len(rewards) + k = len(obs_history)
        # so if k=1, and len(rewards) = 3, len(obs_history)=4
        # when we call self.obs(3), this grabs obs_history[3:4] which is just a single example
        # if k=3, we had len(rewards) = x, len(obs_history)=x+k
        # self.obs(len(rewards)) grabs obs_history[x: x+k]
        # ALWAYS grabs most recent self.k observations, which stacks just like we want!!!
        return self.obs(len(self.rewards)), reward, terminal, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.preprocess(obs)

        self.rewards = []
        self.history = []
        self.obs_history = []

        for _ in range(self.k):  # if k=1, no stacking, just append single obs
            # init obs_history with self.k obs so that we don't get error when we call obs(0)
            self.obs_history.append(obs)

        return self.obs(0), info

    def obs(self, i):
        #TODO review how stacking works, we want our conv encoder to be able to handle this.
        # 
        frames = self.obs_history[i:i + self.k]
        
        # each obs in car racing is 96, 96, 3
        # what shape should we return this as? where does it converted to tensor for input into encoder?
        # converted to tensor + batch dim added in train and test loops
        # so i think if we return c,h,w we're good
        # preprocessig (resizing etc) already handled when obs added to obs_history
        if self.k == 1:  # frames should just be single obs
            return frames[0]
        # if k>1 then we need to concat (stacking obs)
        return np.concatenate(frames, axis=2)

    def close(self):
        self.env.close()
        
    def preprocess(self, obs):
        # obs: (96,96,3), uint8
        obs = obs.astype(np.float32) / 255.0   # normalize
        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        # Convert HWC → CHW
        obs = np.transpose(obs, (2, 0, 1))
        return obs