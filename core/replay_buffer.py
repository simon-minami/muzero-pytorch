import numpy as np
import ray
import torch


@ray.remote
class ReplayBuffer(object):
    """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
    Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
    """

    def __init__(self, capacity, batch_size, prob_alpha=1):
        self.soft_capacity = capacity
        self.batch_size = batch_size

        self.buffer = []
        self.priorities = []
        self.game_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self.prob_alpha = prob_alpha

    def save_game(self, game, priorities=None):
        if priorities is None:
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(len(game))]))
        else:
            assert len(game) == len(priorities), " priorities should be of same length as the game steps"
            self.priorities = np.concatenate((self.priorities, priorities))
        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]
        self._eps_collected += 1

    def sample_batch(self, num_unroll_steps: int, td_steps: int, beta: float = 1, model=None, config=None):
        obs_batch, action_batch, reward_batch, value_batch, policy_batch = [], [], [], [], []

        probs = np.array(self.priorities) ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.priorities), self.batch_size, p=probs)

        total = len(self.priorities)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        indices = torch.tensor(indices)
        weights = torch.tensor(weights).float()

        for idx in indices:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]
            _actions = game.history[game_pos:game_pos + num_unroll_steps]
            # random action selection to complete num_unroll_steps
            _actions += [np.random.randint(0, game.action_space_size)
                         for _ in range(num_unroll_steps - len(_actions))]

            obs_batch.append(game.obs(game_pos))
            action_batch.append(_actions)
            value, reward, policy = game.make_target(game_pos, num_unroll_steps, td_steps, model, config)
            reward_batch.append(reward)
            value_batch.append(value)
            policy_batch.append(policy)
        
        # print(obs_batch[0].shape, action_batch[0].shape, reward_batch[0].shape, value_batch[0].shape, policy_batch[0].shape)
        # honestly just need to check types of this stuff
        # list of tensors? 
        # np array of _
        # what SHOULD it be.
        # then figure out how to fix it.
        #debug
        #TODO something wrong with make target, value batch, getting mixed types somehow (tensor and float?)
        #
        # for i, v in enumerate(value_batch):
        #     try:
        #         print(i, np.array(v).shape)
        #     except:
        #         print(i, "ERROR", v)
        
        # print(len(value_batch), value_batch[0], type(value_batch[0]))
        # each element in value_batch is list of tensors, should be length unroll steps +1
        # value batch itself is a list, len 128 (batch size)
        # in the end we want to return 128,6 tensor 
        # i think we were seing mixed data types as in sometimes element in value batch contained tensor and float
        # i think it makes more sense for value batch to be list of list of floats, not single item tensors
        # for b, v in enumerate(value_batch):
        #     try:
        #         print(b, np.array(v).shape)
        #     except:
        #         print(b, "ERROR", v)

        obs_batch = torch.tensor(np.array(obs_batch)).float()
        action_batch = torch.tensor(np.array(action_batch)).long()
        reward_batch = torch.tensor(np.array(reward_batch)).float()
        value_batch = torch.tensor(np.array(value_batch)).float()
        policy_batch = torch.tensor(np.array(policy_batch)).float()
        # print(obs_batch.shape, action_batch.shape, reward_batch.shape, value_batch.shape, policy_batch.shape)

        return obs_batch, action_batch, reward_batch, value_batch, policy_batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def remove_to_fit(self):
        if self.size() > self.soft_capacity:
            num_excess_games = self.size() - self.soft_capacity
            excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
            del self.buffer[:num_excess_games]
            self.priorities = self.priorities[excess_games_steps:]
            del self.game_look_up[:excess_games_steps]
            self.base_idx += num_excess_games

    def size(self):
        return len(self.buffer)

    def episodes_collected(self):
        return self._eps_collected

    def get_priorities(self):
        return self.priorities
