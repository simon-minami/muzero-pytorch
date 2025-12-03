import torch
import torch.nn as nn

from core.model import BaseMuZeroNet
#TODO change representation to CNN to handl image input
# to start, use same simple encoder architecute that has worked for visual model-free rl tasks
# TODO: want the number of input channels of first conv layer to be 3*self.k where k is number of obs to stack
# for now just assume 3 channel input
# https://arxiv.org/pdf/2004.13649
# RIGHT NOW everything is cpu assumed, might want to try making device agnostic in the future
"""
We employ an encoder architecture from [ 60 ]. This encoder consists of four convolutional layers
with 3 × 3 kernels and 32 channels. The ReLU activation is applied after each conv layer. We use
stride to 1 everywhere, except of the first conv layer, which has stride 2. The output of the convnet
is feed into a single fully-connected layer normalized by LayerNorm [ 3]. Finally, we apply tanh
nonlinearity to the 50 dimensional output of the fully-connected layer.
"""
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 32,32,32 assuming 3,64,64 input
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32**3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        # print(f'shape after conv: {x.shape}')
        x = x.flatten(start_dim=1)  # don' want to flatten batch dim
        # print(f'shape after flatten: {x.shape}')
        return self.fc(x)
        

class MuZeroNet(BaseMuZeroNet):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size,
                 inverse_value_transform, inverse_reward_transform):
        super().__init__(inverse_value_transform, inverse_reward_transform)
        self.hx_size = 32
        # self._representation = nn.Sequential(nn.Linear(input_size, self.hx_size),
        #                                      nn.Tanh())

        self._representation = Encoder(in_channels=3, hidden_dim=self.hx_size)
        
        self._dynamics_state = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                             nn.Tanh(),
                                             nn.Linear(64, self.hx_size),
                                             nn.Tanh())
        self._dynamics_reward = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                              nn.LeakyReLU(),
                                              nn.Linear(64, reward_support_size))
        self._prediction_actor = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, action_space_n))
        self._prediction_value = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, value_support_size))
        self.action_space_n = action_space_n

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        return self._representation(obs_history)

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)
        return next_state, reward
