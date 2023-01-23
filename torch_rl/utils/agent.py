import torch
import torch.nn.functional as F

import numpy as np

from .format import *
from .storage import *


from .other import device
from ..model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, world_size, goal,
                 model_dir, argmax=False, num_envs=1, use_memory=False, use_text=False, whole_view=False):
        obs_space, self.preprocess_obss = get_obss_preprocessor(obs_space)
        if type(goal) is list:
            def goal_onehot(x):
                return F.one_hot(torch.tensor(x), num_classes=world_size[0])
            self.goal = torch.flatten(torch.sum(torch.stack(list(map(goal_onehot, goal))), dim=0)).type(torch.FloatTensor)
        else:
            self.goal = torch.flatten(F.one_hot(torch.tensor(goal), num_classes=world_size[0])).type(torch.FloatTensor)
        self.acmodel = ACModel(obs_space, action_space, world_size, use_memory=use_memory, use_text=use_text, whole_view=whole_view)
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

        self.acmodel.load_state_dict(get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.goal, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)
        dist, dist_scale = dist

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
            actions_scale = dist_scale.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()
            actions_scale = dist_scale.sample()

        return actions.cpu().numpy(), np.clip(actions_scale.cpu(),0,1)

    def get_action(self, obs):
        return (self.get_actions([obs])[0][0], self.get_actions([obs])[1][0])

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
