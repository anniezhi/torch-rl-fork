import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, world_size, use_memory=False, use_text=False):
        super().__init__()      #inherent all methods and properties from parent class

        # Decide which components are enabled
        self.world_n = world_size[0]
        self.world_m = world_size[1]
        self.use_text = use_text
        self.use_memory = use_memory
        vision_n = obs_space["image"][0]
        vision_m = obs_space["image"][1]

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2), padding=((self.world_n-2-vision_n)//2, (self.world_m-2-vision_m)//2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.image_embedding_size = ((self.world_n-2-1)//2-2)*((self.world_m-2-1)//2-2)*64
        # self.image_embedding_size = 1*64  # whatever the view range is, make the final embedding size to 1*64

        self.goal_mlp = nn.Sequential(
            nn.Linear(self.world_m + self.world_n, self.image_embedding_size//2), #(18,32)
            nn.ReLU(),
            nn.Linear(self.image_embedding_size//2,self.image_embedding_size),     #(32,64)
            nn.ReLU()
        )

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        self.embedding_size += self.image_embedding_size   # allow for goal embedding
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, goal, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        goal = goal.repeat([x.shape[0],1]).type(torch.FloatTensor)

        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)   #x.shape=[batch_size,channel_number]

        goal_after = self.goal_mlp(goal)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        embedding = torch.cat((embedding, goal_after), dim=1)

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
