import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO : Static features, futures ones

class MLPtime(nn.Module):

    def __init__(self, in_dim, dropout=0.3):
        super(MLPtime, self).__init__()
        self.batchnorm = TSBatchNorm2d()
        self.fc = nn.Linear(in_dim, in_dim) # ? in_dim or out_dim
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        out = self.batchnorm(x)
        out = out.permute(0, 2, 1)

        out = self.fc(out)
        out = F.relu(out)
        out = self.drop(out)
        out = out.permute(0, 2, 1)


        return out

class MLPfeat(nn.Module):

    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super(MLPfeat, self).__init__()
        self.batchnorm = TSBatchNorm2d()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, in_dim) # in_dim or out_dim, preserve generalization ?
        self.drop2 = nn.Dropout(dropout)


    def forward(self, x):
        out = self.batchnorm(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

        return out


class Mixer(nn.Module):
    def __init__(self, in_dim, feature_dim, hidden_dim, dropout=0.3):
        super(Mixer, self).__init__()
        #self.batchnorm = TSBatchNorm2d()
        self.time = MLPtime(in_dim, dropout)
        #self.batchnorm2 = TSBatchNorm2d()
        self.feat = MLPfeat(feature_dim, hidden_dim, dropout)

    def forward(self, x):
        res1 = x
        #out = self.batchnorm(x)
        #out = out.permute(0, 2, 1)
        out = self.time(x)
        #out = out.permute(0, 2, 1)
        out = out + res1
        res2 = out
        #out = self.batchnorm2(out)

        out = self.feat(out)
        out = out + res2

        return out

class TSMixer(nn.Module):
    def __init__(self, in_dim, feature_dim, hidden_dim, out_dim, n_mixers, dropout=0.3):
        super(TSMixer, self).__init__()
        self.RevIN = reversible_instance_norm(feature_dim)
        self.n_mixers = n_mixers
        self.mixers = nn.ModuleList([Mixer(in_dim, feature_dim, hidden_dim, dropout) for _ in range(n_mixers)])
        self.tempproj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.RevIN(x, 'normalize')
        for i in range(self.n_mixers):
            out = self.mixers[i](out)
        out = out.permute(0, 2, 1)
        out = self.tempproj(out)
        out = out.permute(0, 2, 1)
        out = self.RevIN(out, 'denormalize')
        return out

class reversible_instance_norm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.std = None


    def normalize(self, x):
        #dim = x.dim()
        mean = torch.mean(x, dim= 1, keepdim=True).detach() # dim ?
        std = torch.std(x, dim= 1, keepdim=True).detach()
        #print(x.shape)
        out = (x - mean) / (std + self.eps)
        self.mean = mean
        self.std = std
        out = out * self.scale + self.bias
        return out

    def denormalize(self, y):
        out = y - self.bias
        out = out / self.scale
        out = out * self.std + self.mean
        return out

    def forward(self, x, mode):
        if mode == 'normalize':
            return self.normalize(x)
        elif mode == 'denormalize':
            return self.denormalize(x)


class TSBatchNorm2d(nn.Module):

    def __init__(self):
        super(TSBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)

        # Reshape input_data to (batch_size, 1, timepoints, features)
        x = x.unsqueeze(1)
        output = self.bn(x)
        output = output.squeeze(1)
        return output

