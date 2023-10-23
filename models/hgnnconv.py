import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class GNN_LEP(torch.nn.Module):

    def __init__(self, num_features, hidden_dim):
        super(GNN_LEP, self).__init__()
        self.conv1 = HypergraphConv(num_features, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim * 2)
        # self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return x

class HGNNConv(nn.Module):
    #Hypergraph Convolutional Layer
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 attention=False,
                 heads=1):
        super(HGNNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv1 = HypergraphConv(self.in_channels,
                                    self.out_channels,
                                    use_attention=attention,
                                    heads=heads,
                                    dropout=self.dropout)
        self.conv2 = HypergraphConv(self.out_channels,
                                    self.out_channels,
                                    use_attention=attention,
                                    heads=heads,
                                    dropout=self.dropout)
        self.conv3 = HypergraphConv(self.out_channels,
                                    self.out_channels,
                                    use_attention=attention,
                                    heads=heads,
                                    dropout=self.dropout)

    def forward(self, x, hyperedge_index, hyperedge_weight, hyperedge_attr,
                batch):
        # print(x.shape)
        x = F.relu(
            self.conv1(x, hyperedge_index, hyperedge_weight, hyperedge_attr))
        # print(x.shape)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(
            self.conv2(x, hyperedge_index, hyperedge_weight, hyperedge_attr))
        # print(x.shape)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(
            self.conv3(x, hyperedge_index, hyperedge_weight, hyperedge_attr))
        # print(x.shape)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = global_mean_pool(x, batch)
        return x

def mish(x):
    return x * torch.tanh(F.softplus(x))
