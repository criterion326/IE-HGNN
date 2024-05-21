from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from models.cross_att import CrossAttentionLayer
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class ReadoutModule(torch.nn.Module):

    def __init__(self, nhid):
        """
        :param args: Arguments object.
        """
        super(ReadoutModule, self).__init__()
        # self.args = args

        self.weight = torch.nn.Parameter(torch.Tensor(nhid, nhid))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, batch):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param batch: Batch vector, which assigns each node to a specific example
        :param size: Size
        :return representation: A graph level representation matrix.
        """
        mean_pool = global_mean_pool(x, batch)
        transformed_global = torch.tanh(torch.mm(mean_pool, self.weight))
        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return global_add_pool(weighted, batch)


class MLPModule(torch.nn.Module):

    def __init__(self, nhid, dropout=0.1):
        super(MLPModule, self).__init__()
        # self.args = args
        self.dropout = dropout
        self.lin0 = torch.nn.Linear(nhid * 2 * 2, nhid * 2)
        nn.init.xavier_uniform_(self.lin0.weight.data)
        nn.init.zeros_(self.lin0.bias.data)

        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        nn.init.xavier_uniform_(self.lin1.weight.data)
        # nn.init.zeros_(self.lin1.bias.data)

        self.lin2 = torch.nn.Linear(nhid, nhid // 2)
        nn.init.xavier_uniform_(self.lin2.weight.data)
        # nn.init.zeros_(self.lin2.bias.data)

        self.lin3 = torch.nn.Linear(nhid // 2, 1)
        nn.init.xavier_uniform_(self.lin3.weight.data)
        # nn.init.zeros_(self.lin3.bias.data)

    def forward(self, scores):
        scores = F.relu(self.lin0(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.relu(self.lin1(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = F.relu(self.lin2(scores))
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        # scores = torch.sigmoid(self.lin3(scores)).view(-1)
        scores = self.lin3(scores)
        return scores


class IEHGNN(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gnn1 = HypergraphConv(in_channels=18,
                                   out_channels=hidden_dim,
                                   dropout=0.1)
        self.gnn2 = HypergraphConv(in_channels=43,
                                   out_channels=hidden_dim,
                                   dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.interaction1 = CrossAttentionLayer(hidden_dim, hidden_dim)
        self.interaction2 = CrossAttentionLayer(hidden_dim, hidden_dim)
        self.readout0 = ReadoutModule(hidden_dim)
        self.readout1 = ReadoutModule(hidden_dim)
        self.mlp = MLPModule(hidden_dim, dropout=0.1)

    def forward(self, g1, g2):
        h1 = self.gnn1(g1.x, g1.edge_index, g1.edge_weight, g1.edge_attr,
                       g1.batch)
        h2 = self.gnn2(g2.x, g2.edge_index, g2.edge_weight, g2.edge_attr,
                       g2.batch)
        att_f1_conv0 = self.readout0(h1, g1.batch)
        att_f2_conv0 = self.readout0(h2, g2.batch)
        score0 = torch.cat([att_f1_conv0, att_f2_conv0], dim=1)
        h3, h4 = self.interaction1(h1, g1.batch, h2, g2.batch)
        att_f1_conv1 = self.readout1(h3, g1.batch)
        att_f2_conv1 = self.readout1(h4, g2.batch)
        score1 = torch.cat([att_f1_conv1, att_f2_conv1], dim=1)
        h5, h6 = self.interaction2(h3, g1.batch, h4, g2.batch)
        att_f1_conv2 = self.readout1(h5, g1.batch)
        att_f2_conv2 = self.readout1(h6, g2.batch)
        score2 = torch.cat([att_f1_conv2, att_f2_conv2], dim=1)
        scores = torch.cat([score0, score1, score2], dim=1)
        x = self.mlp(scores)
        return x.view(-1)


class GNN_LEP(torch.nn.Module):

    def __init__(self, num_features, hidden_dim):
        super(GNN_LEP, self).__init__()
        self.conv1 = HypergraphConv(num_features, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return x


class MLP_LEP(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(MLP_LEP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x).view(-1)
