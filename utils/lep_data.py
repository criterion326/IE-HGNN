import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
import ast
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from dataclasses import dataclass
import os
import time
from scipy.stats import spearmanr
from glob import glob
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score, auc
from sklearn.model_selection import KFold, train_test_split
# from Normalizer import normalizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform, PairedGraphTransform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Batch, DataLoader
import atom3d.util.graph as gr
import scipy
import torch
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from atom3d.datasets import LMDBDataset

def get_neighbors_by_distance(pos, r=3, limit=10):
    tree = scipy.spatial.cKDTree(pos)
    edges = tree.query_ball_point(pos, r=r, workers=1)  #最近邻还是阈值？？？
    keep = [0]
    used = edges[0]
    edge_weights = []
    dis = scipy.spatial.distance_matrix(pos[edges[0]], pos[edges[0]])
    edge_weights.append(1.0 / dis.mean() + 1e-5)
    for i in range(1, edges.shape[0]):
        temp, cnt = np.unique(used + edges[i], return_counts=True)
        if len(edges[i]) >= 2 and (
                cnt == 2).sum() <= limit:  #重复三个以内则还可以保留这条超边,efficiency?
            used = temp.tolist()
            keep.append(i)
            dis = scipy.spatial.distance_matrix(pos[edges[i]], pos[edges[i]])
            edge_weights.append(1.0 / dis.mean() + 1e-5)
    if len(edges) == 0:
        raise RuntimeError('len(edeges) is 0')
    return edges[keep], np.around(edge_weights, 6)


def get_hyperedges(edges, edge_weights):
    idx = []
    es = []
    ew = []
    for i, (e, w) in enumerate(zip(edges, edge_weights)):
        idx.extend([i] * len(e))
        ew.extend([w / len(e)] * len(e))
        es.extend(e)
    return torch.LongTensor([es, idx]), torch.FloatTensor(ew)

# PDB atom names -- these include co-crystallized metals
prot_atoms = [
    'C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO',
    'MG', 'CU', 'CL', 'SE', 'F'
]
# RDKit molecule atom names
mol_atoms = [
    'C',
    'N',
    'O',
    'S',
    'F',
    'Si',
    'P',
    'Cl',
    'Br',
    'Mg',
    'Na',
    'Ca',
    'Fe',
    'As',
    'Al',
    'I',
    'B',
    'V',
    'K',
    'Tl',
    'Yb',
    'Sb',
    'Sn',
    'Ag',
    'Pd',
    'Co',
    'Se',
    'Ti',
    'Zn',
    'H',  # H?
    'Li',
    'Ge',
    'Cu',
    'Au',
    'Ni',
    'Cd',
    'In',
    'Mn',
    'Zr',
    'Cr',
    'Pt',
    'Hg',
    'Pb'
]
# Residue names
residues = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR'
]

# below functions are adapted from DeepChem repository:
def one_of_k_encoding(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values."""
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def prot_df_to_graph(df,
                     feat_col='element',
                     allowable_feats=prot_atoms,
                     edge_dist_cutoff=4.5,label=None):
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
    node_feats = torch.FloatTensor(
        [one_of_k_encoding_unk(e, allowable_feats) for e in df[feat_col]])
    es, ew = get_hyperedges(
        *get_neighbors_by_distance(node_pos, r=edge_dist_cutoff, limit=10))
    g = Data(x=node_feats,
             edge_index=es,
                 edge_attr=ew,
                 y=torch.LongTensor(label))
    return g

def mol_df_to_graph(df,
                    allowable_atoms=None,
                    edge_dist_cutoff=4.5,
                    label=None):
    if allowable_atoms is None:
        allowable_atoms = mol_atoms
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())

    # kd_tree = ss.KDTree(node_pos)
    # edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    # edge_index = torch.LongTensor(edge_tuples).t().contiguous()
    # edge_index = to_undirected(edge_index)
    # edge_attr = torch.FloatTensor([
    #     1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5)
    #     for i, j in edge_index.t()
    # ]).view(-1)
    es, ew = get_hyperedges(
        *get_neighbors_by_distance(node_pos, r=edge_dist_cutoff, limit=10))
    node_feats = torch.FloatTensor(
        [one_of_k_encoding_unk(e, allowable_atoms) for e in df['element']])
    g = Data(x=node_feats,
             edge_index=es,
             edge_attr=ew,
             y=torch.LongTensor(label))
    return g



class My_dataset(Dataset):
    r'''
    自定义dataset.
    '''

    def __init__(self,idx,data):
        super().__init__()
        self.data = []
        for i in tqdm(idx, position=0):
            active = data[i]['atoms_active']
            inactive = data[i]['atoms_inactive']
            protein_active = active[active.chain != 'L']
            protein_inactive = inactive[inactive.chain != 'L']
            ligand_active = active[active.chain == 'L']
            ligand_inactive = inactive[inactive.chain == 'L']
            pro_active_hg=prot_df_to_graph(protein_active,label=np.array([1]))
            pro_inactive_hg = prot_df_to_graph(protein_inactive,
                                               label=np.array([0]))
            lig_active_hg = mol_df_to_graph(ligand_active, label=np.array([1]))
            lig_inactive_hg = mol_df_to_graph(ligand_inactive,
                                              label=np.array([0]))
            self.data.append((pro_active_hg,lig_active_hg))
            self.data.append((pro_inactive_hg,lig_inactive_hg))
        # self.data=list(eq_graph_tensors.values())
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CollaterLEP(object):

    def __init__(self):
        pass

    def __call__(self, data_list):
        # batch = Batch.from_data_list([d for d in data_list])
        batch1 = Batch.from_data_list([d[0] for d in data_list])
        batch2 = Batch.from_data_list([d[1] for d in data_list])
        return batch1, batch2
