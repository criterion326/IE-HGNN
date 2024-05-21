import numpy as np
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
import scipy
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
import pickle
import dhg

atom_names = {'C': 0, 'H': 1, 'O': 2, 'N': 3, 'P': 4, 'S': 5}
prot_atoms = [
    'C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO',
    'MG', 'CU', 'CL', 'SE', 'F'
]
# RDKit molecule atom names
mol_atoms = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
]
lig_names = {v: i for i, v in enumerate(mol_atoms)}
pro_names = {v: i for i, v in enumerate(prot_atoms)}


def get_neighbors_by_distance(pos, r=4.5, limit=8):
    tree = scipy.spatial.cKDTree(pos)
    edges = tree.query_ball_point(pos, r=r, workers=1)
    keep = [0]
    used = edges[0]
    edge_weights = []
    dis = scipy.spatial.distance_matrix(pos[edges[0]], pos[edges[0]])
    edge_weights.append(1.0 / dis.mean() + 1e-5)
    for i in range(1, edges.shape[0]):
        temp, cnt = np.unique(used + edges[i], return_counts=True)
        if len(edges[i]) >= 2 and (cnt == 2).sum() <= limit:
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


def get_hyperedges3(hg):
    idx = []
    es = []
    ew = []
    for i, (e, w) in enumerate(
            zip(
                hg.e_of_group('p')[0] + hg.e_of_group('l')[0],
                hg.e_of_group('p')[1] + hg.e_of_group('l')[1])):
        idx.extend([i] * len(e))
        ew.extend([w / len(e)] * len(e))
        es.extend(e)
    return torch.LongTensor(
        [es, idx]), torch.FloatTensor(ew), torch.LongTensor(
            hg.e_of_group('ex')[0]).t_(), torch.FloatTensor(
                hg.e_of_group('ex')[1])


# PDB atom names -- torchese include co-crystallized metals
prot_atoms = [
    'C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO',
    'MG', 'CU', 'CL', 'SE', 'F'
]
# RDKit molecule atom names
mol_atoms = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
]
# Residue names
residues = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'torchR', 'VAL', 'TRP', 'TYR'
]


# below functions are adapted from DeepChem repository:
def one_of_k_encoding(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values."""
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in torche allowable set to torche last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def prot_df_to_graph(df,
                     feat_col='element',
                     allowable_feats=prot_atoms,
                     edge_dist_cutoff=4.5,
                     label=None):
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

    def __init__(self, idx, data):
        super().__init__()
        self.data = []
        for i in tqdm(idx, position=0):
            active = data[i]['atoms_active']
            inactive = data[i]['atoms_inactive']
            protein_active = active[active.chain != 'L']
            protein_inactive = inactive[inactive.chain != 'L']
            ligand_active = active[active.chain == 'L']
            ligand_inactive = inactive[inactive.chain == 'L']
            pro_active_hg = prot_df_to_graph(protein_active,
                                             label=np.array([1]))
            pro_inactive_hg = prot_df_to_graph(protein_inactive,
                                               label=np.array([0]))
            lig_active_hg = mol_df_to_graph(ligand_active, label=np.array([1]))
            lig_inactive_hg = mol_df_to_graph(ligand_inactive,
                                              label=np.array([0]))
            self.data.append((pro_active_hg, lig_active_hg))
            self.data.append((pro_inactive_hg, lig_inactive_hg))
        # self.data=list(eq_graph_tensors.values())
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class lep_dataset(Dataset):

    def __init__(self, items):
        super().__init__()
        self.data = []
        for item in tqdm(items):
            e1, ew1, e2, ew2 = get_hyperedges3(item['a_g'])
            a = Data(torch.Tensor(item['a_x']),
                     edge_index=e1,
                     ex_edge_index=e2,
                     edge_attr=ew1,
                     ex_edge_attr=ew2,
                     pos=torch.Tensor(item['a_pos']),
                     y=torch.from_numpy(np.array(item['label'])))
            e1, ew1, e2, ew2 = get_hyperedges3(item['ina_g'])
            ina = Data(torch.Tensor(item['ina_x']),
                       edge_index=e1,
                       ex_edge_index=e2,
                       edge_attr=ew1,
                       ex_edge_attr=ew2,
                       pos=torch.Tensor(item['ina_pos']),
                       y=torch.from_numpy(np.array(item['label'])))
            self.data.append([a, ina])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CollaterLEP(object):

    def __init__(self):
        pass

    def __call__(self, data_list):
        batch1 = Batch.from_data_list([d[0] for d in data_list])
        batch2 = Batch.from_data_list([d[1] for d in data_list])
        return batch1, batch2


def create_hypergraph(num_v: int, d: dict):
    g = dhg.Hypergraph(num_v=num_v)
    for k, v in d.items():
        if k == 'ex':
            # print('real??',v[1].tolist())
            g.add_hyperedges(e_list=v[0], e_weight=v[1].tolist(), group_name=k)
        else:
            g.add_hyperedges(e_list=v[0], e_weight=v[1], group_name=k)
    return g


def combine_two_graphs(pos1, pos2, r=3):
    tree1 = scipy.spatial.cKDTree(pos1)  #建议tree2是ligand的坐标
    tree2 = scipy.spatial.cKDTree(pos2)
    res = tree1.query_ball_tree(tree2, r=3)

    ex_edges = []
    ex_edge_weights = []
    for i, contacts in enumerate(res):
        if len(contacts) == 0:
            continue
        for j in contacts:
            ex_edges.append((i, j + pos1.shape[0]))  #因为是超图，不用对称
            d = 1.0 / (np.linalg.norm(pos1[i] - pos2[j]) + 1e-5)
            ex_edge_weights.append(d)
    ex_edge_weights = np.around(ex_edge_weights, 6)
    return ex_edges, ex_edge_weights
    # ex_edges[:4],ex_edge_weights[:4]


def main(begin, end):
    data = []
    for i in tqdm.tqdm(range(begin, end)):
        # for i in range(begin, end):
        item = {'label': None, 'smile': None}
        active = d['active'][i]
        inactive = d['inactive'][i]
        item['label'] = (1 if d['labels'][i] == 'A' else 0)
        item['smile'] = d['smiles'][i]
        #分解坐标
        l_ina = inactive[inactive.chain == 'L']
        p_a = active[active.chain != 'L']
        p_ina = inactive[inactive.chain != 'L']
        l_a = active[active.chain == 'L']
        #取三维坐标
        l_a_pos = l_a.loc[:, ['x', 'y', 'z']].values
        p_a_pos = p_a.loc[:, ['x', 'y', 'z']].values
        l_ina_pos = l_ina.loc[:, ['x', 'y', 'z']].values
        p_ina_pos = p_ina.loc[:, ['x', 'y', 'z']].values
        #处理active
        l_a_edges_, l_a_edges_weights = get_neighbors_by_distance(
            l_a_pos, 3)  #+ p_a_pos.shape[0]
        p_a_edges, p_a_edges_weights = get_neighbors_by_distance(p_a_pos, 3)

        ex_a_edges, ex_a_edge_weights = combine_two_graphs(p_a_pos, l_a_pos, 3)
        #ligand是在最后面的
        l_a_edges = [np.array(line) + p_a_pos.shape[0] for line in l_a_edges_]
        # l_a_edges = np.array(a)
        #处理inactive
        l_ina_edges_, l_ina_edges_weights = get_neighbors_by_distance(
            l_ina_pos, 3)
        p_ina_edges, p_ina_edges_weights = get_neighbors_by_distance(
            p_ina_pos, 3)
        ex_ina_edges, ex_ina_edge_weights = combine_two_graphs(
            p_ina_pos, l_ina_pos, 3)
        #ligand是在最后的面的
        l_ina_edges = [
            np.array(line) + p_ina_pos.shape[0] for line in l_ina_edges_
        ]
        # l_ina_edges = np.array(b)
        #建立超图

        item['a_g'] = create_hypergraph(
            l_a_pos.shape[0] + p_a_pos.shape[0], {
                'l': [l_a_edges, l_a_edges_weights],
                'p': [p_a_edges.tolist(), p_a_edges_weights],
                'ex': [ex_a_edges, ex_a_edge_weights]
            })
        item['ina_g'] = create_hypergraph(
            l_ina_pos.shape[0] + p_ina_pos.shape[0], {
                'l': [l_ina_edges, l_ina_edges_weights],
                'p': [p_ina_edges.tolist(), p_ina_edges_weights],
                'ex': [ex_ina_edges, ex_ina_edge_weights]
            })
        #建立节点特征矩阵
        x = p_a['element'].apply(lambda x: pro_names.get(x, 6))  #没有则为6
        a = np.eye(len(prot_atoms) + 1)[x]  #one-hot化
        x2 = l_a['element'].apply(lambda x: lig_names.get(x, 6))  #没有则为6
        a2 = np.eye(len(lig_names) + 1)[x2]  #one-hot化
        dummy_node_feats1 = np.zeros((a.shape[0], a2.shape[1]))
        dummy_node_feats2 = np.zeros((a2.shape[0], a.shape[1]))
        node_feats1 = np.hstack((a, dummy_node_feats1))
        node_feats2 = np.hstack((dummy_node_feats2, a2))
        item['a_x'] = np.vstack((node_feats1, node_feats2))
        #建立ina的节点特征矩阵
        x3 = p_ina['element'].apply(lambda x: pro_names.get(x, 6))  #没有则为6
        a3 = np.eye(len(prot_atoms) + 1)[x3]  #one-hot化
        x4 = l_ina['element'].apply(lambda x: lig_names.get(x, 6))  #没有则为6
        a4 = np.eye(len(lig_names) + 1)[x4]  #one-hot化
        dummy_node_feats3 = np.zeros((a3.shape[0], a4.shape[1]))
        dummy_node_feats4 = np.zeros((a4.shape[0], a3.shape[1]))
        node_feats3 = np.hstack((a3, dummy_node_feats3))
        node_feats4 = np.hstack((dummy_node_feats4, a4))
        item['ina_x'] = np.vstack((node_feats3, node_feats4))
        # x = active['element'].apply(lambda x: atom_names.get(x, 6))  #没有则为6
        # item['a_x'] = np.eye(len(atom_names) + 1)[x]  #one-hot化
        # x = inactive['element'].apply(lambda x: atom_names.get(x, 6))  #没有则为6
        # item['ina_x'] = np.eye(len(atom_names) + 1)[x]
        #记录id
        item['id'] = i
        item['a_pos'] = np.around(np.vstack([p_a_pos, l_a_pos]), 3)
        item['ina_pos'] = np.around(np.vstack([p_ina_pos, l_ina_pos]), 3)
        item['splits'] = (p_a_pos.shape[0], p_ina_pos.shape[0])
        data.append(item)
    return data


if __name__ == '__main__':
    with open('./lep_all.pickle', 'rb') as f:
        d = pickle.load(f)
    data = main(0, len(d))
    with open('./pickle/lep_2gcat_allweight.pickle', 'wb') as f:
        pickle.dump(data, f)
