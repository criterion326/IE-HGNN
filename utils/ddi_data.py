from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from torch_geometric.data import Dataset
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

element_names = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]


@dataclass
class BondConfig:

    bond_type: bool
    conjugation: bool
    ring: bool

    def __post_init__(self):
        self.n_features = 0
        self.feat_names = []
        if self.bond_type:
            self.n_features += 4
            self.feat_names += [
                "Single",
                "Double",
                "Triple",
                "Aromatic",
            ]
        if self.conjugation:
            self.n_features += 1
            self.feat_names += ["Conjugation"]
        if self.ring:
            self.n_features += 1
            self.feat_names += ["inRing"]


@dataclass
class AtomConfig:

    element_type: bool
    degree: bool
    implicit_valence: bool
    formal_charge: bool
    num_rad_e: bool
    hybridization: bool
    combo_hybrid: bool  # if True, sp2 and sp3 will be merged into one feature
    aromatic: bool

    def __post_init__(self):
        self.n_features = 0
        self.feat_names = []

        def update(names):
            self.feat_names += names
            self.n_features += len(names)

        if self.element_type:
            update(element_names)
        if self.degree:
            update([f"degree{ind}" for ind in range(11)])
        if self.implicit_valence:
            update([f"implicitValence{ind}" for ind in range(7)])
        if self.formal_charge:
            update(["formalCharge"])
        if self.num_rad_e:
            update(["numRadElectons"])
        if self.hybridization:
            if not self.combo_hybrid:
                update([
                    "HybridizationType.SP",
                    "HybridizationType.SP2",
                    "HybridizationType.SP3",
                    "HybridizationType.SP3D",
                    "HybridizationType.SP3D2",
                ])
            else:
                update([
                    "HybridizationType.SP",
                    "HybridizationType.SP2or3",
                    "HybridizationType.SP3D",
                    "HybridizationType.SP3D2",
                ])
        if self.aromatic:
            update(["Aromatic"])


def bond_fp(bond, config):

    bt = bond.GetBondType()
    bond_feats = []
    if config.bond_type:
        bond_feats += [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]
    if config.conjugation:
        bond_feats.append(bond.GetIsConjugated())
    if config.ring:
        bond_feats.append(bond.IsInRing())

    return bond_feats


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_fp(atom, atom_config: AtomConfig):
    results = []
    if atom_config.element_type:
        results += one_of_k_encoding_unk(
            atom.GetSymbol(),
            element_names,
        )
    if atom_config.degree:
        results += one_of_k_encoding(atom.GetDegree(),
                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if atom_config.implicit_valence:
        results += one_of_k_encoding_unk(atom.GetImplicitValence(),
                                         [0, 1, 2, 3, 4, 5, 6])
    if atom_config.formal_charge:
        results += [atom.GetFormalCharge()]
    if atom_config.num_rad_e:
        results += [atom.GetNumRadicalElectrons()]
    if atom_config.hybridization:
        feat = atom.GetHybridization()
        if atom_config.combo_hybrid:
            if (feat == Chem.rdchem.HybridizationType.SP2) or (
                    feat == Chem.rdchem.HybridizationType.SP3):
                feat = "SP2/3"
            options = [
                Chem.rdchem.HybridizationType.SP,
                "SP2/3",
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]
        else:
            options = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]
        results += one_of_k_encoding_unk(feat, options)
    if atom_config.aromatic:
        results += [atom.GetIsAromatic()]

    return results


def atom_helper(molecule, ind, atom_config):
    atom = molecule.GetAtomWithIdx(ind)
    atom_feature = atom_fp(atom, atom_config)
    ##### atom feature list #####
    # symbol, one-hot
    # number of neighbors, one-hot
    # implicit valence, one-hot
    # formal charge
    # num. radical electrons
    # hybridization, one-hot
    # is aromatic?
    # num. hydrogen
    # chirality
    #############################
    return atom_feature


import torch
from torch_geometric.data import Data

bond_config = BondConfig(True, True, True)
atom_config = AtomConfig(
    True,
    True,
    True,
    True,
    True,
    True,
    combo_hybrid=False,  # if True, SP2/SP3 are combined into one feature
    aromatic=True,
)


def construct_graph(
    molecule,
    y,
    bond_config,
    atom_config,
):
    n_atoms = molecule.GetNumAtoms()
    node_features = [
        atom_helper(molecule, i, atom_config) for i in range(0, n_atoms)
    ]
    edge_features = []
    edge_indices = []

    for bond in molecule.GetBonds():
        edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_indices.append([bond.GetEndAtomIdx(),
                             bond.GetBeginAtomIdx()
                             ])  # add "opposite" edge for undirected graph
        # bond_feature = bond_fp(bond).reshape((6,))  # ADDED edge feat
        bond_feature = bond_fp(bond, bond_config)
        ##### bond feature list #####
        # type of bond (1,2,3,1.5,etc.), one-hot
        # is conjugated?
        # is in ring?
        # chirality
        #############################
        edge_features.extend([bond_feature, bond_feature
                              ])  # both bonds have the same features
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(
        ),  # as shown in https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float),
    )
    return data


class My_dataset(Dataset):
    '''
    自定义dataset.
    '''

    def __init__(self, df):
        super().__init__()
        self.data = []
        for i in tqdm_notebook(range(len(df)), position=0):
            mol1 = Chem.MolFromSmiles(df['smiles_1'][i])
            mol2 = Chem.MolFromSmiles(df['smiles_2'][i])
            graph1 = construct_graph(mol1, df['label'][i], bond_config,
                                     atom_config)
            graph2 = construct_graph(mol2, df['label'][i], bond_config,
                                     atom_config)
            self.data.append((graph1, graph2))
        # self.data=list(eq_graph_tensors.values())
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CollaterDDI(object):

    def __init__(self):
        pass

    def __call__(self, data_list):
        # batch = Batch.from_data_list([d for d in data_list])
        batch1 = Batch.from_data_list([d[0] for d in data_list])
        batch2 = Batch.from_data_list([d[1] for d in data_list])
        return batch1, batch2


if __name__ == '__main__':
    basedir = './data/zhangDDI/'
    train_df = pd.read_csv(os.path.join(basedir, 'ZhangDDI_train.csv'))
    val_df = pd.read_csv(os.path.join(basedir, 'ZhangDDI_valid.csv'))
    test_df = pd.read_csv(os.path.join(basedir, 'ZhangDDI_test.csv'))
    # df=pd.read_csv(os.path.join(basedir,'ZhangDDI_all=95245.csv'))
    drug_list_df = pd.read_csv(os.path.join(basedir, 'drug_list_zhang.csv'))
    df = pd.concat([train_df, val_df, test_df], axis=0)
    df.drop_duplicates(subset=['drugbank_id_1', 'drugbank_id_2'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    all_drugs = drug_list_df.drugbank_id.values
    np.random.seed(25)
    # idxs = np.random.permutation(len(all_drugs))
    idxs = np.loadtxt('./zhangddi_idxs=20-25.txt', dtype=int)
    # idxs[:400]
    old, new = all_drugs[idxs[3, :470]], all_drugs[idxs[3, 470:]]
    batch_size = 64
    train_datasets = My_dataset(train_df)
    val_datasets = My_dataset(val_df)
    test_datasets = My_dataset(test_df)
    train_loader = DataLoader(train_datasets,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              collate_fn=CollaterLBA())
    val_loader = DataLoader(val_datasets,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=CollaterLBA())
    test_loader = DataLoader(test_datasets,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=CollaterLBA())
