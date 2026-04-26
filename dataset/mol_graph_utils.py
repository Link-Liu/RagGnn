"""
SMILES to Molecular Graph Conversion Utilities
Uses RDKit to convert SMILES strings into PyTorch Geometric Data objects
with real atom/bond features for GNN encoding.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Optional, List


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # 1-118
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-2, -1, 0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# Bond feature dimensions
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def one_hot(value, allowable_set):
    """One-hot encoding with an extra 'other' category."""
    encoding = [0] * (len(allowable_set) + 1)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1  # 'other' category
    return encoding


def get_atom_features(atom) -> List[float]:
    """
    Extract atom features from an RDKit atom object.

    Features include:
    - Atomic number (one-hot)
    - Degree (one-hot)
    - Formal charge (one-hot)
    - Number of hydrogens (one-hot)
    - Hybridization (one-hot)
    - Is aromatic (binary)
    - Is in ring (binary)

    Returns:
        List of floats representing atom features
    """
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)
    return features


def get_bond_features(bond) -> List[float]:
    """
    Extract bond features from an RDKit bond object.

    Features include:
    - Bond type (one-hot)
    - Is conjugated (binary)
    - Is in ring (binary)
    - Stereo (one-hot, simplified)

    Returns:
        List of floats representing bond features
    """
    features = []
    features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    return features


def get_num_atom_features() -> int:
    """Return the total number of atom features."""
    # atomic_num(118+1) + degree(7+1) + formal_charge(6+1) + num_hs(5+1) + hybridization(5+1) + aromatic(1) + in_ring(1)
    total = 0
    for key in ATOM_FEATURES:
        total += len(ATOM_FEATURES[key]) + 1  # +1 for 'other'
    total += 2  # aromatic + in_ring
    return total


def get_num_bond_features() -> int:
    """Return the total number of bond features."""
    total = 0
    for key in BOND_FEATURES:
        total += len(BOND_FEATURES[key]) + 1  # +1 for 'other'
    total += 2  # conjugated + in_ring
    return total


def smiles_to_graph(smiles: str, y: Optional[float] = None) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string of the molecule
        y: Optional label value for the molecule

    Returns:
        PyG Data object with node features, edge indices, and edge features,
        or None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Extract atom (node) features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Extract bond (edge) features and edge indices
    # PyG expects undirected edges, so we add both directions
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Molecule with no bonds (e.g., single atom)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_num_bond_features()), dtype=torch.float)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    # Store the SMILES string as metadata
    data.smiles = smiles

    return data


def batch_smiles_to_graphs(smiles_list: List[str],
                           labels: Optional[List[float]] = None) -> List[Data]:
    """
    Convert a batch of SMILES strings to PyG Data objects.

    Args:
        smiles_list: List of SMILES strings
        labels: Optional list of label values

    Returns:
        List of PyG Data objects (invalid SMILES are skipped)
    """
    graphs = []
    for i, smiles in enumerate(smiles_list):
        y = labels[i] if labels is not None else None
        graph = smiles_to_graph(smiles, y)
        if graph is not None:
            graphs.append(graph)
    return graphs


if __name__ == "__main__":
    # Quick test
    test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CC(=O)Oc1ccccc1C(=O)O']
    print(f"Atom feature dim: {get_num_atom_features()}")
    print(f"Bond feature dim: {get_num_bond_features()}")

    for smi in test_smiles:
        graph = smiles_to_graph(smi, y=0)
        if graph:
            print(f"  {smi}: nodes={graph.x.shape[0]}, edges={graph.edge_index.shape[1]}, "
                  f"node_feat_dim={graph.x.shape[1]}, edge_feat_dim={graph.edge_attr.shape[1]}")
