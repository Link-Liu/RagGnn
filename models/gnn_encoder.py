"""
GNN Encoder for Molecular Graphs
Implements GIN/GINE (Graph Isomorphism Network) encoder for molecular graph representation.
Uses GINEConv when edge features are provided for richer molecular representations.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool
from torch_geometric.data import Data
import torch.nn.functional as F


class GNNEncoder(nn.Module):
    """
    GNN Encoder for molecular graphs.
    Automatically uses GINEConv when edge features are provided,
    otherwise falls back to standard GINConv.
    """

    def __init__(self, num_node_features: int, num_edge_features: int = 0,
                 hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        """
        Initialize GNN Encoder

        Args:
            num_node_features: Number of node features
            num_edge_features: Number of edge features (default: 0)
            hidden_dim: Hidden dimension size
            num_layers: Number of GIN layers
            dropout: Dropout rate
        """
        super(GNNEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim

        # Edge embedding (project edge features to hidden_dim for GINEConv)
        if num_edge_features > 0:
            self.edge_emb = nn.Linear(num_edge_features, hidden_dim)
        else:
            self.edge_emb = None

        # Build GIN / GINE layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if num_edge_features > 0:
                # GINEConv requires edge_dim == in_channels of the MLP
                # We project edge features to hidden_dim and set edge_dim=hidden_dim
                self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            else:
                self.convs.append(GINConv(mlp))

        # Batch normalization
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        # Final projection layer
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through the GNN encoder

        Args:
            x: Node features tensor of shape [num_nodes, num_node_features]
            edge_index: Edge indices tensor of shape [2, num_edges]
            edge_attr: Edge attributes tensor of shape [num_edges, num_edge_features]
            batch: Batch tensor for batched graphs

        Returns:
            Molecular embedding tensor of shape [batch_size, hidden_dim]
        """
        # Embed edge features once (reused in all layers)
        edge_emb = None
        if edge_attr is not None and self.edge_emb is not None:
            edge_emb = self.edge_emb(edge_attr)

        h = x
        for i, conv in enumerate(self.convs):
            # Apply convolution
            if edge_emb is not None:
                h = conv(h, edge_index, edge_emb)
            else:
                h = conv(h, edge_index)

            # Batch norm + activation + dropout
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling to get molecule-level representation
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)

        # Final projection
        h = self.projector(h)

        return h


def create_gnn_encoder(num_node_features: int, num_edge_features: int = 0,
                      hidden_dim: int = 64, num_layers: int = 3) -> GNNEncoder:
    """
    Factory function to create a GNN encoder

    Args:
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        hidden_dim: Hidden dimension size
        num_layers: Number of GIN layers

    Returns:
        Initialized GNNEncoder instance
    """
    return GNNEncoder(num_node_features, num_edge_features, hidden_dim, num_layers)


# Example usage function
def example_usage():
    """Example of how to use the GNN encoder"""
    # Example: Create a simple molecular graph with 5 nodes and 4 edges
    # Node features: atomic numbers (e.g., 6 for carbon, 8 for oxygen, etc.)
    node_features = torch.tensor([
        [6],  # Carbon
        [6],  # Carbon
        [8],  # Oxygen
        [1],  # Hydrogen
        [1]   # Hydrogen
    ], dtype=torch.float)

    # Edge indices (undirected graph)
    edge_index = torch.tensor([
        [0, 1, 1, 2],  # Edges: 0-1, 1-2, 1-3, 1-4
        [1, 2, 3, 4]
    ], dtype=torch.long)

    # Create encoder
    encoder = create_gnn_encoder(num_node_features=1, hidden_dim=32, num_layers=2)

    # Forward pass
    embedding = encoder(node_features, edge_index)
    print(f"Molecular embedding shape: {embedding.shape}")

    return encoder, embedding


if __name__ == "__main__":
    # Run example
    encoder, embedding = example_usage()