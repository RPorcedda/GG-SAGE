import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Type
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GGSAGE(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: Optional[int] = 64,
        out_channels: Optional[int] = 64
        ):
        super(GGSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=True)
        self.name = 'GGSAGE'

    def forward(self,
        x: torch.Tensor, # node features
        edge_index: torch.Tensor
        ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return x

    def predict_links(self,
        z: torch.Tensor, # Node embeddings
        edge_index: torch.Tensor
        ) -> torch.Tensor:
        # Split embeddings and weights
        emb = z[:, :-1]  # All but last dimension for embeddings
        weights = z[:, -1]  # Last dimension for weights
        # Compute weights and distances for positive and negative samples
        weight_u = weights[edge_index[0]]
        weight_v = weights[edge_index[1]]

        emb_u = emb[edge_index[0]]
        emb_v = emb[edge_index[1]]

        # Euclidean distance squared
        dist_squared = torch.sum((emb_u - emb_v) ** 2, dim=1)
        # Gravity link probabilities
        links = torch.sigmoid(weight_v - torch.log(dist_squared+1e-10))

        return links

    def loss_function(self,
        pos_preds: torch.Tensor,
        neg_preds: torch.Tensor
        ) -> torch.Tensor:
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        labels = torch.cat([torch.ones(pos_preds.size(0)), torch.zeros(neg_preds.size(0))], dim=0)
        criterion = torch.nn.BCELoss()

        loss = criterion(preds, labels)
        return loss

    @torch.no_grad()
    def extract_embeddings(self, data:Data) -> torch.Tensor:
        self.eval()
        z = self(data.x, data.edge_index)
        return z