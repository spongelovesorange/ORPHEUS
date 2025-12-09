import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network Layer
    Updates:
    1. Node features (h) -> Invariant
    2. Coordinates (x) -> Equivariant
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, rbf_gamma=0.05):
        super().__init__(aggr='add') # Sum aggregation
        
        # Message Network: phi_m(h_i, h_j, ||x_i - x_j||^2, a_ij)
        # We use RBF for distance embedding, so input is larger
        # FIX: Increase RBF range to 100.0 to capture global context (50-70A)
        self.num_rbf = 64 
        self.rbf_centers = nn.Parameter(torch.linspace(0, 100.0, self.num_rbf))
        self.rbf_gamma = nn.Parameter(torch.tensor([rbf_gamma])) 
        
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + self.num_rbf + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Update Network: phi_h(h_i, m_i)
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Coordinate Network: phi_x(m_ij) -> scalar weight for (x_i - x_j)
        self.coord_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False) # No bias for equivariance
        )
        
        self.edge_dim = edge_dim

    def forward(self, h, x, edge_index, edge_attr, coord_mask=None):
        """
        h: (N, node_dim)
        x: (N, 3)
        edge_index: (2, E)
        edge_attr: (E, edge_dim)
        coord_mask: (N,) boolean mask. True if node coordinates should be updated.
        """
        # Propagate messages
        # Returns: (h_new, x_new)
        
        # 1. Calculate squared distances for all edges
        row, col = edge_index
        diff = x[row] - x[col]
        dist_sq = (diff**2).sum(dim=-1, keepdim=True) # (E, 1)
        
        # RBF Embedding
        dist = torch.sqrt(dist_sq + 1e-8)
        rbf = torch.exp(-self.rbf_gamma * (dist - self.rbf_centers)**2) # (E, num_rbf)
        
        # 2. Propagate
        # Calculate messages m_ij
        msg = self.message(h[row], h[col], rbf, edge_attr) # (E, hidden_dim)
        
        # Calculate coordinate updates
        # (x_i - x_j) * phi_x(m_ij)
        # Clamp coordinate weights for stability
        coord_weight = self.coord_net(msg)
        coord_weight = torch.clamp(coord_weight, min=-10.0, max=10.0)
        
        trans = diff * coord_weight # (E, 3)
        
        # Aggregate coordinate updates
        # We use a separate aggregation for coords
        
        # Let's use the standard flow for h, and manual for x
        h_new = self.propagate(edge_index, x=h, rbf=rbf, edge_attr=edge_attr)
        
        x_agg = torch.zeros_like(x)
        x_agg.index_add_(0, row, trans) # Sum aggregation for coords
        
        if coord_mask is not None:
            # Only update nodes where mask is True
            x_agg = x_agg * coord_mask.unsqueeze(-1).float()
        
        x_new = x + x_agg
        
        return h_new, x_new

    def message(self, x_i, x_j, rbf, edge_attr):
        # x_i, x_j are node features h_i, h_j here
        # input: cat([h_i, h_j, rbf, edge_attr])
        input_feat = torch.cat([x_i, x_j, rbf, edge_attr], dim=-1)
        return self.message_net(input_feat)

    def update(self, aggr_out, x):
        # x is h_i (node features)
        # aggr_out is m_i (aggregated messages)
        input_feat = torch.cat([x, aggr_out], dim=-1)
        return x + self.update_net(input_feat) # Residual connection


class OrpheusStudent(nn.Module):
    def __init__(self, node_dim=17, edge_dim=4, hidden_dim=64, num_layers=4):
        super().__init__()
        
        # Embedding
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim) # If edge_attr is not embedding
        
        # EGNN Layers
        # Split into Global Awareness (1 layer) and Local Refinement (3 layers)
        # Use very small gamma for global layer to see far away
        self.global_layer = EGNNLayer(hidden_dim, hidden_dim, hidden_dim, rbf_gamma=0.001)
        
        self.local_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, hidden_dim, rbf_gamma=0.05) 
            for _ in range(num_layers - 1)
        ])
        
        # Physics Heads
        # 1. Deformation Head (Scalar per node) -> "How much does it deform?"
        self.deformation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Deformation is always positive
        )
        
        # 2. Direction Head (Scalar per node, Cosine Similarity) -> "In or Out?"
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Cosine is [-1, 1]
        )
        
        # REMOVED: Global Trans Head (MLP) - It was invariant and couldn't predict direction.
        # We now use the EGNN global layer output directly.

    def forward(self, h, x, edge_index, edge_attr, coord_mask=None):
        """
        h: Node features (N, node_dim)
        x: Initial coordinates (N, 3)
        coord_mask: (N,) boolean mask
        """
        # Embed
        h = self.node_emb(h)
        edge_attr = self.edge_emb(edge_attr)
        
        # --- STAGE 1: GLOBAL AWARENESS & LOCALIZATION ---
        # Run 1 EGNN layer to gather context from protein (via large RBF)
        # This returns updated coordinates x_global_deformed
        h, x_global_deformed = self.global_layer(h, x, edge_index, edge_attr, coord_mask)
        
        # Calculate the "Pull" vector from the protein
        # Instead of letting the layer deform the ligand, we extract the mean translation
        # This makes the movement RIGID and EQUIVARIANT
        
        pred_trans = torch.zeros(1, 3, device=x.device)
        x_global = x.clone()
        
        if coord_mask is not None:
            # Calculate the update vector for each ligand node
            delta = x_global_deformed - x
            
            # Average the update vectors to get a rigid translation
            # This vector represents the net force exerted by the protein on the ligand
            if coord_mask.sum() > 0:
                pred_trans = delta[coord_mask].mean(dim=0, keepdim=True) # (1, 3)
                
                # Apply rigid translation to all ligand nodes
                x_global[coord_mask] = x[coord_mask] + pred_trans
        
        # --- STAGE 2: LOCAL REFINEMENT (EGNN) ---
        # Run remaining EGNN layers on the globally aligned coordinates
        x_curr = x_global
        
        for layer in self.local_layers:
            h, x_curr = layer(h, x_curr, edge_index, edge_attr, coord_mask)
            
        # Predict Physics Scalars (Auxiliary Tasks)
        pred_deformation = self.deformation_head(h).squeeze(-1)
        pred_direction = self.direction_head(h).squeeze(-1)
        
        return x_curr, pred_deformation, pred_direction, pred_trans
