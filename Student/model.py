import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig

class StudentModel(nn.Module):
    def __init__(self, esm_model_path, ligand_dim=2048, hidden_dim=480, num_tokens=4096):
        """
        Args:
            esm_model_path: Path to the local ESM-2 model directory.
            ligand_dim: Dimension of the ligand input (e.g., Morgan fingerprint size).
            hidden_dim: Hidden dimension for fusion and projection (ESM-2 t12 is 480).
            num_tokens: Number of geometry tokens (classes) to predict.
        """
        super().__init__()
        
        # 1. Protein Encoder (ESM-2)
        # Load config first to ensure we can load from local
        self.esm = EsmModel.from_pretrained(esm_model_path)
        
        # Freeze ESM-2 layers (optional, can unfreeze later)
        for param in self.esm.parameters():
            param.requires_grad = False
            
        # 2. Ligand Encoder
        # Simple projection from fingerprint to hidden_dim
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Fusion Layer (Cross Attention)
        # Query: Protein features, Key/Value: Ligand features
        self.fusion = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        # 4. Geometry Head (Projection to Token ID)
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tokens)
        )
        
    def forward(self, input_ids, attention_mask, ligand_feat):
        """
        Args:
            input_ids: [Batch, Seq_Len] (ESM-2 tokens)
            attention_mask: [Batch, Seq_Len]
            ligand_feat: [Batch, Ligand_Dim] (e.g. Fingerprints)
        """
        
        # 1. Encode Protein
        # ESM output: last_hidden_state [Batch, Seq_Len, 480]
        esm_output = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        protein_emb = esm_output.last_hidden_state
        
        # 2. Encode Ligand
        # [Batch, Ligand_Dim] -> [Batch, 1, Hidden_Dim] (Add seq dim for attention)
        ligand_emb = self.ligand_encoder(ligand_feat).unsqueeze(1)
        
        # 3. Fusion
        # We want to update protein embedding with ligand info
        # Q: Protein, K: Ligand, V: Ligand
        # Note: This is a simplification. Usually we might want self-attention on protein first.
        # But ESM already did self-attention.
        # Here we just want to "attend" to the ligand.
        # Since ligand is just 1 vector, it's basically adding/modulating the protein vector.
        
        # Standard Cross Attention:
        # attn_output, _ = self.fusion(query=protein_emb, key=ligand_emb, value=ligand_emb)
        
        # Alternative: Simple concatenation or addition if ligand is global
        # Let's use a simple Gated fusion or Addition for stability with single vector
        # fused_emb = protein_emb + ligand_emb
        
        # Let's stick to the user's "Fusion" concept.
        # If we use MultiheadAttention with Key/Value as ligand (length 1), 
        # it effectively broadcasts the ligand info to every residue.
        fused_emb, _ = self.fusion(query=protein_emb, key=ligand_emb, value=ligand_emb)
        
        # Residual connection (important!)
        fused_emb = protein_emb + fused_emb
        
        # 4. Predict Tokens
        logits = self.geometry_head(fused_emb) # [Batch, Seq_Len, Num_Tokens]
        
        return logits
