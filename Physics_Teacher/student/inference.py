import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset import OrpheusDataset
from model import OrpheusStudent
import numpy as np
import os

def calculate_rmsd(pred, target):
    """Calculate RMSD between two sets of coordinates."""
    diff = pred - target
    dist_sq = (diff ** 2).sum(dim=-1)
    rmsd = torch.sqrt(dist_sq.mean())
    return rmsd.item()

def inference():
    # Config
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_PATH = "checkpoints/student_model.pth"
    DATA_DIR = "/data/Matcha-main/orpheus_physics/inference_results/run_orpheus_20/orpheus_dataset"
    
    # Load Data
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = OrpheusDataset(DATA_DIR)
    # Use a small batch size or just iterate manually
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # No shuffle to see specific samples
    
    # Load Model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = OrpheusStudent(node_dim=17, edge_dim=4, hidden_dim=64).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print("Checkpoint not found! Please train the model first.")
        return

    model.eval()
    
    print("\n--- Inference Results (First 5 Samples) ---")
    print(f"{'Sample ID':<20} | {'RMSD (Å)':<10} | {'Def Pred':<10} {'Def True':<10} | {'Dir Pred':<10} {'Dir True':<10}")
    print("-" * 90)
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5: break
            
            batch = batch.to(DEVICE)
            
            # Ensure float32
            batch.x = batch.x.float()
            batch.h = batch.h.float()
            batch.edge_attr = batch.edge_attr.float()
            batch.y = batch.y.float()
            batch.deformation = batch.deformation.float()
            batch.direction = batch.direction.float()
            
            # Forward
            pred_x, pred_def, pred_dir, pred_trans = model(batch.h, batch.x, batch.edge_index, batch.edge_attr, coord_mask=batch.ligand_mask)
            
            # Metrics
            mask = batch.ligand_mask
            rmsd = calculate_rmsd(pred_x[mask], batch.y)
            
            # Physics Scalars (Mean over molecule for display)
            def_pred_mean = pred_def[mask].mean().item()
            def_true_mean = batch.deformation.mean().item()
            
            dir_pred_mean = pred_dir[mask].mean().item()
            dir_true_mean = batch.direction.mean().item()
            
            # Sample ID (we didn't save it in the batch object in dataset.py explicitly as an attribute accessible by batch.sample_id easily unless we modify dataset, 
            # but we can just use index for now or try to retrieve if we added it to Data object)
            # In dataset.py we didn't add 'sample_id' to Data() object. 
            # Let's just use index.
            sample_id = f"Sample {i}"
            
            print(f"{sample_id:<20} | {rmsd:<10.4f} | {def_pred_mean:<10.4f} {def_true_mean:<10.4f} | {dir_pred_mean:<10.4f} {dir_true_mean:<10.4f}")

    print("-" * 90)
    print("RMSD: Root Mean Square Deviation of coordinates (Lower is better)")
    print("Def: Deformation Potential (Scalar)")
    print("Dir: Directionality Cosine (Scalar)")

if __name__ == "__main__":
    inference()
