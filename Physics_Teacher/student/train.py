import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from dataset import OrpheusDataset
from model import OrpheusStudent
import os

def train():
    # Config
    BATCH_SIZE = 1 # Keep 1 for now
    LR = 1e-4 
    EPOCHS_STAGE1 = 50 # Global Localization
    EPOCHS_STAGE2 = 50 # Local Refinement
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    data_dir = "/data/Matcha-main/orpheus_physics/inference_results/run_orpheus_20/orpheus_dataset"
    dataset = OrpheusDataset(data_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = OrpheusStudent(node_dim=17, edge_dim=4, hidden_dim=64).to(DEVICE)
    
    # Load Checkpoint if exists (optional, but we usually start fresh for 2-step)
    # Actually, let's start fresh to ensure clean separation
    # But if user wants to continue, we can load.
    # Let's just start fresh for this "2-step" run.
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"Start training on {len(dataset)} samples...")
    
    # --- STAGE 1: GLOBAL LOCALIZATION ---
    print("\n--- STAGE 1: GLOBAL LOCALIZATION (50 Epochs) ---")
    print("Freezing Local Layers...")
    
    # Freeze Local Layers
    for param in model.local_layers.parameters():
        param.requires_grad = False
    for param in model.deformation_head.parameters():
        param.requires_grad = False
    for param in model.direction_head.parameters():
        param.requires_grad = False
        
    # Ensure Global Layers are trainable
    for param in model.global_layer.parameters():
        param.requires_grad = True
        
    for epoch in range(EPOCHS_STAGE1):
        run_epoch(epoch, EPOCHS_STAGE1, model, loader, optimizer, DEVICE, stage=1)
        
    # --- STAGE 2: LOCAL REFINEMENT ---
    print("\n--- STAGE 2: LOCAL REFINEMENT (50 Epochs) ---")
    print("Unfreezing All Layers...")
    
    # Unfreeze All
    for param in model.parameters():
        param.requires_grad = True
        
    for epoch in range(EPOCHS_STAGE2):
        run_epoch(epoch, EPOCHS_STAGE2, model, loader, optimizer, DEVICE, stage=2)

    # Save Model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/student_model.pth")
    print("Training finished. Model saved.")

def run_epoch(epoch, total_epochs, model, loader, optimizer, DEVICE, stage):
    total_loss = 0
    loss_coord_sum = 0
    loss_def_sum = 0
    loss_dir_sum = 0
    loss_bond_sum = 0
    loss_trans_sum = 0
    loss_center_sum = 0
    valid_batches = 0
    
    for batch in loader:
        batch = batch.to(DEVICE)
        # Ensure float32
        batch.x = batch.x.float()
        batch.h = batch.h.float()
        batch.edge_attr = batch.edge_attr.float()
        batch.y = batch.y.float()
        batch.deformation = batch.deformation.float()
        batch.direction = batch.direction.float()
        
        optimizer.zero_grad()
        
        # Forward
        pred_x, pred_def, pred_dir, pred_trans = model(batch.h, batch.x, batch.edge_index, batch.edge_attr, coord_mask=batch.ligand_mask)
        
        mask = batch.ligand_mask
        if not mask.any(): continue

        # Metrics
        batch_idx = batch.batch[mask]
        init_center = global_mean_pool(batch.x[mask], batch_idx)
        target_center = global_mean_pool(batch.y, batch_idx)
        target_trans = target_center - init_center
        
        # Losses
        loss_coord = F.l1_loss(pred_x[mask], batch.y)
        
        if pred_trans.shape[0] == target_trans.shape[0]:
            # SCALING FIX: Scale target by 0.01 to make values ~0-1 range
            # This helps optimization stability significantly for large translations (90A)
            scale_factor = 0.01
            loss_trans = F.l1_loss(pred_trans * scale_factor, target_trans * scale_factor)
            # Note: We scale both or just scale the loss. Scaling inputs to loss is better.
            # But wait, if we scale target, the model will learn to predict scaled values?
            # No, we want model to predict REAL values.
            # So we should NOT scale pred_trans inside the model, but we can scale the LOSS gradient.
            # Actually, standard practice is to normalize targets.
            # Let's try: loss = L1(pred * 0.1, target * 0.1) -> This scales gradients by 0.1.
            # Let's use 0.1 to be safe.
            loss_trans = F.l1_loss(pred_trans, target_trans) / 10.0 
        else:
            loss_trans = torch.tensor(0.0, device=DEVICE) # Should handle batch mismatch properly later

        loss_def = F.mse_loss(pred_def[mask], batch.deformation)
        loss_dir = F.mse_loss(pred_dir[mask], batch.direction)
        
        # Bond Loss
        bond_mask = batch.edge_attr[:, 0] == 1.0
        if bond_mask.any():
            bond_edges = batch.edge_index[:, bond_mask]
            row, col = bond_edges
            init_dist = torch.norm(batch.x[row] - batch.x[col], dim=-1)
            pred_dist = torch.norm(pred_x[row] - pred_x[col], dim=-1)
            loss_bond = F.mse_loss(pred_dist, init_dist)
        else:
            loss_bond = torch.tensor(0.0, device=DEVICE)

        pred_center = global_mean_pool(pred_x[mask], batch_idx)
        loss_center = F.l1_loss(pred_center, target_center)

        # Weighted Loss
        if stage == 1:
            # Focus on Translation and Center
            # Ignore local physics (def, dir, bond) mostly, but keep bond to prevent explosion
            loss = 1.0 * loss_trans + 1.0 * loss_center + 0.1 * loss_bond
        else:
            # Refinement - PHYSICS DISTILLATION MODE
            # We prioritize learning the Physics Fields (Def/Dir) over perfect coordinates
            # This verifies if the student can learn the "Teacher's Intuition"
            loss = 1.0 * loss_coord + 0.1 * loss_trans + 20.0 * loss_def + 20.0 * loss_dir + 5.0 * loss_bond + 1.0 * loss_center
        
        if torch.isnan(loss):
            print("Warning: NaN loss detected!")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_coord_sum += loss_coord.item()
        loss_def_sum += loss_def.item()
        loss_dir_sum += loss_dir.item()
        loss_bond_sum += loss_bond.item()
        loss_center_sum += loss_center.item()
        loss_trans_sum += loss_trans.item()
        valid_batches += 1
        
    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
        if (epoch + 1) % 10 == 0:
            print(f"Stage {stage} | Epoch {epoch+1}/{total_epochs} | Loss: {avg_loss:.4f} (Coord: {loss_coord_sum/valid_batches:.4f}, Def: {loss_def_sum/valid_batches:.4f}, Dir: {loss_dir_sum/valid_batches:.4f}, Trans: {loss_trans_sum/valid_batches:.4f})")

if __name__ == "__main__":
    train()
