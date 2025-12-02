import os
import numpy as np
import copy
from copy import deepcopy
import safetensors
from tqdm import tqdm
import torch
from matcha.utils.rotation import expm_SO3
from matcha.utils.transforms import (apply_tr_rot_changes_to_batch_inplace, apply_tor_changes_to_batch_inplace,
                                     find_rigid_alignment)


def load_from_checkpoint(model, checkpoint_path, strict=True):
    state_dict = safetensors.torch.load_file(os.path.join(
        checkpoint_path, 'model.safetensors'), device="cpu")
    model.load_state_dict(state_dict, strict=strict)
    return model


def euler(model, batch, device, num_steps=20, return_trajectory=False):
    cur_batch = deepcopy(batch).to(device)
    h = 1. / num_steps
    batch_size = len(cur_batch)
    R_eye = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    R_agg = torch.eye(3, device=device).repeat(batch_size, 1, 1)
    tr = cur_batch.ligand.init_tr
    tor = cur_batch.ligand.init_tor
    cur_batch.ligand.t = torch.zeros_like(cur_batch.ligand.t)

    trajectory = []
    if return_trajectory:
        trajectory.append(cur_batch.ligand.pos.clone())

    flow_field = None
    target_t = 0.5 # Extract flow at t=0.5

    tor_agg = torch.zeros_like(tor)
    for step in range(num_steps):
        t_current = step / num_steps
        capture_flow = False
        # Check if this step is the one crossing t=0.5
        if abs(t_current - target_t) < (0.51 / num_steps):
            capture_flow = True
            pos_before = cur_batch.ligand.pos.clone()

        with torch.no_grad():
            dtr, drot, dtor, _, _ = model.forward_step(cur_batch)

            tr = tr + h * dtr
            if dtor is not None:
                tor = h * dtor
                tor_agg += tor
            if drot is not None:
                R = expm_SO3(drot, h)
            else:
                R = R_eye

            apply_tor_changes_to_batch_inplace(
                cur_batch, tor, is_reverse_order=False)
            apply_tr_rot_changes_to_batch_inplace(cur_batch, tr, R)

            cur_batch.ligand.t += h
            R_agg = torch.bmm(R, R_agg)
        
        if capture_flow:
            pos_after = cur_batch.ligand.pos
            # Calculate effective velocity vector: v = dx/dt approx (x_new - x_old) / h
            flow_field = (pos_after - pos_before) / h

        if return_trajectory:
            trajectory.append(cur_batch.ligand.pos.clone())

    tr_agg = tr
    return cur_batch, tr_agg, R_agg, tor_agg, trajectory, flow_field


def run_evaluation(dataloader, num_steps, solver, model, save_trajectory=False):
    def revert_augm(batch):
        batch.ligand.pos[:] = torch.einsum(
            'bij,bjk->bik', batch.ligand.pos, batch.original_augm_rot)
        for batch_idx, num_atoms in enumerate(batch.ligand.num_atoms):
            batch.ligand.pos[batch_idx, num_atoms:] = 0.
        return batch.ligand.pos

    def revert_augm_trajectory(trajectory, original_augm_rot, num_atoms_list):
        # trajectory: list of (B, N, 3)
        # original_augm_rot: (B, 3, 3)
        reverted_traj = []
        for pos in trajectory:
            # pos: (B, N, 3)
            # Apply rotation
            pos_reverted = torch.einsum('bij,bjk->bik', pos, original_augm_rot)
            # Zero out padding
            for batch_idx, num_atoms in enumerate(num_atoms_list):
                pos_reverted[batch_idx, num_atoms:] = 0.
            reverted_traj.append(pos_reverted)
        return reverted_traj

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics_dict = {}
    for batch in tqdm(dataloader, desc="Docking inference"):
        batch = batch['batch']
        optimized, tr_agg, R_agg, tor_agg, trajectory, flow_field = solver(
            model, batch, device=device, num_steps=num_steps, return_trajectory=save_trajectory)

        for batch_idx, num_atoms in enumerate(optimized.ligand.num_atoms):
            optimized.ligand.pos[batch_idx, num_atoms:] = 0.

        # Normal alignment process
        aligned_batch = copy.deepcopy(batch).to(device)
        tr_aligned = torch.zeros_like(tr_agg, device=device)
        rot_aligned = torch.eye(3, device=device).repeat(tr_agg.shape[0], 1, 1)
        apply_tor_changes_to_batch_inplace(
            aligned_batch, tor_agg, is_reverse_order=False)
        for i in range(len(optimized.ligand.pos)):
            pos_pred = aligned_batch.ligand.pos[i,
                                                :optimized.ligand.num_atoms[i]]
            pos_true = optimized.ligand.pos[i, :optimized.ligand.num_atoms[i]]

            rot, tr = find_rigid_alignment(pos_pred, pos_true)
            tr_aligned[i] = tr
            rot_aligned[i] = rot

        apply_tr_rot_changes_to_batch_inplace(
            aligned_batch, tr_aligned, rot_aligned)
        tr_agg = tr_aligned
        R_agg = rot_aligned

        # Handle tr_agg_init_coord computation
        tr_agg_init_coord = torch.bmm(
            tr_agg[:, None, :], optimized.original_augm_rot)

        init_batch = copy.deepcopy(batch).to(device)
        init_batch.ligand.pos = optimized.ligand.pos.clone().to(device)
        transformed_orig = revert_augm(init_batch)

        # Process trajectory if saved
        reverted_trajectory_numpy = None
        if save_trajectory and trajectory is not None:
            reverted_traj = revert_augm_trajectory(trajectory, optimized.original_augm_rot, optimized.ligand.num_atoms)
            # Convert to numpy list of arrays
            # Structure: list of steps. Each step is (B, N, 3).
            # We want to store it per sample.
            # Let's make it a list of (Steps, N, 3) for each sample.
            
            # Stack steps: (Steps, B, N, 3)
            traj_stack = torch.stack(reverted_traj, dim=0) 
            # Permute to (B, Steps, N, 3)
            traj_stack = traj_stack.permute(1, 0, 2, 3)
            reverted_trajectory_numpy = traj_stack.cpu().numpy()

        # Process flow field
        reverted_flow_numpy = None
        if flow_field is not None:
             # Revert augmentation for flow vectors
             flow_reverted = torch.einsum('bij,bjk->bik', flow_field, optimized.original_augm_rot)
             for batch_idx, num_atoms in enumerate(optimized.ligand.num_atoms):
                flow_reverted[batch_idx, num_atoms:] = 0.
             reverted_flow_numpy = flow_reverted.cpu().numpy()

        for i, name in enumerate(batch.names):
            complex_metrics = {}
            complex_metrics['transformed_orig'] = transformed_orig[i,
                                                                   :optimized.ligand.num_atoms[i]].cpu().numpy()
            
            if save_trajectory and reverted_trajectory_numpy is not None:
                # Extract trajectory for this sample and remove padding atoms
                num_atoms = optimized.ligand.num_atoms[i]
                complex_metrics['trajectory'] = reverted_trajectory_numpy[i, :, :num_atoms, :]

            if reverted_flow_numpy is not None:
                num_atoms = optimized.ligand.num_atoms[i]
                complex_metrics['flow_field_05'] = reverted_flow_numpy[i, :num_atoms, :]

            # Handle cases where aggregated values might be None
            if tr_agg is not None:
                complex_metrics['tr_pred_init'] = tr_agg_init_coord[i].cpu(
                ).numpy()
            else:
                complex_metrics['tr_pred_init'] = np.zeros(3)

            if R_agg is not None:
                complex_metrics['rot_pred'] = R_agg[i].cpu().numpy()
            else:
                complex_metrics['rot_pred'] = np.eye(3)
            complex_metrics['rot_augm'] = optimized.original_augm_rot[i].cpu(
            ).numpy()
            complex_metrics['full_protein_center'] = optimized.protein.full_protein_center[i].cpu(
            ).numpy()
            metrics_dict[name] = metrics_dict.get(name, []) + [complex_metrics]
    return metrics_dict


def scoring_inference(loader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics_dict = {}
    for batch in tqdm(loader, desc="Scoring"):
        batch = batch['batch'].to(device)
        with torch.no_grad():
            rmsd_pred = model.forward_step(batch)
            rmsd_pred = rmsd_pred[0]

        scores = rmsd_pred

        for i, (name, score) in enumerate(zip(batch.names, scores.cpu().numpy())):
            metrics_dict[name] = metrics_dict.get(name, []) + [score]
    return metrics_dict
