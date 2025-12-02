import numpy as np
from rdkit import Chem
import torch

def verify_movement():
    # 1. Load the labels we just generated (which used the full 33 frames)
    base_path = "/data/Matcha-main/orpheus_physics"
    label_path = f"{base_path}/inference_results/run_1stp_experiment/orpheus_labels/1stp_mol0_conf0_physics.pt"
    data = torch.load(label_path)
    
    deformations = data['deformation'].numpy() # (32, 16)
    
    # 2. Sum deformation over all 32 steps to get "Total Internal Movement"
    total_deformation = np.sum(deformations, axis=0) # (16,)
    
    # 3. Load molecule to get atom symbols
    run_name = "run_1stp_experiment"
    base_name = "1stp_mol0"
    final_preds = np.load(f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy", allow_pickle=True)[0]
    mol = final_preds[base_name]['orig_mol']
    
    # 4. Rank atoms
    indices = np.argsort(-total_deformation)
    
    print(f"Analysis of Full Trajectory (33 Frames / 32 Steps):")
    print("-" * 40)
    print(f"{'Rank':<5} {'Atom Idx':<10} {'Symbol':<8} {'Total Deformation (A)':<20}")
    print("-" * 40)
    
    for i in range(len(indices)):
        idx = indices[i]
        atom_sym = mol.GetAtomWithIdx(int(idx)).GetSymbol()
        val = total_deformation[idx]
        print(f"{i+1:<5} {idx:<10} {atom_sym:<8} {val:.4f}")

if __name__ == "__main__":
    verify_movement()
