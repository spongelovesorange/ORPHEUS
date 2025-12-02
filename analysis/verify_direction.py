import torch
import numpy as np
from rdkit import Chem

def verify_direction_values():
    # 1. Load the labels generated from the full 33-frame trajectory
    base_path = "/data/Matcha-main/orpheus_physics"
    label_path = f"{base_path}/inference_results/run_1stp_experiment/orpheus_labels/1stp_mol0_conf0_physics.pt"
    try:
        data = torch.load(label_path)
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    directions = data['direction'].numpy() # (32, 16)
    
    # 2. Calculate Mean Direction Cosine over the trajectory
    # This represents the "average intent" of the atom over the whole docking process
    mean_direction = np.mean(directions, axis=0) # (16,)
    
    # 3. Load molecule for symbols
    run_name = "run_1stp_experiment"
    base_name = "1stp_mol0"
    final_preds = np.load(f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy", allow_pickle=True)[0]
    mol = final_preds[base_name]['orig_mol']
    
    print(f"Analysis of Directionality (Full 33 Frames):")
    print("-" * 60)
    print(f"{'Atom Idx':<10} {'Symbol':<8} {'Mean Cosine':<15} {'Interpretation'}")
    print("-" * 60)
    
    # Check specific atoms mentioned by user
    target_atoms = [4, 8]
    
    for idx in range(mol.GetNumAtoms()):
        atom_sym = mol.GetAtomWithIdx(int(idx)).GetSymbol()
        val = mean_direction[idx]
        
        interpretation = "Neutral"
        if val > 0.2: interpretation = "Inward (Attraction)"
        elif val < -0.2: interpretation = "Outward (Repulsion/Adjustment)"
        
        marker = ""
        if idx in target_atoms:
            marker = "<-- CHECK THIS"
            
        print(f"{idx:<10} {atom_sym:<8} {val:<15.4f} {interpretation} {marker}")

if __name__ == "__main__":
    verify_direction_values()
