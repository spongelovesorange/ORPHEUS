import numpy as np
from rdkit import Chem
import os

def main():
    run_name = "run_1stp_experiment"
    base_name = "1stp_mol0"
    base_path = "/data/Matcha-main/orpheus_physics"
    
    # Load data
    final_preds_path = f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy"
    data = np.load(final_preds_path, allow_pickle=True)[0]
    
    if base_name not in data:
        print(f"Error: {base_name} not found.")
        return

    sample = data[base_name]['sample_metrics'][0]
    orig_mol = data[base_name]['orig_mol']
    
    if 'trajectory' not in sample or 'flow_field_05' not in sample:
        print("Error: Missing trajectory or flow field data.")
        return

    traj = sample['trajectory'] # (11, N, 3)
    flow = sample['flow_field_05'] # (N, 3)
    
    # Get position at t=0.5 (Step 5)
    # Trajectory has 11 frames: 0.0, 0.1, ..., 1.0
    # Frame 5 is t=0.5
    pos_t05 = traj[5]
    
    print(f"Visualizing flow at t=0.5 (Frame 5)")
    print(f"Flow magnitude mean: {np.linalg.norm(flow, axis=1).mean():.4f}")
    
    # Create a PDB with vectors
    # We will use a trick: Create a new molecule that contains:
    # 1. The ligand atoms at pos_t05
    # 2. Dummy atoms at pos_t05 + scale * flow
    # 3. Bonds between them
    
    scale = 2.0 # Scale factor for visualization
    
    # Create an editable molecule
    rw_mol = Chem.RWMol(orig_mol)
    conf = rw_mol.GetConformer()
    
    # Set positions for existing atoms
    for i in range(pos_t05.shape[0]):
        x, y, z = pos_t05[i]
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        
    # Add dummy atoms and bonds
    num_atoms = orig_mol.GetNumAtoms()
    for i in range(num_atoms):
        # Calculate tip position
        start = pos_t05[i]
        vec = flow[i]
        end = start + vec * scale
        
        # Add dummy atom (e.g., F for visibility, or just C)
        new_idx = rw_mol.AddAtom(Chem.Atom(9)) # Fluorine
        
        # Set position
        conf.SetAtomPosition(new_idx, (float(end[0]), float(end[1]), float(end[2])))
        
        # Add bond
        rw_mol.AddBond(i, new_idx, Chem.BondType.SINGLE)
        
    # Save
    output_file = "flow_vectors_t05.pdb"
    Chem.MolToPDBFile(rw_mol, output_file)
    print(f"Saved visualization to {output_file}")
    print("Open this file in PyMOL. The 'spikes' (F atoms) show the direction and magnitude of the force.")

if __name__ == "__main__":
    main()
