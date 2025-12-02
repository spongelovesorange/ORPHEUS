import numpy as np
from rdkit import Chem

def visualize_flow_direction(run_name, base_name):
    # Updated path
    base_path = "/data/Matcha-main/orpheus_physics"
    # 1. Load Trajectory
    filename = f"{base_path}/inference_results/{run_name}/stage3_any_conf.npy"
    try:
        data = np.load(filename, allow_pickle=True)[0]
        key = f"{base_name}_conf0"
        if key not in data:
            print(f"Key {key} not found.")
            return
        sample = data[key][0]
        traj = sample['trajectory'] # (11, N, 3)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Load Original Molecule & Protein Center
    final_preds = np.load(f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy", allow_pickle=True)[0]
    orig_mol = final_preds[base_name]['orig_mol']
    
    # We need the protein center to define "inward" vs "outward"
    # In Matcha, 'full_protein_center' is stored in the metrics
    # But let's calculate it from the PDB to be sure, or use the one in sample if available.
    # sample['full_protein_center'] exists in stage3 data!
    
    if 'full_protein_center' in sample:
        pocket_center = sample['full_protein_center']
    else:
        # Fallback: Calculate from PDB
        print("Calculating protein center from PDB...")
        prot = Chem.MolFromPDBFile(f"{base_path}/data/my_dataset/1stp/1stp_protein.pdb")
        pocket_center = prot.GetConformer().GetPositions().mean(axis=0)
        
    print(f"Pocket Center: {pocket_center}")

    # 3. Calculate Internal Velocity Vectors (Same as Level 1)
    num_frames = traj.shape[0]
    num_atoms = traj.shape[1]
    
    # We use the total displacement from start to end (after alignment) as the "net intention"
    # Align Frame 0 to Frame 10
    frame_start = traj[0]
    frame_end = traj[-1]
    
    # Helper for Kabsch alignment
    def get_aligned_pos(mobile, target):
        mob_center = mobile.mean(axis=0)
        tar_center = target.mean(axis=0)
        mob_c = mobile - mob_center
        tar_c = target - tar_center
        H = np.dot(mob_c.T, tar_c)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(Vt.T, U.T)
        return np.dot(mob_c, R) + tar_center

    # Align start to end
    start_aligned = get_aligned_pos(frame_start, frame_end)
    
    # Vector v = End - Start_Aligned
    vectors = frame_end - start_aligned # (N, 3)
    
    # 4. Calculate Cosine Similarity with "To Pocket Center" vector
    cosine_scores = np.zeros(num_atoms)
    
    for i in range(num_atoms):
        atom_pos = frame_end[i]
        vec_v = vectors[i]
        
        # Vector from atom to pocket center
        vec_r = pocket_center - atom_pos
        
        # Normalize
        norm_v = np.linalg.norm(vec_v)
        norm_r = np.linalg.norm(vec_r)
        
        if norm_v < 1e-3: # Atom didn't move much
            cosine_scores[i] = 0.0
        else:
            cosine = np.dot(vec_v, vec_r) / (norm_v * norm_r)
            cosine_scores[i] = cosine

    print(f"Direction Stats: Mean={cosine_scores.mean():.4f}, Min={cosine_scores.min():.4f}, Max={cosine_scores.max():.4f}")

    # 5. Save to PDB (Use Occupancy column for Direction)
    output_file = "internal_direction_heatmap.pdb"
    w = Chem.PDBWriter(output_file)
    conf = orig_mol.GetConformer()
    
    # Update positions to final frame
    for i in range(num_atoms):
        x, y, z = frame_end[i]
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        
    w.write(orig_mol)
    w.close()
    
    # Manually inject Occupancy values
    with open(output_file, 'r') as f:
        lines = f.readlines()
        
    with open(output_file, 'w') as f:
        atom_idx = 0
        for line in lines:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                if atom_idx < num_atoms:
                    # PDB Format: Occupancy is cols 54-60
                    val = cosine_scores[atom_idx]
                    val_str = f"{val:6.2f}"
                    
                    # Replace Occupancy column
                    # Standard PDB: ... X Y Z Occ Temp ...
                    # We assume standard RDKit output
                    # RDKit writes "  1.00" for occupancy usually
                    
                    if len(line) > 60:
                        new_line = line[:54] + val_str + line[60:]
                    else:
                        new_line = line
                        
                    f.write(new_line)
                    atom_idx += 1
                else:
                    f.write(line)
            else:
                f.write(line)

    print(f"Saved {output_file}")
    print("Instructions: Open in PyMOL.")
    print("Type: 'spectrum q, red_white_blue, selection, minimum=-1, maximum=1'")
    print("Blue (+1) = Moving TOWARDS pocket center (Attraction)")
    print("Red (-1) = Moving AWAY from pocket center (Repulsion/Adjustment)")
    print("White (0) = Moving sideways or not moving")
    
    # Print top atoms moving IN and OUT
    print("\nTop 3 Atoms moving INWARD (Attraction):")
    indices_in = np.argsort(-cosine_scores)
    for i in range(3):
        idx = indices_in[i]
        atom = orig_mol.GetAtomWithIdx(int(idx))
        print(f"  Atom {idx} ({atom.GetSymbol()}): {cosine_scores[idx]:.4f}")
        
    print("\nTop 3 Atoms moving OUTWARD (Repulsion):")
    indices_out = np.argsort(cosine_scores)
    for i in range(3):
        idx = indices_out[i]
        atom = orig_mol.GetAtomWithIdx(int(idx))
        print(f"  Atom {idx} ({atom.GetSymbol()}): {cosine_scores[idx]:.4f}")

if __name__ == "__main__":
    visualize_flow_direction("run_1stp_experiment", "1stp_mol0")
