import numpy as np
from rdkit import Chem
# from matcha.utils.spyrmsd import rmsd, coords_to_molecule # Removed incorrect import

def visualize_internal_deformation(run_name, base_name):
    # Updated path
    base_path = "/data/Matcha-main/orpheus_physics"
    # 1. Load Trajectory (Stage 3 - Fine tuning stage is best for internal forces)
    filename = f"{base_path}/inference_results/{run_name}/stage3_any_conf.npy"
    try:
        data = np.load(filename, allow_pickle=True)[0]
        key = f"{base_name}_conf0"
        if key not in data:
            print(f"Key {key} not found.")
            return
        
        sample = data[key][0]
        if 'trajectory' not in sample:
            print("Trajectory not found.")
            return
            
        traj = sample['trajectory'] # (11, N, 3)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Load Original Molecule
    final_preds = np.load(f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy", allow_pickle=True)[0]
    orig_mol = final_preds[base_name]['orig_mol']

    # 3. Calculate Internal Deformation
    # Strategy: Align every frame to the previous frame. 
    # The residual distance is the "internal deformation" or "non-rigid movement".
    
    num_frames = traj.shape[0]
    num_atoms = traj.shape[1]
    
    # Accumulate deformation for each atom
    total_deformation = np.zeros(num_atoms)
    
    # Helper for Kabsch alignment
    def get_rmsd_and_aligned_pos(mobile, target):
        # Center
        mob_center = mobile.mean(axis=0)
        tar_center = target.mean(axis=0)
        mob_c = mobile - mob_center
        tar_c = target - tar_center
        
        # Rotation
        H = np.dot(mob_c.T, tar_c)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = np.dot(Vt.T, U.T)
            
        # Align mobile to target
        mobile_aligned = np.dot(mob_c, R) + tar_center # Keep target frame
        
        # Distance per atom
        dists = np.linalg.norm(mobile_aligned - target, axis=1)
        return dists

    print("Calculating internal deformation across trajectory...")
    for i in range(num_frames - 1):
        frame_curr = traj[i]
        frame_next = traj[i+1]
        
        # Align current to next to remove rigid body motion
        dists = get_rmsd_and_aligned_pos(frame_curr, frame_next)
        
        # Accumulate
        total_deformation += dists

    # Normalize? No, raw magnitude is fine.
    mean_def = total_deformation.mean()
    max_def = total_deformation.max()
    print(f"Deformation Stats: Mean={mean_def:.4f}, Max={max_def:.4f}")
    
    # 4. Visualize with B-Factor trick
    # We will save a PDB where the "B-Factor" column is replaced by our deformation value.
    # In PyMOL, you can color by B-factor: "spectrum b, blue_white_red"
    
    output_file = "internal_deformation_heatmap.pdb"
    w = Chem.PDBWriter(output_file)
    
    # Use the final conformation
    final_pos = traj[-1]
    conf = orig_mol.GetConformer()
    for i in range(num_atoms):
        x, y, z = final_pos[i]
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        
        # Set B-Factor (RDKit PDBWriter doesn't support setting B-factor directly easily via Mol object properties in all versions)
        # But we can use the 'SetAtomPDBResidueInfo' if needed, or just write manually.
        # Actually, RDKit's PDBWriter has no direct SetBFactor method on Atom.
        # We will write a custom PDB modifier or use a property.
        
        atom = orig_mol.GetAtomWithIdx(i)
        # Store in a temp property to print later if needed
        atom.SetDoubleProp("deformation", float(total_deformation[i]))

    w.write(orig_mol)
    w.close()
    
    # Hack: Manually edit the PDB file to insert B-factors
    # RDKit writes B-factors as 0.00 by default.
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
        
    with open(output_file, 'w') as f:
        atom_idx = 0
        for line in lines:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                if atom_idx < num_atoms:
                    # PDB format: B-factor is columns 61-66
                    # Line is a string.
                    # We need to be careful with fixed width.
                    
                    # Get the deformation value
                    val = total_deformation[atom_idx]
                    
                    # Format: %6.2f
                    val_str = f"{val:6.2f}"
                    
                    # Reconstruct line
                    # Standard PDB: 
                    # 0-6: Record name
                    # ...
                    # 60-66: Temp factor (B-factor)
                    
                    if len(line) > 66:
                        new_line = line[:60] + val_str + line[66:]
                    else:
                        # Pad if short
                        new_line = line[:60] + val_str + "\n"
                        
                    f.write(new_line)
                    atom_idx += 1
                else:
                    f.write(line)
            else:
                f.write(line)
                
    print(f"Saved {output_file}")
    print("Instructions: Open in PyMOL, then type: 'spectrum b, blue_white_red, selection'")
    print("Red atoms = High internal deformation (Key interactions?)")
    print("Blue atoms = Rigid/Low deformation")

    # Print top atoms
    indices = np.argsort(-total_deformation)
    print("\nTop 5 Atoms with highest internal deformation:")
    for i in range(5):
        idx = indices[i]
        atom = orig_mol.GetAtomWithIdx(int(idx))
        print(f"  Atom {idx} ({atom.GetSymbol()}): {total_deformation[idx]:.4f}")

if __name__ == "__main__":
    visualize_internal_deformation("run_1stp_experiment", "1stp_mol0")
