import numpy as np
import torch
import os
from tqdm import tqdm
from rdkit import Chem

def get_rmsd_and_aligned_displacement(mobile, target):
    """
    Calculate 'Internal Deformation' displacement after removing rigid body motion.
    """
    # Center
    mob_center = mobile.mean(axis=0)
    tar_center = target.mean(axis=0)
    mob_c = mobile - mob_center
    tar_c = target - tar_center
    
    # Rotation (Kabsch algorithm)
    H = np.dot(mob_c.T, tar_c)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)
        
    # Align mobile to target frame
    mobile_aligned = np.dot(mob_c, R) + tar_center 
    
    # Vector from aligned mobile to target
    displacement_vectors = target - mobile_aligned
    displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=1)
    
    return displacement_magnitudes, displacement_vectors

def calculate_direction_cosine(coords, vectors, pocket_center):
    """
    Calculate Cosine Similarity between displacement vector and 'Atom-to-Pocket' vector.
    """
    # Vector from atom to pocket center
    atom_to_center = pocket_center - coords # (N, 3)
    
    # Normalize
    v_norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    atc_norm = np.linalg.norm(atom_to_center, axis=1, keepdims=True) + 1e-8
    
    vectors_n = vectors / v_norm
    atc_n = atom_to_center / atc_norm
    
    # Dot product
    cosine_sim = np.sum(vectors_n * atc_n, axis=1)
    return cosine_sim

def process_trajectory(traj, pocket_center):
    """
    Process a single trajectory to extract physics features.
    traj: (T, N, 3)
    """
    num_frames = traj.shape[0]
    
    deformations = []
    directions = []
    
    for i in range(num_frames - 1):
        frame_curr = traj[i]
        frame_next = traj[i+1]
        
        # 1. Internal Deformation (Level 1 Feature)
        mag, vec = get_rmsd_and_aligned_displacement(frame_curr, frame_next)
        deformations.append(mag)
        
        # 2. Directionality (Level 2 Feature)
        cosine = calculate_direction_cosine(frame_curr, vec, pocket_center)
        directions.append(cosine)
        
    return np.array(deformations), np.array(directions)

def main():
    run_name = "run_orpheus_20"
    base_path = "/data/Matcha-main/orpheus_physics"
    output_dir = f"{base_path}/inference_results/{run_name}/orpheus_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing ORPHEUS Data Factory...")
    print(f"Output Directory: {output_dir}")
    
    # Load Final Preds to get Molecules (for atomic numbers)
    final_preds_path = f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy"
    print(f"Loading molecules from {final_preds_path}...")
    try:
        final_preds = np.load(final_preds_path, allow_pickle=True)[0]
    except:
        print("Could not load final predictions. Atomic numbers will be missing.")
        final_preds = {}

    # 1. Load All Stages
    stages = {}
    for s in [1, 2, 3]:
        path = f"{base_path}/inference_results/{run_name}/stage{s}_any_conf.npy"
        if os.path.exists(path):
            print(f"Loading Stage {s}...")
            stages[s] = np.load(path, allow_pickle=True)[0]
        else:
            print(f"Error: Stage {s} data not found at {path}")
            return

    # 2. Identify all samples
    # We use Stage 3 keys as the master list
    all_keys = list(stages[3].keys())
    print(f"Found {len(all_keys)} samples to process.")
    
    # 3. Batch Processing
    success_count = 0
    
    for key in tqdm(all_keys):
        try:
            # Stitch Trajectory
            full_traj_list = []
            for s in [1, 2, 3]:
                if key in stages[s]:
                    # stage data is a list of metrics, usually length 1 for inference
                    sample_data = stages[s][key][0]
                    if 'trajectory' in sample_data:
                        full_traj_list.append(sample_data['trajectory'])
            
            if len(full_traj_list) != 3:
                print(f"Skipping {key}: Incomplete trajectory (found {len(full_traj_list)} stages)")
                continue
                
            trajectory = np.concatenate(full_traj_list, axis=0) # (33, N, 3)
            
            # Extract PDB ID first
            parts = key.split("_")
            pdb_id = parts[0]
            
            # Get Pocket Center
            # FIX: Use the Ground Truth Ligand to define the True Pocket Center
            # This ensures we always find the protein pocket, even if the trajectory is far away.
            
            gt_ligand_path = f"{base_path}/data/my_dataset/{pdb_id}/{pdb_id}_ligand.sdf"
            true_pocket_center = None
            
            if os.path.exists(gt_ligand_path):
                try:
                    gt_mol = Chem.MolFromMolFile(gt_ligand_path)
                    if gt_mol:
                        gt_conf = gt_mol.GetConformer()
                        gt_pos = gt_conf.GetPositions()
                        true_pocket_center = gt_pos.mean(axis=0)
                except:
                    pass
            
            if true_pocket_center is None:
                # Fallback to trajectory end (risky if trajectory is bad)
                print(f"Warning {key}: Could not load GT ligand. Using trajectory end as pocket center.")
                true_pocket_center = trajectory[-1].mean(axis=0)
            
            pocket_center = true_pocket_center
                
            # Process
            deformations, directions = process_trajectory(trajectory, pocket_center)

            # Get Atomic Numbers
            # key format: "1stp_mol0_conf0" -> base_name "1stp_mol0"
            parts = key.split("_")
            pdb_id = parts[0]
            base_name = "_".join(parts[:-1])
            
            atomic_numbers = None
            bond_index = None
            
            if base_name in final_preds:
                mol = final_preds[base_name]['orig_mol']
                atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                
                # Get Bonds
                bonds = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bonds.append([i, j])
                    bonds.append([j, i]) # Undirected
                
                if len(bonds) > 0:
                    bond_index = torch.tensor(bonds, dtype=torch.long).t()
                else:
                    bond_index = torch.empty((2, 0), dtype=torch.long)

                # Verify length
                if len(atomic_numbers) != trajectory.shape[1]:
                    print(f"Warning: Atom count mismatch for {key}. Traj: {trajectory.shape[1]}, Mol: {len(atomic_numbers)}")
                    atomic_numbers = [6] * trajectory.shape[1] # Fallback to Carbon
                    bond_index = torch.empty((2, 0), dtype=torch.long) # Invalid bonds
            else:
                # Fallback
                atomic_numbers = [6] * trajectory.shape[1]
                bond_index = torch.empty((2, 0), dtype=torch.long)

            # Get Pocket Atoms (Protein Context)
            # Path: data/my_dataset/{pdb_id}/{pdb_id}_protein.pdb
            protein_path = f"{base_path}/data/my_dataset/{pdb_id}/{pdb_id}_protein.pdb"
            pocket_pos = []
            pocket_atoms = []
            
            if os.path.exists(protein_path):
                try:
                    prot_mol = Chem.MolFromPDBFile(protein_path, sanitize=False)
                    if prot_mol:
                        prot_conf = prot_mol.GetConformer()
                        prot_pos = prot_conf.GetPositions()
                        prot_atomic_nums = [atom.GetAtomicNum() for atom in prot_mol.GetAtoms()]
                        
                        # Debug: Check distances
                        dists = np.linalg.norm(prot_pos - pocket_center, axis=1)
                        min_dist = np.min(dists)
                        print(f"Debug {key}: Min dist to protein: {min_dist:.2f} A. Pocket Center: {pocket_center}")
                        
                        mask = dists < 10.0
                        
                        pocket_pos = prot_pos[mask]
                        pocket_atoms = np.array(prot_atomic_nums)[mask]
                        
                        print(f"Debug {key}: Found {len(pocket_pos)} pocket atoms within 10A.")
                        
                        # Limit max pocket atoms to avoid OOM (e.g. 200)
                        if len(pocket_pos) > 200:
                            # Take closest 200
                            indices = np.argsort(dists[mask])[:200]
                            pocket_pos = pocket_pos[indices]
                            pocket_atoms = pocket_atoms[indices]
                            
                except Exception as e:
                    print(f"Error reading protein {pdb_id}: {e}")
            else:
                print(f"Warning: Protein file not found: {protein_path}")

            # Save
            save_path = os.path.join(output_dir, f"{key}.pt")
            
            torch.save({
                'deformation': torch.tensor(deformations, dtype=torch.float32),
                'direction': torch.tensor(directions, dtype=torch.float32),
                'trajectory': torch.tensor(trajectory, dtype=torch.float32),
                'pocket_center': pocket_center,
                'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
                'bond_index': bond_index,
                'pocket_pos': torch.tensor(pocket_pos, dtype=torch.float32),
                'pocket_atoms': torch.tensor(pocket_atoms, dtype=torch.long),
                'sample_id': key
            }, save_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {key}: {e}")
            
    print(f"Data Factory Finished!")
    print(f"Successfully generated {success_count} / {len(all_keys)} training samples.")
    print(f"Dataset location: {output_dir}")

if __name__ == "__main__":
    main()
