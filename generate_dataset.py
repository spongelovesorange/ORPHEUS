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
    run_name = "run_orpheus_batch"
    base_path = "/data/Matcha-main/orpheus_physics"
    output_dir = f"{base_path}/inference_results/{run_name}/orpheus_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing ORPHEUS Data Factory...")
    print(f"Output Directory: {output_dir}")
    
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
            
            # Get Pocket Center
            # Try to get it from Stage 3 data
            sample_data_s3 = stages[3][key][0]
            if 'full_protein_center' in sample_data_s3:
                pocket_center = sample_data_s3['full_protein_center']
            else:
                # Fallback: Try to find PDB? 
                # For now, skip if no center (or use 0,0,0 but that's bad)
                print(f"Skipping {key}: No pocket center found in metadata.")
                continue
                
            # Process
            deformations, directions = process_trajectory(trajectory, pocket_center)
            
            # Save
            save_path = os.path.join(output_dir, f"{key}.pt")
            torch.save({
                'deformation': torch.tensor(deformations, dtype=torch.float32),
                'direction': torch.tensor(directions, dtype=torch.float32),
                'trajectory_shape': trajectory.shape,
                'pocket_center': pocket_center
            }, save_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {key}: {e}")
            
    print(f"Data Factory Finished!")
    print(f"Successfully generated {success_count} / {len(all_keys)} training samples.")
    print(f"Dataset location: {output_dir}")

if __name__ == "__main__":
    main()
