import numpy as np
from rdkit import Chem
import os

def load_stage_data(stage_num, run_name, dataset_name='any_conf'):
    """读取指定 Stage 的数据文件"""
    base_path = "/data/Matcha-main/orpheus_physics"
    filename = f"{base_path}/inference_results/{run_name}/stage{stage_num}_{dataset_name}.npy"
    if not os.path.exists(filename):
        print(f"Warning: {filename} does not exist.")
        return {}
    print(f"Loading Stage {stage_num} from {filename}...")
    return np.load(filename, allow_pickle=True)[0]

def main():
    run_name = "run_1stp_experiment"
    base_name = "1stp_mol0"
    conf_id = "conf0" 
    full_key = f"{base_name}_{conf_id}"
    
    base_path = "/data/Matcha-main/orpheus_physics"

    # Protein path
    protein_path = f"{base_path}/data/my_dataset/1stp/1stp_protein.pdb"

    # 1. 获取原始分子结构
    final_preds_path = f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy"
    if not os.path.exists(final_preds_path):
        print("Final predictions not found.")
        return
        
    final_data = np.load(final_preds_path, allow_pickle=True)[0]

def main():
    run_name = "run_1stp_experiment"
    base_name = "1stp_mol0"
    conf_id = "conf0" # 我们追踪第0个样本的完整轨迹
    full_key = f"{base_name}_{conf_id}"

    # Protein path
    protein_path = "/data/Matcha-main/data/my_dataset/1stp/1stp_protein.pdb"

    # 1. 获取原始分子结构 (用于写 PDB)
    # 必须从 final_preds 里拿，因为 stage 文件里通常不存 orig_mol 以节省空间
    final_preds_path = f"/data/Matcha-main/inference_results/{run_name}/any_conf_final_preds.npy"
    final_data = np.load(final_preds_path, allow_pickle=True)[0]
    
    if base_name not in final_data:
        print(f"Error: {base_name} not found in final predictions.")
        return

    ligand_mol = final_data[base_name]['orig_mol']
    print(f"Loaded ligand structure for {base_name}")

    # Load Protein
    complex_mol = ligand_mol
    protein_atom_count = 0
    
    if os.path.exists(protein_path):
        print(f"Loading protein from {protein_path}...")
        protein_mol = Chem.MolFromPDBFile(protein_path, removeHs=False)
        if protein_mol is None:
             # Try sanitization off if it fails
            protein_mol = Chem.MolFromPDBFile(protein_path, removeHs=False, sanitize=False)
        
        if protein_mol:
            print(f"Protein loaded. Atoms: {protein_mol.GetNumAtoms()}")
            # Combine Protein + Ligand
            # Note: CombineMols puts the first mol first, then the second.
            try:
                complex_mol = Chem.CombineMols(protein_mol, ligand_mol)
                protein_atom_count = protein_mol.GetNumAtoms()
                print(f"Complex created. Total atoms: {complex_mol.GetNumAtoms()} (Ligand starts at index {protein_atom_count})")
            except Exception as e:
                print(f"Error combining mols: {e}. Using ligand only.")
                complex_mol = ligand_mol
        else:
            print("Failed to load protein PDB. Output will only contain ligand.")
    else:
        print("Protein file not found. Output will only contain ligand.")

    # 2. 读取三个阶段的轨迹并拼接
    full_trajectory = []
    
    for stage in [1, 2, 3]:
        stage_data = load_stage_data(stage, run_name)
        
        if full_key in stage_data:
            # stage_data[key] 是一个 list，通常只有一个元素（因为是针对单个 conf 的评估）
            sample_metrics = stage_data[full_key][0]
            
            if 'trajectory' in sample_metrics:
                traj = sample_metrics['trajectory']
                print(f"  Stage {stage}: Found trajectory with {traj.shape[0]} steps.")
                full_trajectory.append(traj)
            else:
                print(f"  Stage {stage}: No trajectory found!")
        else:
            print(f"  Stage {stage}: Key {full_key} not found.")

    if not full_trajectory:
        print("No trajectory data found.")
        return

    # 3. 拼接轨迹
    # full_trajectory 是 [ (11, N, 3), (11, N, 3), (11, N, 3) ]
    # 我们直接在时间轴 (axis 0) 上拼接
    combined_traj = np.concatenate(full_trajectory, axis=0)
    print(f"Combined trajectory shape: {combined_traj.shape}")

    # 4. 写入 PDB 动画
    output_file = "full_docking_with_protein.pdb"
    w = Chem.PDBWriter(output_file)
    
    print(f"Writing animation to {output_file}...")
    
    conf = complex_mol.GetConformer()
    
    for i in range(combined_traj.shape[0]):
        coords = combined_traj[i]
        
        # Update ligand atom positions
        # Ligand atoms are from index `protein_atom_count` to end
        for j in range(coords.shape[0]):
            atom_idx = protein_atom_count + j
            x, y, z = coords[j]
            conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))
        
        w.write(complex_mol)
    
    w.close()
    print(f"Done! Download {output_file} to view the full docking movie with protein.")

if __name__ == "__main__":
    main()
