import numpy as np
from rdkit import Chem
import os

def load_stage_data(stage_num, run_name, dataset_name='any_conf'):
    """读取指定 Stage 的数据文件"""
    filename = f"/data/Matcha-main/inference_results/{run_name}/stage{stage_num}_{dataset_name}.npy"
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

    # Protein path
    protein_path = "/data/Matcha-main/data/my_dataset/1stp/1stp_protein.pdb"

    # 1. 获取原始分子结构
    final_preds_path = f"/data/Matcha-main/inference_results/{run_name}/any_conf_final_preds.npy"
    if not os.path.exists(final_preds_path):
        print("Final predictions not found.")
        return
        
    final_data = np.load(final_preds_path, allow_pickle=True)[0]
    
    if base_name not in final_data:
        print(f"Error: {base_name} not found in final predictions.")
        return

    ligand_mol = final_data[base_name]['orig_mol']

    # Load Protein
    complex_mol = ligand_mol
    protein_atom_count = 0
    
    if os.path.exists(protein_path):
        protein_mol = Chem.MolFromPDBFile(protein_path, removeHs=False, sanitize=False)
        if protein_mol:
            try:
                complex_mol = Chem.CombineMols(protein_mol, ligand_mol)
                protein_atom_count = protein_mol.GetNumAtoms()
            except:
                pass

    # 2. 读取三个阶段的轨迹并拼接
    full_trajectory = []
    
    for stage in [1, 2, 3]:
        stage_data = load_stage_data(stage, run_name)
        if full_key in stage_data and 'trajectory' in stage_data[full_key][0]:
            traj = stage_data[full_key][0]['trajectory']
            full_trajectory.append(traj)

    if not full_trajectory:
        print("No trajectory data found.")
        return

    combined_traj = np.concatenate(full_trajectory, axis=0)
    print(f"Combined trajectory shape: {combined_traj.shape}")

    # 3. 写入 PDB 动画
    output_file = "full_docking_with_protein.pdb"
    w = Chem.PDBWriter(output_file)
    
    conf = complex_mol.GetConformer()
    
    for i in range(combined_traj.shape[0]):
        coords = combined_traj[i]
        for j in range(coords.shape[0]):
            atom_idx = protein_atom_count + j
            x, y, z = coords[j]
            conf.SetAtomPosition(atom_idx, (float(x), float(y), float(z)))
        w.write(complex_mol)
    
    w.close()
    print(f"Saved {output_file}")

if __name__ == "__main__":
    main()
