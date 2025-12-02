import numpy as np
from rdkit import Chem

def check_stage3_direction():
    # Load Stage 3 only
    run_name = "run_1stp_experiment"
    base_name = "1stp_mol0"
    base_path = "/data/Matcha-main/orpheus_physics"
    filename = f"{base_path}/inference_results/{run_name}/stage3_any_conf.npy"
    data = np.load(filename, allow_pickle=True)[0]
    traj = data[f"{base_name}_conf0"][0]['trajectory']
    
    # Pocket Center (from previous script output or approximation)
    # In visualize_direction.py we used the protein center.
    # Let's use the same one: [16.0, 25.0, 4.0] approx or calculate.
    prot = Chem.MolFromPDBFile(f"{base_path}/data/my_dataset/1stp/1stp_protein.pdb")
    pocket_center = prot.GetConformer().GetPositions().mean(axis=0)
    
    # Align Start to End (Stage 3)
    frame_start = traj[0]
    frame_end = traj[-1]
    
    # Kabsch
    mob_center = frame_start.mean(axis=0)
    tar_center = frame_end.mean(axis=0)
    mob_c = frame_start - mob_center
    tar_c = frame_end - tar_center
    H = np.dot(mob_c.T, tar_c)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)
    start_aligned = np.dot(mob_c, R) + tar_center
    
    # Net Vector
    vectors = frame_end - start_aligned
    
    # Cosine
    print(f"Stage 3 Net Displacement Analysis:")
    print("-" * 40)
    print(f"{'Atom':<5} {'Symbol':<5} {'Cosine':<10} {'Interpretation'}")
    
    final_preds = np.load(f"{base_path}/inference_results/{run_name}/any_conf_final_preds.npy", allow_pickle=True)[0]
    mol = final_preds[base_name]['orig_mol']
    
    target_atoms = [4, 8]
    
    for i in range(mol.GetNumAtoms()):
        atom_sym = mol.GetAtomWithIdx(i).GetSymbol()
        vec_v = vectors[i]
        vec_r = pocket_center - frame_end[i]
        
        norm_v = np.linalg.norm(vec_v)
        norm_r = np.linalg.norm(vec_r)
        
        if norm_v < 1e-3:
            cosine = 0.0
        else:
            cosine = np.dot(vec_v, vec_r) / (norm_v * norm_r)
            
        marker = ""
        if i in target_atoms:
            marker = "<-- CHECK"
            
        print(f"{i:<5} {atom_sym:<5} {cosine:<10.4f} {marker}")

if __name__ == "__main__":
    check_stage3_direction()
