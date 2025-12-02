import numpy as np
from rdkit import Chem

# Load Ground Truth
gt_mol = Chem.MolFromPDBFile('/data/Matcha-main/data/my_dataset/1stp/1stp.pdb')
gt_conf = gt_mol.GetConformer()
gt_pos = gt_conf.GetPositions()
# Filter for ligand (assuming ligand is at the end or we need to find it)
# Actually, MolFromPDBFile might load protein + ligand.
# Let's use the ligand file we have: 1stp_ligand.sdf
gt_ligand = Chem.SDMolSupplier('/data/Matcha-main/data/my_dataset/1stp/1stp_ligand.sdf')[0]
gt_pos = gt_ligand.GetConformer().GetPositions()
gt_center = gt_pos.mean(axis=0)

# Load Docked Result (Last Frame)
# We can read the PDB we just made, or the .npy file.
# Let's read the .npy file to be precise.
data = np.load('/data/Matcha-main/inference_results/run_1stp_experiment/any_conf_final_preds.npy', allow_pickle=True)[0]
sample = data['1stp_mol0']['sample_metrics'][0]
pred_pos = sample['pred_pos']
pred_center = pred_pos.mean(axis=0)

dist = np.linalg.norm(gt_center - pred_center)

print(f"Ground Truth Center: {gt_center}")
print(f"Predicted Center:    {pred_center}")
print(f"Distance between centers: {dist:.4f} A")

# Check if there are other pockets?
# Calculate distance to protein center
protein = Chem.MolFromPDBFile('/data/Matcha-main/data/my_dataset/1stp/1stp_protein.pdb')
prot_conf = protein.GetConformer()
prot_pos = prot_conf.GetPositions()
prot_center = prot_pos.mean(axis=0)
print(f"Protein Center: {prot_center}")
