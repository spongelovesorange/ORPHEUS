import os
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# List of 20 classic PDBBind complexes (small, clean structures)
PDB_IDS = [
    "1a0q", "1a28", "1a4k", "1a4w", "1a5v", 
    "1a94", "1a99", "1a9m", "1a9q", "1aaq",
    "1bcu", "1bma", "1c1r", "1cet", "1e66",
    "1gpk", "1hvy", "1l2s", "1lic", "1stp" # 1stp is already there but fine
]

BASE_DIR = "/data/Matcha-main/orpheus_physics/data/my_dataset"

def download_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download {pdb_id}")
        return None

def process_pdb(pdb_id, pdb_content):
    output_dir = os.path.join(BASE_DIR, pdb_id)
    os.makedirs(output_dir, exist_ok=True)
    
    raw_pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    with open(raw_pdb_path, "w") as f:
        f.write(pdb_content)
        
    # Load into RDKit
    mol = Chem.MolFromPDBFile(raw_pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        print(f"RDKit failed to parse {pdb_id}")
        return False

    # Split Protein and Ligand
    # Strategy: Ligands are usually HETATM. Proteins are ATOM.
    # But RDKit MolFromPDBFile combines them.
    # We need to identify the largest organic molecule as ligand.
    
    # Better approach: Use PDBBlock to separate chains/residues? 
    # Simple heuristic: Extract HETATM records that are NOT water (HOH).
    
    # Let's try a simpler way: 
    # 1. Save Protein: Keep only standard residues
    # 2. Save Ligand: Keep the largest HETATM residue
    
    lines = pdb_content.split('\n')
    protein_lines = []
    hetatm_lines = []
    
    for line in lines:
        if line.startswith("ATOM"):
            protein_lines.append(line)
        elif line.startswith("HETATM"):
            res_name = line[17:20].strip()
            if res_name != "HOH": # Skip water
                hetatm_lines.append(line)
        elif line.startswith("END") or line.startswith("TER"):
            protein_lines.append(line)
            
    # Save Protein
    prot_path = os.path.join(output_dir, f"{pdb_id}_protein.pdb")
    with open(prot_path, "w") as f:
        f.write("\n".join(protein_lines))
        
    # Save Ligand (Temporary PDB)
    lig_pdb_path = os.path.join(output_dir, f"{pdb_id}_ligand_tmp.pdb")
    with open(lig_pdb_path, "w") as f:
        f.write("\n".join(hetatm_lines))
        
    # Convert Ligand PDB to SDF (and clean up)
    try:
        lig_mol = Chem.MolFromPDBFile(lig_pdb_path, removeHs=False, sanitize=False)
        if lig_mol is None:
            print(f"Failed to load ligand for {pdb_id}")
            return False
            
        # Sanitize and 3D conformer
        try:
            Chem.SanitizeMol(lig_mol)
        except:
            pass
            
        # Split into fragments (in case multiple ligands/cofactors)
        frags = Chem.GetMolFrags(lig_mol, asMols=True)
        # Pick largest fragment by atom count
        lig_mol = max(frags, key=lambda m: m.GetNumAtoms())
        
        sdf_path = os.path.join(output_dir, f"{pdb_id}_ligand.sdf")
        w = Chem.SDWriter(sdf_path)
        w.write(lig_mol)
        w.close()
        
        os.remove(lig_pdb_path) # Clean up tmp
        return True
        
    except Exception as e:
        print(f"Error processing ligand for {pdb_id}: {e}")
        return False

def main():
    print(f"Downloading and processing {len(PDB_IDS)} PDBs...")
    
    success_count = 0
    for pdb_id in tqdm(PDB_IDS):
        content = download_pdb(pdb_id)
        if content:
            if process_pdb(pdb_id, content):
                success_count += 1
                
    print(f"Successfully prepared {success_count}/{len(PDB_IDS)} datasets.")
    print(f"Data stored in {BASE_DIR}")

if __name__ == "__main__":
    main()
