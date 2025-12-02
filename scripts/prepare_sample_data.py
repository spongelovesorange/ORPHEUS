import os
import urllib.request
import prody
from rdkit import Chem

def prepare_1stp():
    dataset_dir = "/data/Matcha-main/data/my_dataset/1stp"
    os.makedirs(dataset_dir, exist_ok=True)
    
    pdb_code = "1stp"
    pdb_url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    pdb_file = os.path.join(dataset_dir, f"{pdb_code}.pdb")
    
    print(f"Downloading {pdb_code} from RCSB...")
    try:
        urllib.request.urlretrieve(pdb_url, pdb_file)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    print("Parsing PDB file...")
    pdb = prody.parsePDB(pdb_file)
    
    # 1. Extract Protein (Chain A for simplicity)
    # Matcha handles multi-chain, but for a clean sample let's take Chain A
    print("Extracting Protein (Chain A)...")
    protein = pdb.select('protein and chain A')
    prody.writePDB(os.path.join(dataset_dir, f"{pdb_code}_protein.pdb"), protein)
    
    # 2. Extract Ligand (BTN in Chain A)
    print("Extracting Ligand (BTN)...")
    # We extract the atoms from PDB to preserve coordinates
    ligand_pdb = os.path.join(dataset_dir, "temp_ligand.pdb")
    ligand_sel = pdb.select('resname BTN and chain A')
    
    if ligand_sel is None:
        print("Error: Could not find ligand BTN in Chain A")
        return
        
    prody.writePDB(ligand_pdb, ligand_sel)
    
    # Convert to SDF using RDKit
    # Note: MolFromPDBFile often struggles with bond orders from PDBs. 
    # For Biotin it might work, or we might need a template.
    mol = Chem.MolFromPDBFile(ligand_pdb)
    
    if mol is None:
        print("Warning: RDKit failed to parse PDB ligand directly. Trying with SMILES template...")
        # Biotin SMILES
        smiles = "C1[C@@H]2[C@@H]([C@@H](S1)CCCCC(=O)O)NC(=O)N2"
        template = Chem.MolFromSmiles(smiles)
        mol = Chem.MolFromPDBFile(ligand_pdb)
        if mol is None:
             # If still failing, we just use the SMILES and embed (Loss of Ground Truth)
             print("Could not recover crystal coordinates perfectly. Generating from SMILES (RMSD will be invalid).")
             mol = Chem.AddHs(template)
             Chem.AllChem.EmbedMolecule(mol)
        else:
            try:
                mol = Chem.AllChem.AssignBondOrdersFromTemplate(template, mol)
            except:
                pass
    
    if mol is not None:
        sdf_file = os.path.join(dataset_dir, f"{pdb_code}_ligand.sdf")
        w = Chem.SDWriter(sdf_file)
        w.write(mol)
        w.close()
        print(f"Saved Ground Truth Ligand to: {sdf_file}")
    
    # Clean up
    if os.path.exists(ligand_pdb):
        os.remove(ligand_pdb)
        
    print("\nDone! Data is ready in:", dataset_dir)
    print("You can now run inference.")

if __name__ == "__main__":
    prepare_1stp()
