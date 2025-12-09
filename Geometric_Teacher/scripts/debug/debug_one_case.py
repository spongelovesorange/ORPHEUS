
import os
import torch
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial.distance import cdist
from graphein.protein.resi_atoms import PROTEIN_ATOMS
from torch_geometric.data import Data, Batch

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Mocking the model parts to test data pipeline first
def get_backbone_coords_and_seq(pdb_path):
    print(f"Parsing PDB: {pdb_path}")
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', pdb_path)
    except Exception as e:
        print(f"Failed to parse PDB: {e}")
        return None, None

    coords = []
    seq = []
    
    for model in structure:
        for chain in model:
            print(f"Checking chain {chain.id}...")
            chain_coords = []
            chain_seq = []
            for residue in chain:
                if residue.get_resname() not in THREE_TO_ONE:
                    continue
                try:
                    n = residue['N'].get_vector()
                    ca = residue['CA'].get_vector()
                    c = residue['C'].get_vector()
                    o = residue['O'].get_vector()
                    chain_coords.append([[n[0], n[1], n[2]], [ca[0], ca[1], ca[2]], [c[0], c[1], c[2]], [o[0], o[1], o[2]]])
                    chain_seq.append(THREE_TO_ONE[residue.get_resname()])
                except KeyError:
                    continue
            
            if chain_coords:
                print(f"Found valid chain {chain.id} with {len(chain_coords)} residues.")
                coords = chain_coords
                seq = chain_seq
                break
        if coords:
            break
            
    if not coords:
        print("No coords found.")
        return None, None
        
    return torch.tensor(coords, dtype=torch.float32), "".join(seq)

def test_pipeline():
    pdb_path = "/data/ORPHEUS/Geometric_Teacher/data/PDBBind/P-L/2011-2019/2l3r/2l3r_protein.pdb"
    ligand_path = "/data/ORPHEUS/Geometric_Teacher/data/PDBBind/P-L/2011-2019/2l3r/2l3r_ligand.sdf"
    
    print(f"--- Testing Data Loading ---")
    coords, seq = get_backbone_coords_and_seq(pdb_path)
    if coords is None:
        print("FAIL: PDB parsing failed.")
        return

    print(f"PDB Parsed. Seq len: {len(seq)}")
    print(f"Coords shape: {coords.shape}")
    
    print(f"--- Testing Ligand Loading ---")
    suppl = Chem.SDMolSupplier(ligand_path)
    ligand = suppl[0]
    if ligand is None:
        print("FAIL: Ligand parsing failed.")
        return
    lig_coords = ligand.GetConformer().GetPositions()
    print(f"Ligand Parsed. Atoms: {len(lig_coords)}")
    
    print(f"--- Testing Distance Mask ---")
    ca_coords = coords[:, 1].numpy()
    dists = cdist(ca_coords, lig_coords)
    min_dists = np.min(dists, axis=1)
    print(f"Min dists range: {np.min(min_dists):.2f} - {np.max(min_dists):.2f}")
    
    is_pocket = min_dists < 6.0
    num_pocket = np.sum(is_pocket)
    print(f"Pocket residues found: {num_pocket}")
    
    if num_pocket == 0:
        print("FAIL: No pocket residues found.")
        return

    print("--- Data Pipeline OK ---")
    
    # Check PROTEIN_ATOMS
    print(f"First 4 PROTEIN_ATOMS: {PROTEIN_ATOMS[:4]}")

if __name__ == "__main__":
    test_pipeline()
