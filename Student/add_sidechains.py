import sys
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller, ForceField, PDBxFile
from openmm import unit
import os

def add_sidechains(input_pdb, output_pdb):
    print(f"Adding sidechains to {input_pdb}...")
    
    # 1. Load PDB with PDBFixer
    fixer = PDBFixer(filename=input_pdb)
    
    # 2. Find Missing Residues (Sidechains)
    # PDBFixer automatically detects missing atoms based on residue names
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    
    # 3. Add Missing Atoms (Sidechains)
    fixer.addMissingAtoms()
    
    # 4. Add Hydrogens (Optional but recommended for energy minimization)
    fixer.addMissingHydrogens(7.0)
    
    # 5. Save
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    print(f"Saved full-atom structure to {output_pdb}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_sidechains.py <input_pdb> <output_pdb>")
        sys.exit(1)
        
    add_sidechains(sys.argv[1], sys.argv[2])
