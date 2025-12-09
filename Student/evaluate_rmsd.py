import os
import glob
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import seq1

def get_ca_atoms_and_seq(structure):
    ca_atoms = []
    seq = []
    ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'])
                    seq.append(seq1(residue.get_resname()))
                    ids.append(residue.get_id())
    return ca_atoms, "".join(seq), ids

def calculate_rmsd(predicted_path, truth_path):
    parser = PDBParser(QUIET=True)
    pred_structure = parser.get_structure('pred', predicted_path)
    truth_structure = parser.get_structure('truth', truth_path)
    
    pred_ca, pred_seq, pred_ids = get_ca_atoms_and_seq(pred_structure)
    truth_ca, truth_seq, truth_ids = get_ca_atoms_and_seq(truth_structure)
    
    # If lengths differ, we need to align.
    # Since prediction is generated from a sequence that might be fuller than the PDB (which has gaps),
    # we should align the sequences to find the correspondence.
    
    # However, the prediction sequence comes from the dataset, which comes from the PDB?
    # If the dataset sequence is the full sequence, and PDB has gaps, we need to filter prediction.
    
    if len(pred_ca) != len(truth_ca):
        # Simple alignment by sequence
        from Bio import pairwise2
        alignments = pairwise2.align.globalxx(pred_seq, truth_seq)
        best_aln = alignments[0]
        aln_pred, aln_truth, score, begin, end = best_aln
        
        # Map atoms
        pred_atoms_aligned = []
        truth_atoms_aligned = []
        
        pred_idx = 0
        truth_idx = 0
        
        for p_char, t_char in zip(aln_pred, aln_truth):
            if p_char != '-' and t_char != '-':
                # Match (or mismatch, but aligned positions)
                if p_char == t_char:
                    pred_atoms_aligned.append(pred_ca[pred_idx])
                    truth_atoms_aligned.append(truth_ca[truth_idx])
                pred_idx += 1
                truth_idx += 1
            elif p_char != '-':
                pred_idx += 1
            elif t_char != '-':
                truth_idx += 1
                
        if len(pred_atoms_aligned) < 3:
            return None, 0, f"Too few aligned atoms: {len(pred_atoms_aligned)}"
            
        fixed = truth_atoms_aligned
        moving = pred_atoms_aligned
    else:
        fixed = truth_ca
        moving = pred_ca

    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    sup.apply(moving) # Modifies moving in place
    
    return sup.rms, len(fixed), None

def main():
    results_dir = "inference_results"
    ground_truth_root = "/data/ORPHEUS/Geometric_Teacher/data/PDBBind"
    
    pdb_files = glob.glob(os.path.join(results_dir, "*_predicted.pdb"))
    
    print(f"{'PDB ID':<10} | {'RMSD (A)':<10} | {'Aligned Atoms':<15} | {'Status':<20}")
    print("-" * 65)
    
    for pdb_file in pdb_files:
        pdb_id = os.path.basename(pdb_file).replace("_predicted.pdb", "")
        
        # Find ground truth
        # Pattern: .../P-L/*/<pdb_id>/<pdb_id>_protein.pdb
        pattern = os.path.join(ground_truth_root, "P-L", "*", pdb_id, f"{pdb_id}_protein.pdb")
        gt_files = glob.glob(pattern)
        
        if not gt_files:
            print(f"{pdb_id:<10} | {'N/A':<10} | {'N/A':<15} | {'GT not found'}")
            continue
            
        gt_file = gt_files[0]
        
        try:
            rmsd, n_atoms, error = calculate_rmsd(pdb_file, gt_file)
            if error:
                print(f"{pdb_id:<10} | {'N/A':<10} | {n_atoms:<15} | {error}")
            else:
                print(f"{pdb_id:<10} | {rmsd:<10.4f} | {n_atoms:<15} | {'Success'}")
        except Exception as e:
            print(f"{pdb_id:<10} | {'Error':<10} | {'-':<15} | {str(e)}")

if __name__ == "__main__":
    main()
