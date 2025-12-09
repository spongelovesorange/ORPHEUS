import os
import torch
import json
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import EsmTokenizer
from model import StudentModel

# Import Teacher Model components for decoding
import sys
sys.path.append("/data/ORPHEUS/Geometric_Teacher") # Add teacher path
from utils.utils import load_checkpoints_simple
from box import Box
import yaml
from graphein.protein.tensor.io import protein_to_pyg
from graphein.protein.tensor.data import ProteinBatch

# Define prepare_model_custom locally since it was defined in generate_geometry_labels.py
# and not in super_model.py
from models.gcpnet.models.base import instantiate_encoder, PretrainedEncoder
from models.decoders import GeometricDecoder
from models.vqvae import VQVAETransformer
from models.super_model import SuperModel

def prepare_model_custom(configs, logger, encoder_config_path, decoder_configs, **kwargs):
    # Re-implementation of the custom loader from generate_geometry_labels.py
    log_details = configs.model.get('log_details', True)

    if not kwargs.get("decoder_only", False):
        if configs.model.encoder.name == "gcpnet":
            if configs.model.encoder.pretrained.enabled:
                # We don't need to load pretrained weights here, just structure
                # Weights will be loaded by load_checkpoints_simple later
                ret = instantiate_encoder(encoder_config_path)
                print(f"DEBUG: instantiate_encoder returned type: {type(ret)}")
                if isinstance(ret, tuple):
                    print(f"DEBUG: tuple length: {len(ret)}")
                    print(f"DEBUG: first element type: {type(ret[0])}")
                
                components, _ = ret
                encoder = PretrainedEncoder(components)
            else:
                components, _ = instantiate_encoder(encoder_config_path)
                encoder = PretrainedEncoder(components)
        else:
            raise ValueError("Invalid encoder model specified!")
    else:
        encoder = None

    if configs.model.vqvae.decoder.name == "geometric_decoder":
        decoder = GeometricDecoder(configs, decoder_configs=decoder_configs)
    else:
        raise ValueError("Invalid decoder model specified!")

    vqvae = VQVAETransformer(
        decoder=decoder,
        configs=configs,
        logger=logger,
        decoder_only=kwargs.get("decoder_only", False),
    )

    vqvae = SuperModel(encoder, vqvae, configs, decoder_only=kwargs.get("decoder_only", False))

    return vqvae


def load_teacher_decoder(config_path, checkpoint_path, device):
    """Load the VQVAE Decoder to convert tokens back to structure."""
    with open(config_path) as f:
        config = Box(yaml.full_load(f))
    
    # We need encoder config just to init the model class, though we won't use encoder
    # Use the configs from the checkpoint directory to ensure model size matches
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # If checkpoint_path is .../checkpoints/best_valid.pth, then dirname is .../checkpoints
    # Configs are usually in the parent of checkpoints dir, i.e. .../
    # Let's try to find them.
    
    possible_config_dir = os.path.dirname(checkpoint_dir)
    
    # Override config paths if they exist in checkpoint dir
    if os.path.exists(os.path.join(possible_config_dir, "config_gcpnet_encoder.yaml")):
        encoder_config_path = os.path.join(possible_config_dir, "config_gcpnet_encoder.yaml")
        print(f"Using encoder config from: {encoder_config_path}")
    else:
        encoder_config_path = "/data/ORPHEUS/Geometric_Teacher/configs/config_gcpnet_encoder.yaml"
        print(f"Using default encoder config: {encoder_config_path}")

    if os.path.exists(os.path.join(possible_config_dir, "config_geometric_decoder.yaml")):
        decoder_config_path = os.path.join(possible_config_dir, "config_geometric_decoder.yaml")
        print(f"Using decoder config from: {decoder_config_path}")
    else:
        decoder_config_path = "/data/ORPHEUS/Geometric_Teacher/configs/config_geometric_decoder.yaml"
        print(f"Using default decoder config: {decoder_config_path}")
    
    # Also need to load the main VQVAE config from checkpoint dir if possible, 
    # because it contains VQVAE specific params like dim, heads etc.
    if os.path.exists(os.path.join(possible_config_dir, "config_vqvae.yaml")):
        config_path = os.path.join(possible_config_dir, "config_vqvae.yaml")
        print(f"Using VQVAE config from: {config_path}")

    with open(config_path) as f:
        config = Box(yaml.full_load(f))
    
    with open(decoder_config_path) as f:
        decoder_configs = Box(yaml.full_load(f))
        
    # Mock logger
    class Logger:
        def info(self, msg): pass
    logger = Logger()
    
    # Initialize SuperModel
    model = prepare_model_custom(config, logger, encoder_config_path, decoder_configs)
    
    # Load Checkpoint
    # Note: We only strictly need the VQVAE/Decoder part
    model = load_checkpoints_simple(checkpoint_path, model, logger)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt", type=str, default="checkpoints/best_student.pth")
    parser.add_argument("--teacher_ckpt", type=str, default="/data/ORPHEUS/Geometric_Teacher/checkpoints/2025-07-19__20-09-19/checkpoints/best_valid.pth")
    parser.add_argument("--teacher_config", type=str, default="/data/ORPHEUS/Geometric_Teacher/configs/config_vqvae.yaml")
    parser.add_argument("--data_path", type=str, default="data/geometry_labels.jsonl")
    parser.add_argument("--esm_path", type=str, default="pretrained/esm2_t12_35M_UR50D")
    parser.add_argument("--output_dir", type=str, default="inference_results")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--pdbbind_root", type=str, default="/data/ORPHEUS/Geometric_Teacher/data/PDBBind")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Student
    print("Loading Student Model...")
    student = StudentModel(esm_model_path=args.esm_path)
    student.load_state_dict(torch.load(args.student_ckpt, map_location=args.device))
    student.to(args.device)
    student.eval()
    
    tokenizer = EsmTokenizer.from_pretrained(args.esm_path)
    
    # 2. Load Teacher (Decoder)
    print("Loading Teacher Decoder...")
    teacher = load_teacher_decoder(args.teacher_config, args.teacher_ckpt, args.device)
    
    # 3. Load Data Samples
    print("Loading Data...")
    samples = []
    with open(args.data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.num_samples: break
            samples.append(json.loads(line))
            
    # 4. Inference
    print(f"Running inference on {len(samples)} samples...")
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    for idx, sample in enumerate(samples):
        pdb_id = sample['pdb_id']
        seq = sample['seq']
        smiles = sample['smiles']
        
        print(f"Processing {pdb_id}...")
        
        # --- Student Prediction ---
        # Prepare Input
        encoding = tokenizer(seq, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')
        input_ids = encoding['input_ids'].to(args.device)
        attention_mask = encoding['attention_mask'].to(args.device)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = mfgen.GetFingerprintAsNumPy(mol)
            ligand_feat = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(args.device)
        else:
            print(f"Skipping {pdb_id}: Invalid SMILES")
            continue
            
        with torch.no_grad():
            logits = student(input_ids, attention_mask, ligand_feat) # [1, 1024, 4096]
            predicted_tokens = torch.argmax(logits, dim=-1) # [1, 1024]
            
        # Extract valid tokens (remove padding)
        # ESM adds <cls> at 0, so seq starts at 1.
        # We need to map back to original length.
        seq_len = len(seq)
        # predicted_tokens[0, 1 : seq_len+1] corresponds to the sequence
        valid_tokens = predicted_tokens[0, 1 : seq_len+1]
        
        # --- Teacher Decoding ---
        # We need to construct a batch for the VQVAE decoder
        # The decoder expects quantized embeddings or indices.
        # We have indices.
        
        # Teacher's decode method usually takes quantized vectors, 
        # but we can use vqvae.quantizer.get_codebook_entry to get vectors from indices
        
        indices = valid_tokens.unsqueeze(0) # [1, L]
        
        # Get embeddings from codebook
        # vqvae.quantizer.embedding(indices) -> [1, L, D]
        # Use the method from VQVAETransformer's vector_quantizer
        z_q = teacher.vqvae.vector_quantizer.get_output_from_indices(indices)
        
        # Permute for decoder if needed? 
        # VQVAE usually expects [B, D, L] for Conv1d, or [B, L, D] for Transformer?
        # Let's check VQVAE forward.
        # It seems VQVAETransformer uses [B, L, D].
        
        # Run Decoder
        # We need to create a dummy batch object because the decoder might expect it for masks etc.
        # Or we can call teacher.vqvae.decoder directly?
        
        # Let's try calling teacher.vqvae.decoder(z_q, ...)
        # We need masks.
        mask = torch.ones((1, seq_len), dtype=torch.bool, device=args.device)
        nan_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=args.device)
        
        try:
            # Decode
            # The decoder returns: (coord_pred, ...)
            # We need to check the signature of teacher.vqvae.decoder
            # It's a GeometricDecoder.
            
            # Actually, let's look at SuperModel.forward
            # It calls self.vqvae(x, ...)
            # Inside vqvae:
            # quantized, indices, ... = self.quantizer(z)
            # output = self.decoder(quantized, mask, nan_mask)
            
            decoded_outputs = teacher.vqvae.decoder(z_q, mask)
            
            # decoded_outputs is usually a tuple (coords, ...) or just coords depending on config
            # GeometricDecoder returns (x_out, ...)
            
            coords_pred = decoded_outputs[0] # [1, L, 9] usually (N, CA, C) flattened
            
            # Reshape to [1, L, 3, 3]
            coords_pred = coords_pred.view(1, seq_len, 3, 3)
            
            # Save to PDB
            # We can use graphein or simple PDB writer
            save_path = os.path.join(args.output_dir, f"{pdb_id}_predicted.pdb")
            
            # Try to find ligand file to append
            ligand_path = None
            # Search pattern: .../P-L/*/<pdb_id>/<pdb_id>_ligand.sdf or .mol2
            import glob
            search_pattern = os.path.join(args.pdbbind_root, "P-L", "*", pdb_id, f"{pdb_id}_ligand.sdf")
            found = glob.glob(search_pattern)
            if not found:
                search_pattern = os.path.join(args.pdbbind_root, "P-L", "*", pdb_id, f"{pdb_id}_ligand.mol2")
                found = glob.glob(search_pattern)
            
            if found:
                ligand_path = found[0]
                print(f"Found ligand file: {ligand_path}")
            else:
                print(f"Warning: Ligand file not found for {pdb_id}")

            save_pdb(coords_pred[0], seq, save_path, ligand_path)
            print(f"Saved structure to {save_path}")
            
            # Add Sidechains
            try:
                from add_sidechains import add_sidechains
                full_atom_path = save_path.replace(".pdb", "_full.pdb")
                add_sidechains(save_path, full_atom_path)
            except ImportError:
                print("Could not import add_sidechains. Please ensure pdbfixer and openmm are installed.")
            except Exception as e:
                print(f"Error adding sidechains: {e}")
            
        except Exception as e:
            print(f"Decoding failed for {pdb_id}: {e}")
            import traceback
            traceback.print_exc()

def save_pdb(coords, seq, path, ligand_path=None):
    """
    Simple PDB writer for backbone coordinates.
    coords: [L, 3] (CA only) or [L, 4, 3] (N, CA, C, O)
    ligand_path: Optional path to ligand file (SDF/MOL2) to append.
    """
    # If coords is [L, 4, 3], we can write full backbone.
    # If [L, 3], just CA.
    
    # Let's assume [L, 4, 3] for now as GeometricDecoder usually outputs that.
    # Or maybe [L, 3] if it's CA-only model.
    
    # Check shape
    if coords.shape[-2] == 3: # x, y, z
        if coords.ndim == 2: # [L, 3] -> CA only
            atom_names = ["CA"]
            coords = coords.unsqueeze(1)
        elif coords.ndim == 3: # [L, Atoms, 3]
            if coords.shape[1] == 3:
                atom_names = ["N", "CA", "C"]
            elif coords.shape[1] == 4:
                atom_names = ["N", "CA", "C", "O"]
            else:
                atom_names = [f"A{i}" for i in range(coords.shape[1])]
    
    # Define mapping manually since Biopython import is failing
    aa_1_3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    
    def one_to_three(res):
        return aa_1_3.get(res, "UNK")

    with open(path, 'w') as f:
        atom_idx = 1
        # print(f"DEBUG: Saving PDB {path}, seq length: {len(seq)}, first 5 residues: {seq[:5]}")
        for res_i, res_name in enumerate(seq):
            # 3-letter code
            res_name_3 = one_to_three(res_name)
                
            for atom_i, atom_name in enumerate(atom_names):
                pos = coords[res_i, atom_i].detach().cpu().numpy()
                # PDB format
                # ATOM      1  N   MET A   1      27.340  24.430  26.180  1.00 20.00           N
                f.write(f"ATOM  {atom_idx:5d}  {atom_name:<3} {res_name_3} A{res_i+1:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {atom_name[0]}\n")
                atom_idx += 1
        
        # Append Ligand if provided
        if ligand_path:
            try:
                if ligand_path.endswith(".sdf"):
                    suppl = Chem.SDMolSupplier(ligand_path)
                    mol = suppl[0]
                elif ligand_path.endswith(".mol2"):
                    mol = Chem.MolFromMol2File(ligand_path)
                else:
                    mol = None
                
                if mol:
                    conf = mol.GetConformer()
                    for i, atom in enumerate(mol.GetAtoms()):
                        pos = conf.GetAtomPosition(i)
                        symbol = atom.GetSymbol()
                        # Ensure unique atom names by appending index: C1, C2, etc.
                        atom_name = f"{symbol}{i+1}"
                        if len(atom_name) > 4: atom_name = atom_name[:4]
                        
                        # HETATM 1234  C1   LIG B   1      10.000  20.000  30.000  1.00  0.00           C
                        f.write(f"HETATM{atom_idx:5d}  {atom_name:<4} LIG B   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00           {symbol}\n")
                        atom_idx += 1
                    print(f"Appended ligand from {ligand_path}")
                else:
                    print(f"Failed to load ligand from {ligand_path}")
            except Exception as e:
                print(f"Error appending ligand: {e}")
                
if __name__ == "__main__":
    main()
