"""
CONTEXT:
We are building the 'Geometry Teacher' data pipeline for the ORPHEUS project using PDBBind.
We have a pre-trained GCP-VQVAE model loaded as `model`.
The goal is to generate training labels for a Student model.

TASK:
Write a Python script that processes PDBBind complexes to extract "Pocket Tokens".

LOGIC STEPS:
1. Iterate through PDBBind folders. For each complex:
2. Load protein PDB (using Biopython) and Ligand SDF (using RDKit).
3. Extract Protein Backbone coords (N, Ca, C) and Ligand coordinates.
4. COMPUTE MASK: Identify residues where C-alpha is within 6.0 Angstroms of any Ligand atom.
5. RUN VQVAE: Pass the *entire* protein backbone coords to `model.encode` to get the discrete codebook indices (tokens).
6. FILTER: Extract only the tokens corresponding to the mask (the pocket residues).
7. SAVE: Save a dictionary containing: sequence (str), smiles (str), and pocket_token_labels (dict: residue_idx -> token_id).

REQUIREMENTS:
- Use `scipy.spatial.distance.cdist` for efficient distance calculation.
- Handle cases where PDB parsing might fail.
- Assume `model` takes input shape [Batch, Length, 3, 3].
"""

import os
import argparse
import torch
import numpy as np
import math
import json
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Suppress RDKit warnings
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, Batch
# from torch_cluster import knn_graph
from graphein.protein.resi_atoms import PROTEIN_ATOMS, STANDARD_AMINO_ACIDS

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Import project modules
from utils.utils import load_configs, load_checkpoints_simple, print_trainable_parameters
from models.super_model import SuperModel
from models.decoders import GeometricDecoder
from models.vqvae import VQVAETransformer
from models.gcpnet.models.base import instantiate_encoder, load_pretrained_encoder, PretrainedEncoder

# --- Custom KNN Implementation ---
def custom_knn_graph(x, k, batch=None, loop=False):
    """
    Simple KNN implementation using torch.cdist to avoid torch_cluster dependency.
    Assumes x is [N, 3].
    Returns edge_index [2, E] where row 0 is source (neighbor), row 1 is target (query).
    """
    # Compute pairwise distances
    # dist: [N, N]
    dist = torch.cdist(x, x)
    
    if not loop:
        # Set diagonal to infinity to avoid self-loops
        dist.fill_diagonal_(float('inf'))
        
    # Get top k indices (smallest distances)
    # values, indices: [N, k]
    # indices[i] contains the k neighbors of node i
    _, indices = dist.topk(k, dim=1, largest=False)
    
    num_nodes = x.size(0)
    
    # Construct edge_index
    # We want edges pointing FROM neighbors TO the query node.
    # edge_index[0] = neighbors (source)
    # edge_index[1] = query nodes (target)
    
    target = torch.arange(num_nodes, device=x.device).view(-1, 1).repeat(1, k).view(-1)
    source = indices.view(-1)
    
    return torch.stack([source, target], dim=0)

# --- Custom Model Loading ---

def prepare_model_custom(configs, logger, encoder_config_path, decoder_configs, **kwargs):
    log_details = configs.model.get('log_details', True)

    if not kwargs.get("decoder_only", False):
        if configs.model.encoder.name == "gcpnet":
            if configs.model.encoder.pretrained.enabled:
                encoder = load_pretrained_encoder(
                    encoder_config_path,
                    checkpoint_path=configs.model.encoder.pretrained.checkpoint_path,
                )
            else:
                components, _ = instantiate_encoder(encoder_config_path)
                encoder = PretrainedEncoder(components)
        else:
            raise ValueError("Invalid encoder model specified!")

        if configs.model.encoder.get('freeze_parameters', False):
            for param in encoder.parameters():
                param.requires_grad = False
            if log_details:
                logger.info("Encoder parameters frozen.")

    else:
        encoder = None

    if configs.model.vqvae.decoder.name == "geometric_decoder":
        decoder = GeometricDecoder(configs, decoder_configs=decoder_configs)
    else:
        raise ValueError("Invalid decoder model specified!")

    if configs.model.vqvae.decoder.get('freeze_parameters', False):
        for param in decoder.parameters():
            param.requires_grad = False
        if log_details:
            logger.info("Decoder parameters frozen.")

    vqvae = VQVAETransformer(
        decoder=decoder,
        configs=configs,
        logger=logger,
        decoder_only=kwargs.get("decoder_only", False),
    )

    vqvae = SuperModel(encoder, vqvae, configs, decoder_only=kwargs.get("decoder_only", False))

    return vqvae

# --- Helper Functions ---

def custom_knn_graph(x, k, batch=None, loop=False):
    # Fallback KNN implementation using pure PyTorch if torch_cluster fails
    # x: [N, 3]
    # batch: [N]
    
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
    edge_index_list = []
    
    num_graphs = batch.max().item() + 1
    for i in range(num_graphs):
        mask = batch == i
        x_i = x[mask] # [N_i, 3]
        
        # Compute pairwise distance
        dist = torch.cdist(x_i, x_i) # [N_i, N_i]
        
        # Get top k
        # We want smallest distances. 
        # If loop=False, we should ignore diagonal (dist 0). 
        # But topk returns smallest.
        
        if not loop:
            dist.fill_diagonal_(float('inf'))
            
        # k might be larger than N_i
        k_i = min(k, x_i.size(0) - (1 if not loop else 0))
        
        if k_i <= 0:
            continue
            
        _, indices = dist.topk(k_i, dim=1, largest=False) # [N_i, k_i]
        
        # Create edge index
        # source: indices (neighbors)
        # target: arange (center)
        
        target = torch.arange(x_i.size(0), device=x.device).unsqueeze(1).expand(-1, k_i).reshape(-1)
        source = indices.reshape(-1)
        
        # Adjust indices to global batch
        # Find start index of this graph
        start_idx = torch.where(mask)[0][0]
        
        edge_index_list.append(torch.stack([source + start_idx, target + start_idx], dim=0))
        
    if not edge_index_list:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
        
    return torch.cat(edge_index_list, dim=1)

def get_backbone_coords_and_seq(pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', pdb_path)
    except Exception as e:
        print(f"Failed to parse PDB {pdb_path}: {e}")
        return None, None

    coords = []
    seq = []
    
    # Assume single chain or take the first one
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() not in THREE_TO_ONE:
                    # print(f"Skipping non-standard residue: {residue.get_resname()}")
                    continue
                
                try:
                    n = residue['N'].get_vector()
                    ca = residue['CA'].get_vector()
                    c = residue['C'].get_vector()
                    o = residue['O'].get_vector() 
                    
                    coords.append([
                        [n[0], n[1], n[2]],
                        [ca[0], ca[1], ca[2]],
                        [c[0], c[1], c[2]],
                        [o[0], o[1], o[2]]
                    ])
                    seq.append(THREE_TO_ONE[residue.get_resname()])
                except KeyError as e:
                    # print(f"Missing atom in residue {residue.get_resname()} {residue.id}: {e}")
                    continue
            
            # If we found a chain with valid residues, stop. 
            # If this chain was empty (e.g. water), we should continue to next chain?
            # The original code broke after the first chain regardless.
            if coords:
                break 
        if coords:
            break
        
    if not coords:
        print(f"No valid backbone coords found in {pdb_path}")
        return None, None
        
    return torch.tensor(coords, dtype=torch.float32), "".join(seq)

def prepare_graph_batch(coords, seq, device):
    # coords: [L, 4, 3]
    L = len(seq)
    
    # Create Data object
    # We need to provide attributes that ProteinFeaturiser expects
    
    # Residue types (indices)
    residue_type = torch.tensor([STANDARD_AMINO_ACIDS.index(res) for res in seq], dtype=torch.long)
    
    # Sequence position
    seq_pos = torch.arange(L, dtype=torch.long)
    
    # Coords for ProteinFeaturiser (needs to be [L, 37, 3] usually, but we only have backbone)
    # We'll pad it to match PROTEIN_ATOMS length if needed, or just provide what we have if featuriser handles it.
    # dataset.py sets coords to [L, 37, 3] with NaNs.
    full_coords = torch.full((L, len(PROTEIN_ATOMS), 3), float('nan'), dtype=torch.float32)
    full_coords[:, :4, :] = coords # N, CA, C, O are first 4 in PROTEIN_ATOMS? 
    # Let's check PROTEIN_ATOMS order. Usually N, CA, C, O.
    # If not, we might need to map. But dataset.py assumes coords[:, :3] is N, CA, C.
    
    data = Data(
        x=torch.zeros(L, 1), # Placeholder, will be overwritten
        seq=seq,
        residue_type=residue_type,
        seq_pos=seq_pos,
        coords=full_coords,
        pos=coords[:, 1], # Add pos (CA coords)
        mask=torch.ones(L, dtype=torch.bool),
        x_bb=coords[:, :3] # Add x_bb explicitly as it's used in dataset.py logic sometimes
    )
    
    # Create Batch
    batch = Batch.from_data_list([data])
    batch = batch.to(device)
    
    # Compute KNN edges (k=16 as per config)
    # We use CA coordinates for KNN
    ca_coords = coords[:, 1].to(device)
    # edge_index = knn_graph(ca_coords, k=16, batch=batch.batch, loop=False)
    edge_index = custom_knn_graph(ca_coords, k=16, batch=batch.batch, loop=False)
    batch.edge_index = edge_index
    
    # Add extra attributes required by SuperModel
    batch_dict = {
        'graph': batch,
        'nan_masks': torch.ones((1, L), dtype=torch.bool).to(device),
        'masks': torch.ones((1, L), dtype=torch.bool).to(device),
        'indices': None
    }
    
    return batch_dict


def get_pocket_tokens(pdb_path, ligand_path, model, device, threshold=6.0):
    # 1. Parse Structures
    coords, seq = get_backbone_coords_and_seq(pdb_path)
    if coords is None:
        return None
    
    # 2. Parse Ligand
    try:
        suppl = Chem.SDMolSupplier(ligand_path)
        ligand = suppl[0]
        if ligand is None:
            print(f"Ligand is None for {ligand_path}")
            return None
        lig_coords = ligand.GetConformer().GetPositions()
        smiles = Chem.MolToSmiles(ligand)
    except Exception as e:
        print(f"Failed to parse ligand {ligand_path}: {e}")
        return None
    
    # 3. Compute Mask (Distance Calculation)
    ca_coords = coords[:, 1].numpy()
    dists = cdist(ca_coords, lig_coords)
    min_dists = np.min(dists, axis=1)
    is_pocket = min_dists < threshold
    
    if not np.any(is_pocket):
        print(f"No pocket residues found for {pdb_path} (min dist: {np.min(min_dists):.2f})")
        return None

    # 4. VQVAE Inference
    try:
        batch_dict = prepare_graph_batch(coords, seq, device)
        
        # Pad masks to model.max_length
        if hasattr(model, 'max_length'):
            max_len = model.max_length
            curr_len = batch_dict['masks'].shape[1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                batch_dict['masks'] = F.pad(batch_dict['masks'], (0, pad_len), value=False)
                batch_dict['nan_masks'] = F.pad(batch_dict['nan_masks'], (0, pad_len), value=False)
        
        with torch.no_grad():
            # SuperModel forward returns a dict
            output = model(batch_dict)
            indices = output['indices'] # [1, L]
            
        tokens = indices[0].cpu().numpy()
    except Exception as e:
        print(f"Inference failed for {pdb_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 5. Filter Labels
    pocket_labels = {}
    for idx, is_p in enumerate(is_pocket):
        if is_p:
            pocket_labels[idx] = int(tokens[idx])
            
    return {
        "pdb_id": os.path.basename(os.path.dirname(pdb_path)), # Assuming folder name is PDB ID
        "seq": seq,
        "smiles": smiles,
        "pocket_labels": pocket_labels
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Geometry Labels for Student Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to PDBBind dataset root")
    parser.add_argument("--config_path", type=str, default="configs/config_vqvae.yaml", help="Path to VQVAE config")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VQVAE checkpoint")
    parser.add_argument("--output_path", type=str, default="geometry_labels.jsonl", help="Output file path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Determine config paths
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    # If checkpoint is in a 'checkpoints' subdir, go up one level to find configs if they were saved with the run
    # But based on unzip, they are in the same dir as 'checkpoints' folder?
    # Structure: checkpoints/2025.../config_vqvae.yaml AND checkpoints/2025.../checkpoints/best_valid.pth
    # So if args.checkpoint_path is .../checkpoints/best_valid.pth, then dirname is .../checkpoints
    # and configs are in .../ (parent of dirname)
    
    possible_config_dir = os.path.dirname(checkpoint_dir)
    
    vqvae_config_path = args.config_path
    if os.path.exists(os.path.join(possible_config_dir, "config_vqvae.yaml")):
        vqvae_config_path = os.path.join(possible_config_dir, "config_vqvae.yaml")
        print(f"Using config from checkpoint directory: {vqvae_config_path}")
        
    encoder_config_path = "configs/config_gcpnet_encoder.yaml"
    if os.path.exists(os.path.join(possible_config_dir, "config_gcpnet_encoder.yaml")):
        encoder_config_path = os.path.join(possible_config_dir, "config_gcpnet_encoder.yaml")
        print(f"Using encoder config from checkpoint directory: {encoder_config_path}")

    decoder_config_path = "configs/config_geometric_decoder.yaml"
    if os.path.exists(os.path.join(possible_config_dir, "config_geometric_decoder.yaml")):
        decoder_config_path = os.path.join(possible_config_dir, "config_geometric_decoder.yaml")
        print(f"Using decoder config from checkpoint directory: {decoder_config_path}")

    # Load Configs
    with open(vqvae_config_path) as f:
        import yaml
        from box import Box
        config = Box(yaml.full_load(f))
        
    with open(decoder_config_path) as f:
        decoder_configs = Box(yaml.full_load(f))
        
    # Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    # Mock logger
    class Logger:
        def info(self, msg): print(msg)
    logger = Logger()
    
    model = prepare_model_custom(config, logger, encoder_config_path, decoder_configs)
    model = load_checkpoints_simple(args.checkpoint_path, model, logger)
    model.to(args.device)
    model.eval()
    
    # Iterate Data
    results = []
    
    print(f"Scanning {args.data_dir} for PDB complexes...")
    pdb_dirs = []
    for root, dirs, files in os.walk(args.data_dir):
        pdb_id = os.path.basename(root)
        # Check for protein file existence to confirm it's a data directory
        if os.path.exists(os.path.join(root, f"{pdb_id}_protein.pdb")):
            pdb_dirs.append(root)
    
    print(f"Found {len(pdb_dirs)} complexes.")
    
    count = 0
    print(f"Processing and saving to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        for pdb_dir in tqdm(pdb_dirs):
            pdb_id = os.path.basename(pdb_dir)
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
            ligand_path = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
            
            if not os.path.exists(pdb_path) or not os.path.exists(ligand_path):
                # Try .mol2 if .sdf missing? Or just skip.
                continue
                
            result = get_pocket_tokens(pdb_path, ligand_path, model, args.device)
            if result:
                f.write(json.dumps(result) + "\n")
                f.flush()
                count += 1
            
    print(f"Done. Saved {count} samples to {args.output_path}.")

if __name__ == "__main__":
    main()
