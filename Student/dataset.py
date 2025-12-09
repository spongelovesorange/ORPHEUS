import json
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from transformers import EsmTokenizer

class GeometryStudentDataset(Dataset):
    def __init__(self, jsonl_path, esm_model_path, max_len=1024, ligand_dim=2048):
        """
        Args:
            jsonl_path: Path to the geometry_labels.jsonl file.
            esm_model_path: Path to the ESM-2 model (for tokenizer).
            max_len: Maximum sequence length for padding/truncation.
            ligand_dim: Dimension of Morgan fingerprint.
        """
        self.data = []
        print(f"Loading data from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    continue
        print(f"Loaded {len(self.data)} samples.")
        
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_path)
        self.max_len = max_len
        self.ligand_dim = ligand_dim
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['seq']
        smiles = item['smiles']
        pocket_labels = item['pocket_labels'] # Dict {str(idx): token_id}
        
        # 1. Process Protein Sequence
        # ESM tokenizer adds <cls> at start and <eos> at end automatically if we use encode_plus?
        # Actually, standard ESM tokenizer usage:
        # <cls> M A L ... <eos>
        # So index i in original seq maps to i+1 in tokenized seq.
        
        encoding = self.tokenizer(
            seq, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0) # [Max_Len]
        attention_mask = encoding['attention_mask'].squeeze(0) # [Max_Len]
        
        # 2. Process Ligand (Morgan Fingerprint)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Fallback for invalid SMILES (should be rare if data is clean)
            fp = np.zeros((self.ligand_dim,), dtype=np.float32)
        else:
            # Use new MorganGenerator if available (RDKit >= 2023.09)
            try:
                from rdkit.Chem import rdFingerprintGenerator
                mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.ligand_dim)
                fp = mfgen.GetFingerprintAsNumPy(mol)
            except ImportError:
                # Fallback for older RDKit
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.ligand_dim)
                fp = np.array(fp, dtype=np.float32)
            
        ligand_feat = torch.tensor(fp, dtype=torch.float32)
        
        # 3. Process Labels
        # Initialize with -100 (PyTorch CrossEntropyLoss ignore_index default)
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        
        # Map pocket labels to padded sequence
        # Note: ESM adds <cls> at index 0. So original residue i is at i+1.
        # We need to be careful about truncation.
        
        seq_len = len(seq)
        effective_len = min(seq_len, self.max_len - 2) # Reserve for cls and eos
        
        for res_idx_str, token_id in pocket_labels.items():
            res_idx = int(res_idx_str)
            
            # Check if this residue is within our truncated sequence
            if res_idx < effective_len:
                # Map to ESM token position (res_idx + 1 because of <cls>)
                target_pos = res_idx + 1
                labels[target_pos] = token_id
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ligand_feat': ligand_feat,
            'labels': labels
        }
