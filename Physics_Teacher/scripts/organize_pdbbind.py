import os
import glob

def organize_dataset():
    source_root = "/data/Matcha-main/orpheus_physics/data/pdbbind/P-L"
    target_root = "/data/Matcha-main/orpheus_physics/data/my_dataset"
    
    os.makedirs(target_root, exist_ok=True)
    
    # Find all PDB folders
    # Structure: source_root / year_range / pdb_id
    pdb_folders = glob.glob(os.path.join(source_root, "*", "*"))
    
    print(f"Found {len(pdb_folders)} samples in {source_root}")
    
    count = 0
    pdb_ids = []
    
    for folder in pdb_folders:
        if not os.path.isdir(folder):
            continue
            
        pdb_id = os.path.basename(folder)
        
        # Create symlink
        target_path = os.path.join(target_root, pdb_id)
        if not os.path.exists(target_path):
            os.symlink(folder, target_path)
            
        pdb_ids.append(pdb_id)
        count += 1
        
    print(f"Symlinked {count} folders to {target_root}")
    
    # Save list
    list_path = "/data/Matcha-main/orpheus_physics/data/pdbbind_full.txt"
    with open(list_path, "w") as f:
        for pid in pdb_ids:
            f.write(pid + "\n")
            
    print(f"Saved full list to {list_path}")

if __name__ == "__main__":
    organize_dataset()
