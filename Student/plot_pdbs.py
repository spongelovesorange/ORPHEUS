import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def parse_pdb_ca(path):
    coords = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                # PDB format: x is 30-38, y is 38-46, z is 46-54
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords)

def plot_structure(coords, pdb_id, output_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(coords) == 0:
        print(f"Warning: No CA atoms found in {pdb_id}")
        return

    # Plot backbone trace
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], '-o', markersize=3, label='Predicted Backbone (CA)')
    
    # Start and End
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color='green', s=100, label='Start (N-term)')
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], color='red', s=100, label='End (C-term)')
    
    ax.set_title(f"Predicted Structure: {pdb_id}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    results_dir = "inference_results"
    pdb_files = glob.glob(os.path.join(results_dir, "*_predicted.pdb"))
    
    if not pdb_files:
        print(f"No PDB files found in {results_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files. Generating plots...")
    
    for pdb_file in pdb_files:
        pdb_id = os.path.basename(pdb_file).replace("_predicted.pdb", "")
        coords = parse_pdb_ca(pdb_file)
        
        output_path = os.path.join(results_dir, f"{pdb_id}_plot.png")
        plot_structure(coords, pdb_id, output_path)

if __name__ == "__main__":
    main()
