# ORPHEUS Physics Teacher Module

This module contains the tools and scripts developed to extract "physical insights" from the Matcha docking model, serving as the "Physics Teacher" for the ORPHEUS knowledge distillation project.

## Structure

### 1. Label Generation (`/`)
- **`generate_labels.py`**: The core script. It processes the full 33-frame inference trajectory from Matcha, removes rigid body motion, and extracts two key physical features:
    - **Internal Deformation**: Magnitude of non-rigid atomic movement (identifies active atoms like Oxygen in Biotin).
    - **Directionality**: Cosine similarity of atomic movement relative to the pocket center (identifies attraction/repulsion).
    - **Output**: `.pt` files containing `(T, N)` tensors for training the Student model.

### 2. Visualization (`/visualization`)
- **`visualize_deformation.py`**: Generates PDBs colored by internal deformation magnitude (Red = High deformation).
- **`visualize_direction.py`**: Generates PDBs colored by movement direction (Red = Inward/Attraction, Blue = Outward/Repulsion).
- **`visualize_flow.py`**: (Experimental) Visualizes raw flow vectors.
- **`save_full_trajectory.py`**: Stitches Stage 1, 2, and 3 trajectories into a single PDB animation for qualitative analysis.

### 3. Analysis (`/analysis`)
- **`verify_movement.py`**: Quantitative verification script. Prints the top atoms with the highest accumulated deformation (confirmed Oxygen atoms as top rank).
- **`inspect_npy.py`**: Helper to inspect raw Matcha output files.
- **`check_centers.py`**: Helper to check pocket centers.

## Usage

To generate physics labels for a dataset:
```bash
python orpheus_physics/generate_labels.py
```

To visualize the "Internal Deformation" of a specific case:
```bash
python orpheus_physics/visualization/visualize_deformation.py
```
