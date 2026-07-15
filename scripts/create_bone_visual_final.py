#!/usr/bin/env python3
"""
Create PLY files for visual confirmation of bone fragment assembly results.
"""

import json
import numpy as np
import h5py
import trimesh
import os

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def apply_transform(vertices, translation, quaternion):
    """Apply rigid transformation to vertices."""
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    return (rotation_matrix @ vertices.T).T + translation

def create_bone_visualization(result_file, hdf5_path, output_prefix):
    """Create visualization for bone fragment assembly."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    sample_name = data['name']  # e.g., "pig_bone/pig_bone_20"
    sample_id = sample_name.split('/')[-1]  # "pig_bone_20"
    pieces = data['pieces'].split(',')  # e.g., ["pig_bone_20_2", "pig_bone_20_1"]
    gt_trans_rots = np.array(data['gt_trans_rots'])
    pred_trans_rots = np.array(data['pred_trans_rots'])
    part_acc = data['part_acc']
    
    print(f"\nProcessing: {sample_name}")
    print(f"Sample ID: {sample_id}")
    print(f"Pieces: {pieces}")
    print(f"Part accuracy: {part_acc}")
    print(f"Translation RMSE: {data['rmse_t']:.4f}m")
    print(f"Rotation RMSE: {data['rmse_r']:.2f}°")
    
    # Load meshes from HDF5
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Navigate to the sample
            sample_group = f['pig_bone'][sample_id]
            pieces_group = sample_group['pieces']
            pieces_names = [name.decode() for name in sample_group['pieces_names']]
            
            print(f"Available pieces: {pieces_names}")
            
            # Load fragment meshes
            fragments = []
            piece_indices = []
            
            # Map piece names to indices
            for piece in pieces:
                if piece in pieces_names:
                    piece_idx = pieces_names.index(piece)
                    piece_indices.append(piece_idx)
                else:
                    print(f"Warning: {piece} not found in pieces_names")
                    return
            
            # Load meshes
            for i, piece_idx in enumerate(piece_indices):
                piece_group = pieces_group[str(piece_idx)]
                vertices = np.array(piece_group['vertices'])
                faces = np.array(piece_group['faces'])
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                fragments.append(mesh)
                print(f"Loaded piece {piece_idx} ({pieces[i]}): {vertices.shape[0]} vertices, {faces.shape[0]} faces")
            
            if len(fragments) != len(pieces):
                print("Mismatch in number of fragments, skipping...")
                return
            
            # Create initial scattered state
            scattered_meshes = []
            for i, mesh in enumerate(fragments):
                mesh_copy = mesh.copy()
                # Color code: red for first fragment, green for second
                mesh_copy.visual.face_colors = [255, 100, 100, 200] if i == 0 else [100, 255, 100, 200]
                scattered_meshes.append(mesh_copy)
            
            # Save scattered fragments
            if scattered_meshes:
                scattered_combined = trimesh.util.concatenate(scattered_meshes)
                scattered_filename = f"{output_prefix}_INITIAL_scattered.ply"
                scattered_combined.export(scattered_filename)
                print(f"Saved scattered: {scattered_filename}")
            
            # Create ground truth assembly
            gt_meshes = []
            for i, (mesh, transform) in enumerate(zip(fragments, gt_trans_rots)):
                mesh_copy = mesh.copy()
                translation = transform[:3]
                quaternion = transform[3:]  # [w, x, y, z]
                
                mesh_copy.vertices = apply_transform(mesh_copy.vertices, translation, quaternion)
                mesh_copy.visual.face_colors = [255, 100, 100, 200] if i == 0 else [100, 255, 100, 200]
                gt_meshes.append(mesh_copy)
            
            # Save ground truth
            if gt_meshes:
                gt_combined = trimesh.util.concatenate(gt_meshes)
                gt_filename = f"{output_prefix}_GROUND_TRUTH.ply"
                gt_combined.export(gt_filename)
                print(f"Saved ground truth: {gt_filename}")
            
            # Create final prediction assembly
            final_pred = pred_trans_rots[-1]  # Use final prediction
            pred_meshes = []
            for i, (mesh, transform) in enumerate(zip(fragments, final_pred)):
                mesh_copy = mesh.copy()
                translation = transform[:3]
                quaternion = transform[3:]
                
                mesh_copy.vertices = apply_transform(mesh_copy.vertices, translation, quaternion)
                mesh_copy.visual.face_colors = [255, 100, 100, 200] if i == 0 else [100, 255, 100, 200]
                pred_meshes.append(mesh_copy)
            
            # Save final prediction
            if pred_meshes:
                pred_combined = trimesh.util.concatenate(pred_meshes)
                pred_filename = f"{output_prefix}_FINAL_assembled.ply"
                pred_combined.export(pred_filename)
                print(f"Saved final assembly: {pred_filename}")
            
            # Create comparison visualization (GT + Prediction side by side)
            if gt_meshes and pred_meshes:
                # Create a copy of prediction meshes for comparison
                pred_meshes_copy = []
                offset = np.array([0.3, 0, 0])  # 30cm offset on X axis
                
                for i, mesh in enumerate(pred_meshes):
                    mesh_copy = mesh.copy()
                    mesh_copy.vertices += offset
                    # Change colors for prediction (blue tones)
                    mesh_copy.visual.face_colors = [100, 100, 255, 200] if i == 0 else [100, 255, 255, 200]
                    pred_meshes_copy.append(mesh_copy)
                
                comparison_meshes = gt_meshes + pred_meshes_copy
                comparison_combined = trimesh.util.concatenate(comparison_meshes)
                comparison_filename = f"{output_prefix}_COMPARISON_all.ply"
                comparison_combined.export(comparison_filename)
                print(f"Saved comparison: {comparison_filename}")
                print(f"  Left (red/green): Ground Truth")
                print(f"  Right (blue/cyan): Prediction")
    
    except Exception as e:
        print(f"Error processing {sample_name}: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    results_dir = "logs/GARF/real_bone_fractures/version_0/json_results"
    hdf5_path = "input/Fractura/bone_real.hdf5"
    
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found at {hdf5_path}")
        exit(1)
    
    # Process best case (pig_bone_20 - 100% accuracy)
    print("=" * 60)
    print("BEST CASE: pig_bone_20 (100% part accuracy)")
    print("=" * 60)
    create_bone_visualization(
        f"{results_dir}/0.json",
        hdf5_path,
        "bone_best"
    )
    
    # Process challenging case (pig_bone_27 - 50% accuracy)
    print("\n" + "=" * 60)
    print("CHALLENGING CASE: pig_bone_27 (50% part accuracy)")
    print("=" * 60)
    create_bone_visualization(
        f"{results_dir}/18.json",
        hdf5_path,
        "bone_challenging"
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nGenerated PLY files for download:")
    os.system("ls -la bone_*.ply")
    
    print("\nFile descriptions:")
    print("- *_INITIAL_scattered.ply: Original fragments before assembly")
    print("- *_GROUND_TRUTH.ply: Correct assembly")
    print("- *_FINAL_assembled.ply: Model's prediction")
    print("- *_COMPARISON_all.ply: Side-by-side GT (left) vs Prediction (right)")
    
    print("\nSCP command to download:")
    print(f"scp {os.environ.get('USER', 'user')}@spartan.hpc.unimelb.edu.au:{os.getcwd()}/bone_*.ply .")