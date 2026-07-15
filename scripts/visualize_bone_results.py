#!/usr/bin/env python3
"""
Visualize bone fragment assembly results from GARF evaluation.
Creates PLY files for the best and worst cases for easy download.
"""

import json
import numpy as np
import trimesh
import os
from pathlib import Path

def create_visualization(result_file, data_root, output_prefix):
    """Create visualization for a specific result."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    sample_name = data['name']
    pieces = data['pieces'].split(',')
    gt_trans_rots = np.array(data['gt_trans_rots'])
    pred_trans_rots = np.array(data['pred_trans_rots'])
    
    print(f"\nProcessing: {sample_name}")
    print(f"Pieces: {pieces}")
    print(f"Part accuracy: {data['part_acc']}")
    print(f"Translation RMSE: {data['rmse_t']:.4f}")
    print(f"Rotation RMSE: {data['rmse_r']:.4f}")
    
    # Load original fragment meshes
    fragments = []
    for piece in pieces:
        mesh_path = os.path.join(data_root, f"{piece}.obj")
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
            fragments.append(mesh)
            print(f"Loaded: {mesh_path}")
        else:
            print(f"Warning: {mesh_path} not found")
    
    if not fragments:
        print("No fragments loaded, skipping...")
        return
    
    # Create ground truth assembly
    gt_assembly = []
    for i, (mesh, transform) in enumerate(zip(fragments, gt_trans_rots)):
        mesh_copy = mesh.copy()
        # Apply transformation: [tx, ty, tz, qw, qx, qy, qz]
        translation = transform[:3]
        quaternion = transform[3:]  # [w, x, y, z]
        
        # Convert quaternion to rotation matrix
        w, x, y, z = quaternion
        rotation_matrix = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation
        
        mesh_copy.apply_transform(transform_matrix)
        mesh_copy.visual.face_colors = [255, 0, 0, 128] if i == 0 else [0, 255, 0, 128]  # Red for first, green for second
        gt_assembly.append(mesh_copy)
    
    # Create predicted assembly (using last prediction)
    pred_assembly = []
    final_pred = pred_trans_rots[-1]  # Use final prediction
    for i, (mesh, transform) in enumerate(zip(fragments, final_pred)):
        mesh_copy = mesh.copy()
        translation = transform[:3]
        quaternion = transform[3:]
        
        w, x, y, z = quaternion
        rotation_matrix = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation
        
        mesh_copy.apply_transform(transform_matrix)
        mesh_copy.visual.face_colors = [255, 0, 0, 128] if i == 0 else [0, 255, 0, 128]
        pred_assembly.append(mesh_copy)
    
    # Save visualizations
    if gt_assembly:
        gt_combined = trimesh.util.concatenate(gt_assembly)
        gt_filename = f"{output_prefix}_GROUND_TRUTH.ply"
        gt_combined.export(gt_filename)
        print(f"Saved ground truth: {gt_filename}")
    
    if pred_assembly:
        pred_combined = trimesh.util.concatenate(pred_assembly)
        pred_filename = f"{output_prefix}_PREDICTED.ply"
        pred_combined.export(pred_filename)
        print(f"Saved prediction: {pred_filename}")
    
    # Save individual fragments for reference
    for i, (piece, mesh) in enumerate(zip(pieces, fragments)):
        fragment_filename = f"{output_prefix}_fragment_{i+1}_{piece}.ply"
        mesh.export(fragment_filename)
        print(f"Saved fragment: {fragment_filename}")

# Main execution
if __name__ == "__main__":
    results_dir = "logs/GARF/real_bone_fractures/version_0/json_results"
    data_root = "input"  # Adjust this path if needed
    
    # Process best case (pig_bone_20 - 100% accuracy)
    print("=" * 60)
    print("BEST CASE: pig_bone_20 (100% part accuracy)")
    print("=" * 60)
    create_visualization(
        f"{results_dir}/0.json",
        data_root,
        "bone_best_case"
    )
    
    # Process challenging case (pig_bone_27 - 50% accuracy)
    print("\n" + "=" * 60)
    print("CHALLENGING CASE: pig_bone_27 (50% part accuracy)")
    print("=" * 60)
    create_visualization(
        f"{results_dir}/18.json",
        data_root,
        "bone_challenging_case"
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print("Generated files:")
    for pattern in ["bone_*_GROUND_TRUTH.ply", "bone_*_PREDICTED.ply", "bone_*_fragment_*.ply"]:
        os.system(f"ls -la {pattern} 2>/dev/null")