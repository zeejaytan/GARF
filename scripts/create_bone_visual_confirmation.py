#!/usr/bin/env python3
"""
Create Visual Confirmation for Fixed GARF Results
================================================
Generate PLY files showing the bone fragment assembly results
"""

import os
import h5py
import json
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def apply_pose_to_mesh(mesh, pose):
    """Apply 7DOF pose to mesh"""
    translation = pose[:3]
    quaternion = pose[3:7]  # [qw, qx, qy, qz]
    
    # Convert quaternion to rotation matrix
    rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    
    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation
    
    # Apply transformation
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(transform)
    return mesh_copy

def load_bone_fragments():
    """Load bone fragments from HDF5"""
    print("📦 Loading bone fragments...")
    
    hdf5_path = "input/Fractura/bone_real.hdf5"
    
    with h5py.File(hdf5_path, 'r') as f:
        pig_bone_group = f['pig_bone']
        
        # Get the specific object from results
        object_name = "pig_bone_20"  # From the JSON results
        
        if object_name not in pig_bone_group:
            print(f"❌ {object_name} not found in dataset")
            return None, None
        
        obj_group = pig_bone_group[object_name]
        pieces_group = obj_group['pieces']
        
        fragments = []
        fragment_names = []
        
        # Load the two fragments mentioned in results
        piece_names = ["pig_bone_20_1", "pig_bone_20_2"]
        
        for piece_name in piece_names:
            if piece_name in pieces_group:
                piece_data = pieces_group[piece_name]
                
                if 'vertices' in piece_data and 'faces' in piece_data:
                    vertices = piece_data['vertices'][:]
                    faces = piece_data['faces'][:]
                    
                    # Create mesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    fragments.append(mesh)
                    fragment_names.append(piece_name)
                    print(f"  ✓ Loaded {piece_name}: {len(vertices)} vertices, {len(faces)} faces")
        
        print(f"📦 Successfully loaded {len(fragments)} bone fragments")
        return fragments, fragment_names

def create_assembly_visualization():
    """Create visual confirmation files"""
    print("🎨 Creating visual confirmation for bone assembly")
    print("=" * 60)
    
    # Load fragments
    fragments, fragment_names = load_bone_fragments()
    if not fragments:
        print("❌ Failed to load fragments")
        return
    
    # Load results
    results_path = "logs/GARF/real_bone_fractures/version_0/json_results/0.json"
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Results loaded:")
    print(f"  Object: {results['name']}")
    print(f"  Fragments: {results['num_parts']}")
    print(f"  Part accuracy: {results['part_acc']:.1%}")
    print(f"  Translation RMSE: {results['rmse_t']:.4f}")
    print(f"  Rotation RMSE: {results['rmse_r']:.4f}")
    
    # Get poses
    gt_poses = np.array(results['gt_trans_rots'])
    pred_trajectories = np.array(results['pred_trans_rots'])
    
    # Use final predicted poses
    final_pred_poses = pred_trajectories[-1]  # Last timestep
    initial_pred_poses = pred_trajectories[0]  # First timestep
    
    print(f"\n📐 Pose data:")
    print(f"  Ground truth poses: {gt_poses.shape}")
    print(f"  Prediction trajectory: {pred_trajectories.shape}")
    print(f"  Final poses: {final_pred_poses.shape}")
    
    # 1. Create scattered initial positions
    print(f"\n🔄 Creating scattered initial assembly...")
    scattered_fragments = []
    for i, (frag, pose) in enumerate(zip(fragments, initial_pred_poses)):
        scattered_mesh = apply_pose_to_mesh(frag, pose)
        scattered_fragments.append(scattered_mesh)
    
    if scattered_fragments:
        scattered_combined = trimesh.util.concatenate(scattered_fragments)
        scattered_path = "bone_fixed_INITIAL_scattered.ply"
        scattered_combined.export(scattered_path)
        print(f"  ✓ Saved: {scattered_path}")
    
    # 2. Create ground truth assembly
    print(f"\n🎯 Creating ground truth assembly...")
    gt_fragments = []
    for i, (frag, pose) in enumerate(zip(fragments, gt_poses)):
        gt_mesh = apply_pose_to_mesh(frag, pose)
        gt_fragments.append(gt_mesh)
    
    if gt_fragments:
        gt_combined = trimesh.util.concatenate(gt_fragments)
        gt_path = "bone_fixed_GROUND_TRUTH.ply"
        gt_combined.export(gt_path)
        print(f"  ✓ Saved: {gt_path}")
    
    # 3. Create GARF predicted assembly (FINAL)
    print(f"\n🔧 Creating GARF final assembly...")
    pred_fragments = []
    for i, (frag, pose) in enumerate(zip(fragments, final_pred_poses)):
        pred_mesh = apply_pose_to_mesh(frag, pose)
        pred_fragments.append(pred_mesh)
    
    if pred_fragments:
        pred_combined = trimesh.util.concatenate(pred_fragments)
        pred_path = "bone_fixed_FINAL_assembled.ply"
        pred_combined.export(pred_path)
        print(f"  ✓ Saved: {pred_path}")
    
    # 4. Create trajectory visualization (every 10th step)
    print(f"\n📈 Creating trajectory visualization...")
    trajectory_meshes = []
    
    # Color fragments differently for trajectory
    colors = [[255, 0, 0, 128], [0, 255, 0, 128]]  # Red and green with transparency
    
    step_interval = max(1, len(pred_trajectories) // 8)  # Show ~8 steps
    
    for step_idx in range(0, len(pred_trajectories), step_interval):
        step_poses = pred_trajectories[step_idx]
        
        for frag_idx, (frag, pose) in enumerate(zip(fragments, step_poses)):
            step_mesh = apply_pose_to_mesh(frag, pose)
            
            # Add color to distinguish trajectory steps
            if hasattr(step_mesh.visual, 'vertex_colors'):
                step_mesh.visual.vertex_colors = colors[frag_idx % len(colors)]
            
            trajectory_meshes.append(step_mesh)
    
    if trajectory_meshes:
        trajectory_combined = trimesh.util.concatenate(trajectory_meshes)
        trajectory_path = "bone_fixed_TRAJECTORY_steps.ply"
        trajectory_combined.export(trajectory_path)
        print(f"  ✓ Saved: {trajectory_path}")
    
    # 5. Analysis summary
    print(f"\n📊 ASSEMBLY ANALYSIS:")
    
    # Calculate fragment distances
    def analyze_assembly(poses, label):
        translations = poses[:, :3]
        centroid = np.mean(translations, axis=0)
        distances = np.linalg.norm(translations - centroid, axis=1)
        
        print(f"  {label}:")
        print(f"    Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
        print(f"    Fragment spread: {np.std(distances):.4f}")
        print(f"    Max distance: {np.max(distances):.4f}")
        
        return np.std(distances)
    
    initial_spread = analyze_assembly(initial_pred_poses, "Initial (scattered)")
    final_spread = analyze_assembly(final_pred_poses, "Final (assembled)")
    gt_spread = analyze_assembly(gt_poses, "Ground truth")
    
    # Convergence analysis
    convergence_ratio = final_spread / initial_spread
    gt_similarity = abs(final_spread - gt_spread) / gt_spread
    
    print(f"\n🎯 CONVERGENCE ANALYSIS:")
    print(f"  Convergence ratio: {convergence_ratio:.4f} (lower = better)")
    print(f"  GT similarity: {gt_similarity:.4f} (lower = better)")
    
    if convergence_ratio < 0.5:
        print(f"  ✅ EXCELLENT convergence - fragments assembled well")
    elif convergence_ratio < 0.8:
        print(f"  ✅ GOOD convergence - fragments came together")
    else:
        print(f"  ⚠️ LIMITED convergence - fragments still scattered")
    
    if gt_similarity < 0.2:
        print(f"  ✅ EXCELLENT match to ground truth")
    elif gt_similarity < 0.5:
        print(f"  ✅ GOOD match to ground truth")
    else:
        print(f"  ⚠️ POOR match to ground truth")
    
    print(f"\n📁 Generated visualization files:")
    output_files = [
        "bone_fixed_INITIAL_scattered.ply",
        "bone_fixed_GROUND_TRUTH.ply", 
        "bone_fixed_FINAL_assembled.ply",
        "bone_fixed_TRAJECTORY_steps.ply"
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"  ✓ {file_path} ({size_mb:.1f} MB)")
    
    print(f"\n🎉 VISUAL CONFIRMATION COMPLETE!")
    print(f"Open the PLY files in a 3D viewer to verify the assembly works correctly")

if __name__ == "__main__":
    create_assembly_visualization()