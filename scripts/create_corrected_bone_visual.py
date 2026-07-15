#!/usr/bin/env python3
"""
Create CORRECTED PLY files for bone fragment assembly results.
Properly handles the reference vs scattered vs assembled states.
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

def create_bone_visualization_corrected(result_file, hdf5_path, output_prefix):
    """Create CORRECTED visualization for bone fragment assembly."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    sample_name = data['name']
    sample_id = sample_name.split('/')[-1]
    pieces = data['pieces'].split(',')
    gt_trans_rots = np.array(data['gt_trans_rots'])
    pred_trans_rots = np.array(data['pred_trans_rots'])
    part_acc = data['part_acc']
    
    print(f"\nProcessing: {sample_name}")
    print(f"Part accuracy: {part_acc}")
    print(f"Translation RMSE: {data['rmse_t']:.4f}m")
    print(f"Rotation RMSE: {data['rmse_r']:.2f}°")
    
    # Analyze transformations
    print(f"\nTransformation analysis:")
    for i, (gt_tf, pred_tf) in enumerate(zip(gt_trans_rots, pred_trans_rots[-1])):
        gt_trans, gt_quat = gt_tf[:3], gt_tf[3:]
        pred_trans, pred_quat = pred_tf[:3], pred_tf[3:]
        
        trans_diff = np.linalg.norm(gt_trans - pred_trans)
        # For quaternions, check both q and -q (they represent same rotation)
        quat_diff1 = np.linalg.norm(gt_quat - pred_quat)
        quat_diff2 = np.linalg.norm(gt_quat + pred_quat)
        quat_diff = min(quat_diff1, quat_diff2)
        
        print(f"  Piece {i} ({pieces[i]}): trans_err={trans_diff:.4f}m, quat_err={quat_diff:.4f}")
        
        # Check if this is likely the anchor piece (minimal transform)
        gt_trans_norm = np.linalg.norm(gt_trans)
        is_anchor_like = gt_trans_norm < 0.1  # Less than 10cm movement
        print(f"    GT translation magnitude: {gt_trans_norm:.4f}m {'(anchor-like)' if is_anchor_like else '(moved)'}")
    
    # Load meshes from HDF5
    try:
        with h5py.File(hdf5_path, 'r') as f:
            sample_group = f['pig_bone'][sample_id]
            pieces_group = sample_group['pieces']
            pieces_names = [name.decode() for name in sample_group['pieces_names']]
            
            # Load fragment meshes
            fragments = []
            piece_indices = []
            
            for piece in pieces:
                if piece in pieces_names:
                    piece_idx = pieces_names.index(piece)
                    piece_indices.append(piece_idx)
                else:
                    print(f"Warning: {piece} not found")
                    return
            
            for i, piece_idx in enumerate(piece_indices):
                piece_group = pieces_group[str(piece_idx)]
                vertices = np.array(piece_group['vertices'])
                faces = np.array(piece_group['faces'])
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                fragments.append(mesh)
                print(f"Loaded {pieces[i]}: {vertices.shape[0]} vertices")
            
            # 1. REFERENCE STATE: Fragments as stored in HDF5 (no transforms applied)
            ref_meshes = []
            for i, mesh in enumerate(fragments):
                mesh_copy = mesh.copy()
                mesh_copy.visual.face_colors = [255, 100, 100, 200] if i == 0 else [100, 255, 100, 200]
                ref_meshes.append(mesh_copy)
            
            if ref_meshes:
                ref_combined = trimesh.util.concatenate(ref_meshes)
                ref_filename = f"{output_prefix}_REFERENCE_positions.ply"
                ref_combined.export(ref_filename)
                print(f"Saved reference state: {ref_filename}")
            
            # 2. GROUND TRUTH ASSEMBLY: Apply GT transforms
            gt_meshes = []
            for i, (mesh, transform) in enumerate(zip(fragments, gt_trans_rots)):
                mesh_copy = mesh.copy()
                translation = transform[:3]
                quaternion = transform[3:]
                
                mesh_copy.vertices = apply_transform(mesh_copy.vertices, translation, quaternion)
                mesh_copy.visual.face_colors = [255, 100, 100, 200] if i == 0 else [100, 255, 100, 200]
                gt_meshes.append(mesh_copy)
            
            if gt_meshes:
                gt_combined = trimesh.util.concatenate(gt_meshes)
                gt_filename = f"{output_prefix}_GROUND_TRUTH_assembly.ply"
                gt_combined.export(gt_filename)
                print(f"Saved ground truth assembly: {gt_filename}")
            
            # 3. PREDICTED ASSEMBLY: Apply predicted transforms
            final_pred = pred_trans_rots[-1]
            pred_meshes = []
            for i, (mesh, transform) in enumerate(zip(fragments, final_pred)):
                mesh_copy = mesh.copy()
                translation = transform[:3]
                quaternion = transform[3:]
                
                mesh_copy.vertices = apply_transform(mesh_copy.vertices, translation, quaternion)
                mesh_copy.visual.face_colors = [255, 100, 100, 200] if i == 0 else [100, 255, 100, 200]
                pred_meshes.append(mesh_copy)
            
            if pred_meshes:
                pred_combined = trimesh.util.concatenate(pred_meshes)
                pred_filename = f"{output_prefix}_PREDICTED_assembly.ply"
                pred_combined.export(pred_filename)
                print(f"Saved predicted assembly: {pred_filename}")
            
            # 4. THREE-WAY COMPARISON: Reference | GT Assembly | Predicted Assembly
            if ref_meshes and gt_meshes and pred_meshes:
                # Offset each assembly horizontally for comparison
                spacing = 0.4  # 40cm spacing
                
                # GT assembly in middle (no offset)
                gt_comparison = [mesh.copy() for mesh in gt_meshes]
                
                # Predicted assembly on right
                pred_comparison = []
                for mesh in pred_meshes:
                    mesh_copy = mesh.copy()
                    mesh_copy.vertices += np.array([spacing, 0, 0])
                    # Blue tones for prediction
                    mesh_copy.visual.face_colors = [100, 100, 255, 200] if len(pred_comparison) == 0 else [100, 255, 255, 200]
                    pred_comparison.append(mesh_copy)
                
                # Reference state on left
                ref_comparison = []
                for mesh in ref_meshes:
                    mesh_copy = mesh.copy()
                    mesh_copy.vertices += np.array([-spacing, 0, 0])
                    # Gray tones for reference
                    mesh_copy.visual.face_colors = [150, 150, 150, 200] if len(ref_comparison) == 0 else [200, 200, 200, 200]
                    ref_comparison.append(mesh_copy)
                
                all_comparison = ref_comparison + gt_comparison + pred_comparison
                comparison_combined = trimesh.util.concatenate(all_comparison)
                comparison_filename = f"{output_prefix}_THREE_WAY_comparison.ply"
                comparison_combined.export(comparison_filename)
                print(f"Saved three-way comparison: {comparison_filename}")
                print(f"  Left (gray): Reference positions")
                print(f"  Center (red/green): Ground truth assembly")
                print(f"  Right (blue/cyan): Predicted assembly")
    
    except Exception as e:
        print(f"Error processing {sample_name}: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    results_dir = "logs/GARF/real_bone_fractures/version_0/json_results"
    hdf5_path = "input/Fractura/bone_real.hdf5"
    
    # Process best case
    print("=" * 70)
    print("BEST CASE: pig_bone_20 (100% part accuracy)")
    print("=" * 70)
    create_bone_visualization_corrected(
        f"{results_dir}/0.json",
        hdf5_path,
        "bone_CORRECTED_best"
    )
    
    # Process challenging case  
    print("\n" + "=" * 70)
    print("CHALLENGING CASE: pig_bone_27 (50% part accuracy)")
    print("=" * 70)
    create_bone_visualization_corrected(
        f"{results_dir}/18.json",
        hdf5_path,
        "bone_CORRECTED_challenging"
    )
    
    print("\n" + "=" * 70)
    print("CORRECTED VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated corrected PLY files:")
    os.system("ls -lh bone_CORRECTED_*.ply")
    
    print("\nFile descriptions:")
    print("- *_REFERENCE_positions.ply: Original fragments as stored in HDF5")
    print("- *_GROUND_TRUTH_assembly.ply: Correct assembly after applying GT transforms") 
    print("- *_PREDICTED_assembly.ply: Model's assembly after applying predicted transforms")
    print("- *_THREE_WAY_comparison.ply: All three states side by side")
    
    print("\nSCP command:")
    print(f"scp {os.environ.get('USER', 'user')}@spartan.hpc.unimelb.edu.au:{os.getcwd()}/bone_CORRECTED_*.ply .")