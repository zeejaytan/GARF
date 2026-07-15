#!/usr/bin/env python3
"""
Create HDF5 file for Tray-000 archaeological data following bone dataset structure
"""

import h5py
import numpy as np
import trimesh
import os
from pathlib import Path

def load_obj_mesh(obj_path):
    """Load OBJ file and return vertices and faces"""
    mesh = trimesh.load(obj_path)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    return vertices, faces

def create_tray_hdf5():
    """Create HDF5 file for Tray-000 data following bone structure"""
    
    print("🏺 Creating Tray-000 HDF5 dataset...")
    print("Following bone dataset structure...")
    
    # Input and output paths
    tray_dir = Path("input/Tray-000")
    output_path = "input/tray_archaeological.hdf5"
    
    # Get all sherd files
    sherd_files = sorted(tray_dir.glob("sherd*.obj"))
    print(f"📦 Found {len(sherd_files)} sherd files")
    
    if len(sherd_files) != 40:
        print(f"⚠️  Expected 40 sherds, found {len(sherd_files)}")
        return
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        print("📁 Creating HDF5 structure...")
        
        # Create artifact group (equivalent to pig_bone)
        artifact_group = f.create_group('artifact')
        
        # Create Tray-000 subgroup (equivalent to pig_bone_1, etc.)
        tray_group = artifact_group.create_group('Tray-000')
        
        # Create pieces subgroup
        pieces_group = tray_group.create_group('pieces')
        
        # Store piece names
        piece_names = []
        
        print("🔄 Processing sherds...")
        for i, sherd_file in enumerate(sherd_files):
            print(f"  Processing {sherd_file.name}... ({i+1}/40)")
            
            # Load mesh data
            try:
                vertices, faces = load_obj_mesh(sherd_file)
                
                # Create piece subgroup
                piece_group = pieces_group.create_group(str(i))
                
                # Store vertices and faces
                piece_group.create_dataset('vertices', data=vertices)
                piece_group.create_dataset('faces', data=faces)
                
                # Create dummy shared_faces (required by GARF structure)
                shared_faces = np.zeros(len(faces), dtype=np.int64)
                piece_group.create_dataset('shared_faces', data=shared_faces)
                
                # Store piece name
                piece_names.append(sherd_file.stem.encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Error processing {sherd_file.name}: {e}")
                return
        
        # Store piece names
        pieces_names_dtype = h5py.special_dtype(vlen=str)
        tray_group.create_dataset('pieces_names', data=piece_names, dtype=pieces_names_dtype)
        
        # Create dummy removal masks and order (required by GARF structure)
        # These aren't used for evaluation but needed for data loading
        num_pieces = len(piece_names)
        removal_masks = np.ones((1, num_pieces), dtype=bool)  # All pieces present
        removal_order = np.arange(num_pieces, dtype=np.int64)  # Sequential order
        
        tray_group.create_dataset('removal_masks', data=removal_masks)
        tray_group.create_dataset('removal_order', data=removal_order)
        
        # Create data_split group (required by GARF)
        data_split_group = f.create_group('data_split')
        artifact_split = data_split_group.create_group('artifact')
        
        # All data goes to validation split (used for testing)
        tray_id = b'Tray-000'
        artifact_split.create_dataset('val', data=tray_id)
        artifact_split.create_dataset('train', data=tray_id)  # Dummy
        artifact_split.create_dataset('test', data=tray_id)   # Dummy
        
        print("✅ HDF5 file created successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Structure:")
        print(f"   - artifact/Tray-000/pieces/0-39/ (vertices, faces, shared_faces)")
        print(f"   - artifact/Tray-000/pieces_names (40 names)")
        print(f"   - data_split/artifact/val,train,test -> Tray-000")

def verify_hdf5():
    """Verify the created HDF5 file"""
    print("\n🔍 Verifying HDF5 structure...")
    
    with h5py.File("input/tray_archaeological.hdf5", 'r') as f:
        print("Top-level groups:", list(f.keys()))
        
        if 'artifact' in f:
            artifact = f['artifact']
            print("artifact contents:", list(artifact.keys()))
            
            if 'Tray-000' in artifact:
                tray = artifact['Tray-000']
                print("Tray-000 structure:", list(tray.keys()))
                
                pieces = tray['pieces']
                print(f"Number of pieces: {len(pieces)}")
                
                # Check first piece
                piece_0 = pieces['0']
                print("Piece 0 datasets:", list(piece_0.keys()))
                
                vertices = piece_0['vertices']
                faces = piece_0['faces']
                print(f"Piece 0 - vertices: {vertices.shape}, faces: {faces.shape}")
        
        if 'data_split' in f:
            data_split = f['data_split']
            print("data_split groups:", list(data_split.keys()))
            if 'artifact' in data_split:
                splits = data_split['artifact']
                print("artifact splits:", list(splits.keys()))

if __name__ == "__main__":
    create_tray_hdf5()
    verify_hdf5()