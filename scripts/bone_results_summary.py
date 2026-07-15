#!/usr/bin/env python3
"""
Create a comprehensive summary of bone fragment assembly results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_bone_results():
    """Analyze all bone fragment results and create summary."""
    results_dir = Path("logs/GARF/real_bone_fractures/version_0/json_results")
    results = []
    
    # Load all results
    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results.append({
            'file_id': json_file.stem,
            'sample_name': data['name'],
            'pieces': data['pieces'],
            'num_parts': data['num_parts'],
            'part_accuracy': data['part_acc'],
            'rmse_translation': data['rmse_t'],
            'rmse_rotation': data['rmse_r'],
            'shape_chamfer_distance': data['shape_cd'],
            'mesh_scale': data['mesh_scale']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = {
        'total_samples': len(df),
        'mean_part_accuracy': df['part_accuracy'].mean(),
        'perfect_assemblies': (df['part_accuracy'] == 1.0).sum(),
        'partial_assemblies': ((df['part_accuracy'] > 0) & (df['part_accuracy'] < 1.0)).sum(),
        'failed_assemblies': (df['part_accuracy'] == 0.0).sum(),
        'mean_rmse_translation': df['rmse_translation'].mean(),
        'mean_rmse_rotation': df['rmse_rotation'].mean(),
        'std_rmse_translation': df['rmse_translation'].std(),
        'std_rmse_rotation': df['rmse_rotation'].std(),
        'min_rmse_translation': df['rmse_translation'].min(),
        'max_rmse_translation': df['rmse_translation'].max(),
        'min_rmse_rotation': df['rmse_rotation'].min(),
        'max_rmse_rotation': df['rmse_rotation'].max()
    }
    
    # Save detailed results
    df.to_csv('bone_fragments_detailed_results.csv', index=False)
    
    # Save summary statistics
    with open('bone_fragments_summary.txt', 'w') as f:
        f.write("BONE FRAGMENT ASSEMBLY RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total samples tested: {summary_stats['total_samples']}\n")
        f.write(f"Perfect assemblies (100%): {summary_stats['perfect_assemblies']}\n")
        f.write(f"Partial assemblies (>0% <100%): {summary_stats['partial_assemblies']}\n")
        f.write(f"Failed assemblies (0%): {summary_stats['failed_assemblies']}\n\n")
        
        f.write(f"Mean part accuracy: {summary_stats['mean_part_accuracy']:.3f}\n")
        f.write(f"Success rate (≥50%): {(df['part_accuracy'] >= 0.5).sum()}/{len(df)} ({(df['part_accuracy'] >= 0.5).mean()*100:.1f}%)\n\n")
        
        f.write("TRANSLATION ERRORS (meters):\n")
        f.write(f"  Mean RMSE: {summary_stats['mean_rmse_translation']:.4f}\n")
        f.write(f"  Std RMSE: {summary_stats['std_rmse_translation']:.4f}\n")
        f.write(f"  Range: {summary_stats['min_rmse_translation']:.4f} - {summary_stats['max_rmse_translation']:.4f}\n\n")
        
        f.write("ROTATION ERRORS (degrees):\n")
        f.write(f"  Mean RMSE: {summary_stats['mean_rmse_rotation']:.2f}\n")
        f.write(f"  Std RMSE: {summary_stats['std_rmse_rotation']:.2f}\n")
        f.write(f"  Range: {summary_stats['min_rmse_rotation']:.2f} - {summary_stats['max_rmse_rotation']:.2f}\n\n")
        
        f.write("BEST CASES (Part Accuracy = 1.0):\n")
        perfect_cases = df[df['part_accuracy'] == 1.0]
        for _, row in perfect_cases.iterrows():
            f.write(f"  {row['sample_name']}: T_RMSE={row['rmse_translation']:.4f}m, R_RMSE={row['rmse_rotation']:.2f}°\n")
        
        f.write("\nWORST CASES (Part Accuracy < 0.5):\n")
        poor_cases = df[df['part_accuracy'] < 0.5]
        for _, row in poor_cases.iterrows():
            f.write(f"  {row['sample_name']}: Accuracy={row['part_accuracy']:.1f}, T_RMSE={row['rmse_translation']:.4f}m, R_RMSE={row['rmse_rotation']:.2f}°\n")
    
    return df, summary_stats

if __name__ == "__main__":
    print("Analyzing bone fragment assembly results...")
    df, stats = analyze_bone_results()
    
    print(f"\nProcessed {len(df)} bone fragment samples")
    print(f"Mean part accuracy: {stats['mean_part_accuracy']:.3f}")
    print(f"Perfect assemblies: {stats['perfect_assemblies']}")
    print(f"Success rate (≥50%): {(df['part_accuracy'] >= 0.5).sum()}/{len(df)}")
    
    print("\nFiles created:")
    print("- bone_fragments_detailed_results.csv")
    print("- bone_fragments_summary.txt")