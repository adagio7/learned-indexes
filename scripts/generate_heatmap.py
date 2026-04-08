"""
Query Profiling & Heatmap Generator

This script analyzes 'build/profile_data.csv' to provide a deep dive into 
the Adaptive Learned Index's performance.

It generates:
1. figures/performance_composition.png: Breakdown of Sync, Selection, and Scan.
2. figures/profile_heatmap.png: Latency trends over a sequence of 10k queries.

Data is aggregated into buckets to highlight architectural bottlenecks.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_heatmap():
    csv_file = "build/profile_data.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    
    # Calculate means
    mean_sync = df['Sync_ns'].mean()
    mean_selection = df['Selection_ns'].mean()
    mean_model = df['ModelSearch_ns'].mean()
    
    print(f"Average Sync (ns): {mean_sync:.2f}")
    print(f"Average Selection (ns): {mean_selection:.2f}")
    print(f"Average Model (ns): {mean_model:.2f}")

    # 1. Pie Chart of Time Composition
    labels = ['Sync (Atomic Load)', 'Selection (Binary Search)', 'Model Pred + Scan']
    sizes = [mean_sync, mean_selection, mean_model]
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, shadow=True)
    plt.title('LearnedIndex Query Latency Breakdown (Per Query)')
    plt.axis('equal')
    plt.savefig('figures/performance_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Latency Heatmap over Query Sequence (Pure Matplotlib)
    bucket_size = 100
    df['Bucket'] = df['QueryID'] // bucket_size
    heatmap_data = df.groupby('Bucket')[['Sync_ns', 'Selection_ns', 'ModelSearch_ns']].mean().values.T
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Latency (ns)')
    
    plt.title(f'Latency Heatmap over 10,000 Queries (Buckets of {bucket_size})')
    plt.yticks([0, 1, 2], ['Sync', 'Selection', 'Model'])
    plt.xlabel('Bucket (100 queries each)')
    plt.savefig('figures/profile_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots saved: figures/performance_composition.png, figures/profile_heatmap.png")

if __name__ == "__main__":
    generate_heatmap()
