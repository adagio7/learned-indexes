"""
Multi-Distribution Adversarial Stress Test Visualization

This script parses 'build/adversarial_results.csv' to compare the 
static baseline and adaptive learned index across Lognormal, Sine Wave, 
and Gaussian distributions.

It generates bar charts with speedup annotations based on the 
scientific zero-contention measurement protocol.

Output: figures/adversarial_plot.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_multi():
    try:
        df = pd.read_csv('build/adversarial_results.csv')
    except:
        df = pd.read_csv('adversarial_results.csv')
        
    dists = df['Distribution'].unique()
    indices = df['Index'].unique()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bar_width = 0.35
    x = np.arange(len(dists))
    
    for i, idx in enumerate(indices):
        data = df[df['Index'] == idx]['TimeMS'].values
        ax.bar(x + (i * bar_width) - (bar_width/2), data, bar_width, label=idx)
        
        # Add labels on top of bars
        for j, val in enumerate(data):
            ax.text(j + (i * bar_width) - (bar_width/2), val + 0.5, f'{val:.1f}ms', 
                    ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Data Distribution')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Adversarial Stress Test: Static vs. Adaptive Learned Index')
    ax.set_xticks(x)
    ax.set_xticklabels(dists)
    ax.legend()
    
    # Add speedup annotations
    for j, dist in enumerate(dists):
        b_series = df[(df['Distribution'] == dist) & (df['Index'] == 'Baseline')]['TimeMS']
        l_series = df[(df['Distribution'] == dist) & (df['Index'] == 'Learned')]['TimeMS']
        if len(b_series) > 0 and len(l_series) > 0:
            b_val = b_series.values[0]
            l_val = l_series.values[0]
            speedup = b_val / l_val
            ax.text(j, max(b_val, l_val) * 1.15, f'{speedup:.2f}x Speedup', 
                    ha='center', color='green' if speedup > 1 else 'red', 
                    fontweight='extra bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, df['TimeMS'].max() * 1.4)
    plt.tight_layout()
    plt.savefig('figures/adversarial_plot.png', dpi=300)
    print("Multi-distribution plot saved to figures/adversarial_plot.png")

if __name__ == "__main__":
    plot_multi()
