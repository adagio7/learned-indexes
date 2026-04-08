"""
Traditional Throughput Visualization Script

This script reads 'build/results.csv' to visualize the performance of the 
Workload-Aware Learned Index against a Static Baseline over a sequence 
of query blocks.

Output: figures/latency_plot.png
"""
import csv
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    csv_file = "build/results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        exit(1)
        
    blocks = []
    baseline_times = []
    learned_times = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            blocks.append(int(row['Block']))
            baseline_times.append(float(row['BaselineTime']))
            learned_times.append(float(row['LearnedTime']))
    
    plt.figure(figsize=(10, 6))
    plt.plot(blocks, baseline_times, marker='o', label='Static Baseline Index')
    plt.plot(blocks, learned_times, marker='s', label='Workload-Aware Learned Index')
    
    plt.title('Query Latency Over Time (Zipfian Workload)')
    plt.xlabel('Query Block (100k queries per block)')
    plt.ylabel('Latency (ms)')
    plt.xticks(blocks)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = "figures/latency_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
