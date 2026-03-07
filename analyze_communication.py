"""
Analyze communication overhead for hierarchy vs fully connected.
Excludes heartbeat messages and shows per-round breakdown.
"""
import pandas as pd
import numpy as np

# Model size in bytes
MODEL_SIZE_MB = 8.37
MODEL_SIZE_BYTES = MODEL_SIZE_MB * 1024 * 1024  # 8,774,778 bytes

print("=" * 80)
print("Communication Analysis: Hierarchy vs Fully Connected")
print("=" * 80)
print(f"Model size: {MODEL_SIZE_MB} MB ({MODEL_SIZE_BYTES:,} bytes)")
print("=" * 80)

# Read the CSV files
try:
    # Hierarchy communication (100 rounds)
    hierarchy_df = pd.read_csv('outputs/communication/c6-d06-node_packets.csv')
    
    # Fully connected communication (3 rounds)  
    fully_df = pd.read_csv('outputs/communication/messages.csv')
    
    print("\n1. RAW DATA OVERVIEW")
    print("-" * 80)
    print(f"Hierarchy CSV: {len(hierarchy_df):,} total rows")
    print(f"Fully connected CSV: {len(fully_df):,} total rows")
    
    # Filter out heartbeat messages
    print("\n2. FILTERING HEARTBEAT MESSAGES AND SMALL PACKETS")
    print("-" * 80)
    
    # For hierarchy: exclude heartbeat-related commands and small packets
    hierarchy_filtered = hierarchy_df[~hierarchy_df['cmd'].str.contains('beat|Beat', case=False, na=False)]
    if 'packet_bytes' in hierarchy_filtered.columns:
        hierarchy_filtered = hierarchy_filtered[hierarchy_filtered['packet_bytes'] > 2000]
    print(f"Hierarchy after filtering: {len(hierarchy_filtered):,} rows (only packets > 2000 bytes)")
    
    # For fully connected: exclude heartbeat messages and small packets
    fully_filtered = fully_df[~fully_df['cmd'].str.contains('beat|Beat', case=False, na=False)]
    if 'package_size' in fully_filtered.columns:
        fully_filtered = fully_filtered[fully_filtered['package_size'] > 2000]
    print(f"Fully connected after filtering: {len(fully_filtered):,} rows (only packets > 2000 bytes)")
    
    # Replace all package_size > 2000 with model size (8.37 MB) for fully connected
    if 'package_size' in fully_filtered.columns:
        large_packets = len(fully_filtered)
        fully_filtered.loc[:, 'package_size'] = MODEL_SIZE_BYTES
        print(f"Replaced {large_packets} packets with model size ({MODEL_SIZE_MB} MB)")
    
    # Analyze hierarchy communication PER ROUND
    print("\n3. HIERARCHY COMMUNICATION - PER ROUND BREAKDOWN")
    print("=" * 80)
    
    if 'packet_bytes' in hierarchy_filtered.columns:
        hierarchy_sent = hierarchy_filtered[hierarchy_filtered['direction'] == 'sent']
        
        # Group by round
        hierarchy_per_round = hierarchy_sent.groupby('round').agg({
            'packet_bytes': ['sum', 'count']
        }).reset_index()
        hierarchy_per_round.columns = ['round', 'total_bytes', 'num_packets']
        hierarchy_per_round['total_mb'] = hierarchy_per_round['total_bytes'] / (1024 * 1024)
        
        print(f"{'Round':<10} {'Packets':<15} {'Bytes':<20} {'MB':<15} {'Models':<10}")
        print("-" * 80)
        
        for _, row in hierarchy_per_round.iterrows():
            models = row['total_bytes'] / MODEL_SIZE_BYTES
            print(f"{int(row['round']):<10} {int(row['num_packets']):<15} {int(row['total_bytes']):>18,}  {row['total_mb']:>13.2f}  {models:>8.2f}x")
        
        # Summary
        total_hierarchy_bytes = hierarchy_per_round['total_bytes'].sum()
        total_hierarchy_packets = hierarchy_per_round['num_packets'].sum()
        avg_bytes = hierarchy_per_round['total_bytes'].mean()
        avg_packets = hierarchy_per_round['num_packets'].mean()
        
        print("-" * 80)
        print(f"{'AVERAGE':<10} {avg_packets:<15.0f} {avg_bytes:>18,.0f}  {avg_bytes/(1024*1024):>13.2f}  {avg_bytes/MODEL_SIZE_BYTES:>8.2f}x")
        print(f"{'TOTAL':<10} {total_hierarchy_packets:<15.0f} {total_hierarchy_bytes:>18,}  {total_hierarchy_bytes/(1024*1024):>13.2f}")
        
    # Analyze fully connected communication PER ROUND
    print("\n4. FULLY CONNECTED COMMUNICATION - PER ROUND BREAKDOWN")
    print("=" * 80)
    
    if 'package_size' in fully_filtered.columns:
        fully_sent = fully_filtered[fully_filtered['direction'] == 'sent']
        
        # Group by round
        fully_per_round = fully_sent.groupby('round').agg({
            'package_size': ['sum', 'count']
        }).reset_index()
        fully_per_round.columns = ['round', 'total_bytes', 'num_packets']
        fully_per_round['total_mb'] = fully_per_round['total_bytes'] / (1024 * 1024)
        
        print(f"{'Round':<10} {'Packets':<15} {'Bytes':<20} {'MB':<15} {'Models':<10}")
        print("-" * 80)
        
        for _, row in fully_per_round.iterrows():
            models = row['total_bytes'] / MODEL_SIZE_BYTES
            print(f"{int(row['round']):<10} {int(row['num_packets']):<15} {int(row['total_bytes']):>18,}  {row['total_mb']:>13.2f}  {models:>8.2f}x")
        
        # Summary
        total_fully_bytes = fully_per_round['total_bytes'].sum()
        total_fully_packets = fully_per_round['num_packets'].sum()
        avg_bytes_fully = fully_per_round['total_bytes'].mean()
        avg_packets_fully = fully_per_round['num_packets'].mean()
        
        print("-" * 80)
        print(f"{'AVERAGE':<10} {avg_packets_fully:<15.0f} {avg_bytes_fully:>18,.0f}  {avg_bytes_fully/(1024*1024):>13.2f}  {avg_bytes_fully/MODEL_SIZE_BYTES:>8.2f}x")
        print(f"{'TOTAL':<10} {total_fully_packets:<15.0f} {total_fully_bytes:>18,}  {total_fully_bytes/(1024*1024):>13.2f}")
    
    # Comparison
    print("\n5. COMPARISON (Average Per Round)")
    print("=" * 80)
    
    if 'packet_bytes' in hierarchy_filtered.columns and 'package_size' in fully_filtered.columns:
        reduction_bytes = (1 - avg_bytes / avg_bytes_fully) * 100
        reduction_packets = (1 - avg_packets / avg_packets_fully) * 100
        
        print(f"{'Metric':<30} {'Hierarchy':<20} {'Fully Connected':<20} {'Reduction':<15}")
        print("-" * 80)
        print(f"{'Packets per round':<30} {avg_packets:<20,.0f} {avg_packets_fully:<20,.0f} {reduction_packets:>13.1f}%")
        print(f"{'Bytes per round':<30} {avg_bytes:<20,.0f} {avg_bytes_fully:<20,.0f} {reduction_bytes:>13.1f}%")
        print(f"{'MB per round':<30} {avg_bytes/(1024*1024):<20.2f} {avg_bytes_fully/(1024*1024):<20.2f} {reduction_bytes:>13.1f}%")
        print(f"{'Models per round':<30} {avg_bytes/MODEL_SIZE_BYTES:<20.2f} {avg_bytes_fully/MODEL_SIZE_BYTES:<20.2f}")
        
        print("\n" + "=" * 80)
        print(f"RESULT: Hierarchy uses {avg_bytes/avg_bytes_fully:.1%} of fully connected communication")
        print(f"        Communication reduction: {reduction_bytes:.1f}%")
        print("=" * 80)

except FileNotFoundError as e:
    print(f"\nError: Could not find CSV file - {e}")
    print("\nPlease ensure the CSV files are in outputs/communication/")
except Exception as e:
    print(f"\nError analyzing data: {e}")
    import traceback
    traceback.print_exc()

print()
