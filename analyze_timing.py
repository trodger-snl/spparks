#!/usr/bin/env python3
"""
Analyze timing results from SPPARKS batch processing
"""

import pandas as pd
import numpy as np

# Read the timing data
df = pd.read_csv('/Users/Tron/spparks/batch_analysis_output/timing_results.csv')

print("="*80)
print("SPPARKS BATCH PROCESSING - TIMING ANALYSIS")
print("="*80)

# Basic statistics
print(f"\nDataset Summary:")
print(f"  Total files processed: {len(df)}")
print(f"  Timestep range: {df['timestep'].min()} to {df['timestep'].max()}")
print(f"  Atom count range: {df['n_atoms'].min():,} to {df['n_atoms'].max():,}")

# Calculate total times
total_load_time = df['load_time'].sum()
total_method1_time = df['method1_time'].sum()
total_method2_time = df['method2_time'].sum()
total_processing_time = total_load_time + total_method1_time + total_method2_time

print(f"\n📊 TOTAL RUNTIME BREAKDOWN:")
print(f"  Data Loading Time:    {total_load_time:8.2f} seconds ({total_load_time/60:6.2f} minutes)")
print(f"  Method 1 (Glyphs):    {total_method1_time:8.2f} seconds ({total_method1_time/60:6.2f} minutes)")
print(f"  Method 2 (Struct):    {total_method2_time:8.2f} seconds ({total_method2_time/60:6.2f} minutes)")
print(f"  {'─'*50}")
print(f"  TOTAL PROCESSING:     {total_processing_time:8.2f} seconds ({total_processing_time/60:6.2f} minutes)")

# Percentage breakdown
print(f"\n📈 PERCENTAGE BREAKDOWN:")
print(f"  Data Loading:         {100*total_load_time/total_processing_time:6.1f}%")
print(f"  Method 1 (Glyphs):    {100*total_method1_time/total_processing_time:6.1f}%")
print(f"  Method 2 (Struct):    {100*total_method2_time/total_processing_time:6.1f}%")

# Performance comparison
print(f"\n⚡ PERFORMANCE COMPARISON:")
print(f"  Method 1 vs Method 2: {total_method1_time/total_method2_time:.2f}x")
if total_method1_time < total_method2_time:
    print(f"  → Method 1 is {total_method2_time/total_method1_time:.2f}x FASTER than Method 2")
else:
    print(f"  → Method 2 is {total_method1_time/total_method2_time:.2f}x FASTER than Method 1")

# Average times per file
print(f"\n📋 AVERAGE TIMES PER FILE:")
print(f"  Data Loading:         {df['load_time'].mean():6.3f} ± {df['load_time'].std():5.3f} seconds")
print(f"  Method 1 (Glyphs):    {df['method1_time'].mean():6.3f} ± {df['method1_time'].std():5.3f} seconds")
print(f"  Method 2 (Struct):    {df['method2_time'].mean():6.3f} ± {df['method2_time'].std():5.3f} seconds")

# Scaling analysis
correlation_load = np.corrcoef(df['n_atoms'], df['load_time'])[0,1]
correlation_m1 = np.corrcoef(df['n_atoms'], df['method1_time'])[0,1]
correlation_m2 = np.corrcoef(df['n_atoms'], df['method2_time'])[0,1]

print(f"\n📊 SCALING WITH DATASET SIZE (correlation with n_atoms):")
print(f"  Data Loading:         r = {correlation_load:.3f}")
print(f"  Method 1 (Glyphs):    r = {correlation_m1:.3f}")
print(f"  Method 2 (Struct):    r = {correlation_m2:.3f}")

# Find most expensive files
print(f"\n🔥 MOST EXPENSIVE FILES:")
print("Top 5 by total processing time:")
df['total_time'] = df['load_time'] + df['method1_time'] + df['method2_time']
top_files = df.nlargest(5, 'total_time')[['filename', 'n_atoms', 'total_time']]
for _, row in top_files.iterrows():
    print(f"  {row['filename']:20} ({row['n_atoms']:7,} atoms): {row['total_time']:6.2f} seconds")

# Processing rate
total_atoms_processed = df['n_atoms'].sum() * 2  # Each atom processed by both methods
total_render_time = total_method1_time + total_method2_time
atoms_per_second = total_atoms_processed / total_render_time

print(f"\n🚀 THROUGHPUT ANALYSIS:")
print(f"  Total atoms processed: {total_atoms_processed:,} (both methods)")
print(f"  Total rendering time:  {total_render_time:.1f} seconds")
print(f"  Processing rate:       {atoms_per_second:,.0f} atoms/second")
print(f"  Processing rate:       {atoms_per_second*60:,.0f} atoms/minute")

print("\n" + "="*80)