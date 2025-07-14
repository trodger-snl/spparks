#!/usr/bin/env python3
"""
Interactive HDF5 Unstructured Domain Explorer for SPPARKS

Enhanced interactive visualization tool with GUI for exploring unstructured HDF5 
thermal domains and optimizing SPPARKS subdomain decomposition.

Features:
- Interactive 3D visualization with PyVista
- GUI controls for parameter adjustment
- Real-time load balancing analysis
- Dynamic processor decomposition testing
- Export capabilities for SPPARKS configurations

Author: Claude Code Assistant
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Import the base classes from the main visualizer
from hdf5_domain_visualizer import HDF5Reader, DomainVisualizer, SubdomainAnalyzer

try:
    import pyvista as pv
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    ADVANCED_FEATURES = True
except ImportError:
    print("Advanced features require: pip install pyvista")
    ADVANCED_FEATURES = False
    

class InteractiveExplorer:
    """Interactive GUI for HDF5 domain exploration and SPPARKS optimization"""
    
    def __init__(self):
        """Initialize the interactive explorer"""
        self.reader = None
        self.visualizer = None
        self.analyzer = None
        self.domain_bounds = None
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title("HDF5 Domain Explorer for SPPARKS")
        self.root.geometry("800x600")
        
        # Variables for controls
        self.hdf5_file = tk.StringVar()
        self.layer_idx = tk.IntVar(value=0)
        self.nx_proc = tk.IntVar(value=2)
        self.ny_proc = tk.IntVar(value=2) 
        self.nz_proc = tk.IntVar(value=2)
        self.lattice_spacing = tk.DoubleVar(value=1e-6)
        self.domain_margin = tk.DoubleVar(value=0.001)
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding="10")
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(file_frame, text="HDF5 File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.hdf5_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        ttk.Button(file_frame, text="Load", command=self.load_file).grid(row=0, column=3)
        
        # Domain analysis frame
        analysis_frame = ttk.LabelFrame(self.root, text="Domain Analysis", padding="10")
        analysis_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(analysis_frame, text="Layer Index:").grid(row=0, column=0, sticky="w")
        ttk.Scale(analysis_frame, from_=0, to=33, variable=self.layer_idx, 
                 orient="horizontal").grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(analysis_frame, textvariable=self.layer_idx).grid(row=0, column=2)
        
        # SPPARKS configuration frame
        spparks_frame = ttk.LabelFrame(self.root, text="SPPARKS Configuration", padding="10")
        spparks_frame.pack(fill="x", padx=10, pady=5)
        
        # Processor decomposition
        ttk.Label(spparks_frame, text="Processors (nx × ny × nz):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(spparks_frame, from_=1, to=16, width=5, textvariable=self.nx_proc).grid(row=0, column=1)
        ttk.Label(spparks_frame, text="×").grid(row=0, column=2)
        ttk.Spinbox(spparks_frame, from_=1, to=16, width=5, textvariable=self.ny_proc).grid(row=0, column=3)
        ttk.Label(spparks_frame, text="×").grid(row=0, column=4)
        ttk.Spinbox(spparks_frame, from_=1, to=16, width=5, textvariable=self.nz_proc).grid(row=0, column=5)
        
        # Lattice spacing
        ttk.Label(spparks_frame, text="Lattice Spacing (m):").grid(row=1, column=0, sticky="w")
        ttk.Entry(spparks_frame, textvariable=self.lattice_spacing, width=15).grid(row=1, column=1, columnspan=2, sticky="w")
        
        # Domain margin
        ttk.Label(spparks_frame, text="Domain Margin (m):").grid(row=2, column=0, sticky="w")
        ttk.Entry(spparks_frame, textvariable=self.domain_margin, width=15).grid(row=2, column=1, columnspan=2, sticky="w")
        
        # Action buttons frame
        action_frame = ttk.LabelFrame(self.root, text="Actions", padding="10")
        action_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(action_frame, text="Analyze Domain", command=self.analyze_domain).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Visualize 3D", command=self.visualize_3d).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Load Balance Analysis", command=self.analyze_load_balance).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Export Config", command=self.export_config).pack(side="left", padx=5)
        
        # Results text area
        results_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding="10")
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap="word")
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def browse_file(self):
        """Browse for HDF5 file"""
        filename = filedialog.askopenfilename(
            title="Select HDF5 file",
            filetypes=[("HDF5 files", "*.hdf5 *.h5"), ("All files", "*.*")]
        )
        if filename:
            self.hdf5_file.set(filename)
    
    def load_file(self):
        """Load the selected HDF5 file"""
        try:
            if not self.hdf5_file.get():
                messagebox.showerror("Error", "Please select an HDF5 file first")
                return
                
            self.reader = HDF5Reader(self.hdf5_file.get())
            if self.reader.open():
                self.visualizer = DomainVisualizer(self.reader)
                self.analyzer = SubdomainAnalyzer(self.reader)
                
                # Update layer scale maximum
                for widget in self.root.winfo_children():
                    if isinstance(widget, ttk.LabelFrame) and widget.cget("text") == "Domain Analysis":
                        for child in widget.winfo_children():
                            if isinstance(child, ttk.Scale):
                                child.configure(to=self.reader.num_layers-1)
                
                self.log_result(f"Successfully loaded: {self.hdf5_file.get()}")
                self.log_result(f"Layers: {self.reader.num_layers}")
                self.log_result(f"Time range: {self.reader.layer_times[0]:.3f} to {self.reader.layer_times[-1]:.3f}s")
                
                # Get domain bounds
                min_bounds, max_bounds = self.reader.get_domain_bounds(0)
                self.domain_bounds = (min_bounds, max_bounds)
                self.log_result(f"Domain bounds: X[{min_bounds[0]:.6f}, {max_bounds[0]:.6f}] "
                              f"Y[{min_bounds[1]:.6f}, {max_bounds[1]:.6f}] "
                              f"Z[{min_bounds[2]:.6f}, {max_bounds[2]:.6f}]")
                
            else:
                messagebox.showerror("Error", "Failed to open HDF5 file")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
    def analyze_domain(self):
        """Analyze the current domain and layer"""
        if not self.reader:
            messagebox.showerror("Error", "Please load an HDF5 file first")
            return
            
        try:
            layer_idx = self.layer_idx.get()
            
            # Get layer information
            layer_info = self.reader.get_layer_info(layer_idx)
            self.log_result(f"\n--- Layer {layer_idx} Analysis ---")
            self.log_result(f"Time: {layer_info['layer_time']:.3f}s")
            self.log_result(f"Chunks: {layer_info['num_chunks']}")
            self.log_result(f"Nodes: {layer_info['num_nodes']}")
            self.log_result(f"Elements: {layer_info['num_elements']}")
            self.log_result(f"Max time points: {layer_info['max_time_points']}")
            
            # Memory usage
            memory_usage = self.reader.estimate_memory_usage(layer_idx)
            self.log_result(f"Estimated memory: {memory_usage['total_mb']:.1f} MB")
            
            # Chunk statistics
            chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
            sizes = chunk_bboxes[:, 3:] - chunk_bboxes[:, :3]
            volumes = np.prod(sizes, axis=1)
            
            self.log_result(f"Chunk volume statistics:")
            self.log_result(f"  Mean: {np.mean(volumes):.2e} m³")
            self.log_result(f"  Std:  {np.std(volumes):.2e} m³")
            self.log_result(f"  Min:  {np.min(volumes):.2e} m³")
            self.log_result(f"  Max:  {np.max(volumes):.2e} m³")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing domain: {str(e)}")
    
    def visualize_3d(self):
        """Create 3D visualization using PyVista"""
        if not ADVANCED_FEATURES:
            messagebox.showerror("Error", "3D visualization requires PyVista: pip install pyvista")
            return
            
        if not self.reader:
            messagebox.showerror("Error", "Please load an HDF5 file first")
            return
            
        try:
            layer_idx = self.layer_idx.get()
            chunk_bboxes = self.reader.get_chunk_bboxes(layer_idx)
            
            # Create PyVista plotter
            plotter = pv.Plotter(window_size=(1200, 800))
            plotter.set_background('white')
            
            # Add chunk representations (subsample for performance)
            max_chunks = 200
            if len(chunk_bboxes) > max_chunks:
                indices = np.random.choice(len(chunk_bboxes), max_chunks, replace=False)
                chunk_bboxes = chunk_bboxes[indices]
            
            # Create boxes for each chunk
            for i, bbox in enumerate(chunk_bboxes):
                bounds = [bbox[0], bbox[3], bbox[1], bbox[4], bbox[2], bbox[5]]
                box = pv.Box(bounds=bounds)
                volume = np.prod(bbox[3:] - bbox[:3])
                plotter.add_mesh(box, style='wireframe', color='blue', opacity=0.3)
            
            # Add SPPARKS domain overlay if configured
            if self.domain_bounds:
                min_bounds, max_bounds = self.domain_bounds
                margin = self.domain_margin.get()
                
                spparks_bounds = [
                    min_bounds[0] - margin, max_bounds[0] + margin,
                    min_bounds[1] - margin, max_bounds[1] + margin,
                    min_bounds[2] - margin, max_bounds[2] + margin
                ]
                
                spparks_box = pv.Box(bounds=spparks_bounds)
                plotter.add_mesh(spparks_box, style='wireframe', color='red', 
                               line_width=3, opacity=0.8, label='SPPARKS Domain')
                
                # Add processor subdivision
                nx, ny, nz = self.nx_proc.get(), self.ny_proc.get(), self.nz_proc.get()
                dx = (spparks_bounds[1] - spparks_bounds[0]) / nx
                dy = (spparks_bounds[3] - spparks_bounds[2]) / ny
                dz = (spparks_bounds[5] - spparks_bounds[4]) / nz
                
                # Add processor boundary lines
                for i in range(1, nx):
                    x = spparks_bounds[0] + i * dx
                    line = pv.Line([x, spparks_bounds[2], spparks_bounds[4]], 
                                  [x, spparks_bounds[3], spparks_bounds[5]])
                    plotter.add_mesh(line, color='green', line_width=2)
                
                for j in range(1, ny):
                    y = spparks_bounds[2] + j * dy
                    line = pv.Line([spparks_bounds[0], y, spparks_bounds[4]], 
                                  [spparks_bounds[1], y, spparks_bounds[5]])
                    plotter.add_mesh(line, color='green', line_width=2)
                
                for k in range(1, nz):
                    z = spparks_bounds[4] + k * dz
                    line = pv.Line([spparks_bounds[0], spparks_bounds[2], z], 
                                  [spparks_bounds[1], spparks_bounds[3], z])
                    plotter.add_mesh(line, color='green', line_width=2)
            
            plotter.add_axes()
            plotter.show_grid()
            plotter.add_text(f"Layer {layer_idx} - Chunks: {len(chunk_bboxes)}", 
                           position='upper_left', font_size=12)
            plotter.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating 3D visualization: {str(e)}")
    
    def analyze_load_balance(self):
        """Analyze load balancing for current processor configuration"""
        if not self.reader:
            messagebox.showerror("Error", "Please load an HDF5 file first")
            return
            
        try:
            layer_idx = self.layer_idx.get()
            nx, ny, nz = self.nx_proc.get(), self.ny_proc.get(), self.nz_proc.get()
            
            # Define SPPARKS domain
            min_bounds, max_bounds = self.domain_bounds
            margin = self.domain_margin.get()
            spparks_bounds = np.array([
                min_bounds[0] - margin, min_bounds[1] - margin, min_bounds[2] - margin,
                max_bounds[0] + margin, max_bounds[1] + margin, max_bounds[2] + margin
            ])
            
            # Analyze load balancing
            load_df = self.analyzer.analyze_load_balancing(
                spparks_bounds, (nx, ny, nz), layer_idx)
            
            self.log_result(f"\n--- Load Balance Analysis ({nx}×{ny}×{nz} processors) ---")
            self.log_result(f"Total processors: {len(load_df)}")
            
            if len(load_df) > 0:
                self.log_result(f"Chunks per processor:")
                self.log_result(f"  Mean: {load_df['num_chunks'].mean():.1f}")
                self.log_result(f"  Std:  {load_df['num_chunks'].std():.1f}")
                self.log_result(f"  Min:  {load_df['num_chunks'].min()}")
                self.log_result(f"  Max:  {load_df['num_chunks'].max()}")
                
                self.log_result(f"Load imbalance factor: {load_df['chunks_normalized'].std():.3f}")
                
                # Show worst load imbalanced processors
                worst_procs = load_df.nlargest(3, 'num_chunks')[['processor_id', 'i', 'j', 'k', 'num_chunks']]
                self.log_result(f"Most loaded processors:")
                for _, row in worst_procs.iterrows():
                    self.log_result(f"  Proc {row['processor_id']} ({row['i']},{row['j']},{row['k']}): {row['num_chunks']} chunks")
                
                best_procs = load_df.nsmallest(3, 'num_chunks')[['processor_id', 'i', 'j', 'k', 'num_chunks']]
                self.log_result(f"Least loaded processors:")
                for _, row in best_procs.iterrows():
                    self.log_result(f"  Proc {row['processor_id']} ({row['i']},{row['j']},{row['k']}): {row['num_chunks']} chunks")
            
            # Try to recommend better decomposition
            for target_procs in [4, 8, 16, 32]:
                if target_procs != nx * ny * nz:
                    recommendation = self.analyzer.recommend_decomposition(
                        spparks_bounds, target_procs, layer_idx)
                    if recommendation:
                        rec_nx, rec_ny, rec_nz = recommendation['processor_counts']
                        self.log_result(f"\nRecommended {target_procs}-proc decomposition: {rec_nx}×{rec_ny}×{rec_nz}")
                        self.log_result(f"  Load imbalance: {recommendation['load_imbalance']:.3f}")
                        break
                        
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing load balance: {str(e)}")
    
    def export_config(self):
        """Export SPPARKS configuration file"""
        if not self.reader or not self.domain_bounds:
            messagebox.showerror("Error", "Please load and analyze domain first")
            return
            
        try:
            # Define SPPARKS domain
            min_bounds, max_bounds = self.domain_bounds
            margin = self.domain_margin.get()
            spparks_bounds = np.array([
                min_bounds[0] - margin, min_bounds[1] - margin, min_bounds[2] - margin,
                max_bounds[0] + margin, max_bounds[1] + margin, max_bounds[2] + margin
            ])
            
            nx, ny, nz = self.nx_proc.get(), self.ny_proc.get(), self.nz_proc.get()
            lattice_spacing = self.lattice_spacing.get()
            
            # Generate configuration
            config = self.analyzer.generate_spparks_config(
                spparks_bounds, (nx, ny, nz), lattice_spacing)
            
            # Save to file
            output_file = filedialog.asksaveasfilename(
                title="Save SPPARKS configuration",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(config)
                self.log_result(f"\nExported SPPARKS configuration to: {output_file}")
                messagebox.showinfo("Success", f"Configuration saved to: {output_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting configuration: {str(e)}")
    
    def log_result(self, message):
        """Log a message to the results text area"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def run(self):
        """Run the interactive explorer"""
        try:
            self.root.mainloop()
        finally:
            if self.reader:
                self.reader.close()


def main():
    """Main function to run the interactive explorer"""
    
    print("=== Interactive HDF5 Domain Explorer for SPPARKS ===")
    
    if not ADVANCED_FEATURES:
        print("Note: Advanced 3D visualization requires PyVista")
        print("Install with: pip install pyvista")
        print()
    
    # Check if file provided as command line argument
    if len(sys.argv) > 1:
        hdf5_file = sys.argv[1]
        if not Path(hdf5_file).exists():
            print(f"Error: File not found: {hdf5_file}")
            return
    else:
        hdf5_file = None
    
    # Create and run interactive explorer
    explorer = InteractiveExplorer()
    
    # Auto-load file if provided
    if hdf5_file:
        explorer.hdf5_file.set(hdf5_file)
        explorer.root.after(100, explorer.load_file)  # Load after GUI is ready
    
    explorer.run()


if __name__ == "__main__":
    main()