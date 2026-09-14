import MDAnalysis as mda
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde
from scipy.spatial.distance import pdist
import os
import sys

sys.path.append("/home/alexi/Documents/ValDX/")
from ValDX.VDX_dataclasses import Segments

def calculate_wasserstein_distance_pairwise(universe1, universe2, selection, residue_numbers):
    # Create atom selections for both universes
    sel1 = universe1.select_atoms(selection + f" and resid {' '.join(map(str, residue_numbers))}")
    sel2 = universe2.select_atoms(selection + f" and resid {' '.join(map(str, residue_numbers))}")
    
    # Check if selections are valid
    if len(sel1) == 0 or len(sel2) == 0:
        raise ValueError("Invalid selection or residue numbers")
    
    # Get pairwise distances for all frames
    distances1 = []
    for ts in sel1.universe.trajectory:
        pairwise_distances = pdist(sel1.positions)
        distances1.extend(pairwise_distances)
    
    distances2 = []
    for ts in sel2.universe.trajectory:
        pairwise_distances = pdist(sel2.positions)
        distances2.extend(pairwise_distances)
    
    # Calculate MSD and MAD
    msd = np.mean((np.mean(distances1) - np.mean(distances2))**2)
    mad = np.mean(np.abs(np.mean(distances1) - np.mean(distances2)))
    
    print(f"Original distances - MSD: {msd:.4f}, MAD: {mad:.4f}")
    
    # Calculate Wasserstein distance on pairwise distances
    w_distance = wasserstein_distance(distances1, distances2)
    return w_distance

def calculate_wasserstein_distance_pairwise_kde(universe1, universe2, selection, residue_numbers, kde_bw_method='scott'):
    # Create atom selections for both universes
    sel1 = universe1.select_atoms(selection + f" and resid {' '.join(map(str, residue_numbers))}")
    sel2 = universe2.select_atoms(selection + f" and resid {' '.join(map(str, residue_numbers))}")
    
    # Check if selections are valid
    if len(sel1) == 0 or len(sel2) == 0:
        raise ValueError("Invalid selection or residue numbers")
    
    # Get pairwise distances for all frames
    distances1 = []
    for ts in sel1.universe.trajectory:
        pairwise_distances = pdist(sel1.positions)
        distances1.extend(pairwise_distances)
    
    distances2 = []
    for ts in sel2.universe.trajectory:
        pairwise_distances = pdist(sel2.positions)
        distances2.extend(pairwise_distances)
    
    # Calculate MSD and MAD for original distances
    msd_orig = np.mean((np.mean(distances1) - np.mean(distances2))**2)
    mad_orig = np.mean(np.abs(np.mean(distances1) - np.mean(distances2)))
    
    print(f"Original distances - MSD: {msd_orig:.4f}, MAD: {mad_orig:.4f}")
    
    # Calculate KDE for both distance sets
    kde1 = gaussian_kde(distances1, bw_method=kde_bw_method)
    kde2 = gaussian_kde(distances2, bw_method=kde_bw_method)
    
    # Create a common grid for evaluation
    min_dist = min(min(distances1), min(distances2))
    max_dist = max(max(distances1), max(distances2))
    grid = np.linspace(min_dist, max_dist, 1000)
    
    # Evaluate KDEs on the grid
    pdf1 = kde1(grid)
    pdf2 = kde2(grid)
    
    # Calculate MSD and MAD for KDE-smoothed distributions
    msd_kde = np.mean((pdf1 - pdf2)**2)
    mad_kde = np.mean(np.abs(pdf1 - pdf2))
    
    print(f"KDE-smoothed distributions - MSD: {msd_kde:.8f}, MAD: {mad_kde:.4f}")
    
    # Calculate Wasserstein distance on KDE estimates
    w_distance = wasserstein_distance(grid, grid, pdf1, pdf2)
    return w_distance

if __name__ == "__main__":
    # Load the trajectories
    clean_af_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.pdb"
    clean_af_traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.xtc"][0]
    shaw_top_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/bpti.pdb"
    shaw_traj_paths = ["/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/SHAW/reduced_BPTI_SHAW_stride_400.xtc"][0]
    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    expt_dir = os.path.join(BPTI_dir, "BPTI_expt_data")
    segs_name = "BPTI_residue_segs.txt"
    segs_path = os.path.join(expt_dir, segs_name)
    segments = Segments(segs_path=segs_path)
    residues = segments.residues

    # Test with PDB only
    u1 = mda.Universe(clean_af_top_path)
    u2 = mda.Universe(shaw_top_path)
    print("PDB only:")
    distance_pdb = calculate_wasserstein_distance_pairwise(u1, u2, "name CA", residues)
    print(f"Wasserstein distance (pairwise): {distance_pdb:.4f}")
    distance_pdb_kde = calculate_wasserstein_distance_pairwise_kde(u1, u2, "name CA", residues)
    print(f"Wasserstein distance (pairwise KDE): {distance_pdb_kde:.4f}")

    print("\nPDB and XTC:")
    # Test with PDB and XTC
    u1 = mda.Universe(clean_af_top_path, clean_af_traj_paths)
    u2 = mda.Universe(shaw_top_path, shaw_traj_paths)
    distance_xtc = calculate_wasserstein_distance_pairwise(u1, u2, "name CA", residues)
    print(f"Wasserstein distance (pairwise): {distance_xtc:.4f}")
    distance_xtc_kde = calculate_wasserstein_distance_pairwise_kde(u1, u2, "name CA", residues)
    print(f"Wasserstein distance (pairwise KDE): {distance_xtc_kde:.4f}")