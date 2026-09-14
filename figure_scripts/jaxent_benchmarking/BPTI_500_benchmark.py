#!/usr/bin/env python
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/home/alexi/Documents/ValDX")
from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings


def pre_process_main_BPTI():
    """Get paths for BPTI data."""
    # BPTI data
    expt_name = "Experimental"
    test_name = "BPTI_af_rank1"

    BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
    expt_dir = os.path.join(BPTI_dir, "BPTI_expt_data")

    # Paths to experimental data
    segs_name = "BPTI_residue_segs_trimmed.txt"
    segs_path = os.path.join(expt_dir, segs_name)

    hdx_name = "BPTI_expt_dfracs_clean_trimmed.dat"
    hdx_path = os.path.join(expt_dir, hdx_name)

    rates_name = "BPTI_Intrinsic_rates.dat"
    rates_path = os.path.join(expt_dir, rates_name)

    # Path to trajectory and topology
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"

    top_paths = [topology_path]
    traj_paths = [trajectory_path]
    test_names = ["BPTI_sampled_500"]
    sim_name = "BPTI_MD"

    return hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names


def pre_featurize_BPTI(hdx_path, segs_path, top_path, traj_path, out_dir=None, n_reps=5):
    """Pre-featurize BPTI data for HDX prediction."""
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "BPTI_500_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Pre-featurizing data for {n_reps} replicates...")

    # Set up settings
    settings = Settings(name="BPTI_500_featurize")
    settings.replicates = n_reps
    settings.split_mode = "r"  # Use random splitting
    settings.times = [0.167, 1, 10]

    # Create ValDXer instance
    VDX = ValDXer(settings)
    VDX.load_HDX_data(
        HDX_path=hdx_path, SEG_path=segs_path, calc_name="Experimental", experimental=True
    )
    VDX.load_structures(top_path=top_path, traj_paths=[traj_path], calc_name="BPTI_500")

    # Featurize the data for all replicates
    start_time = time.time()
    dfs, rep_names, predictHDX_dirs, out_prefixes = VDX.featurise_HDX_all_reps(
        calc_name="BPTI_500", n_reps=n_reps, train=True
    )
    featurize_time = time.time() - start_time
    print(f"Featurization time: {featurize_time:.6f} s")

    featurization_results = {
        "dfs": dfs,
        "rep_names": rep_names,
        "predictHDX_dirs": predictHDX_dirs,
        "out_prefixes": out_prefixes,
        "featurize_time": featurize_time,
    }

    return featurization_results, VDX


def benchmark_maxENT_iterations(
    VDX, featurization_results, hdx_path, segs_path, rates_path, expt_name, gamma_lists=None
):
    """Benchmark the maxENT optimizer for different gamma values."""
    if gamma_lists is None:
        # Default gamma lists for benchmarking at different magnitudes
        gamma_lists = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 10^-1
        ]

    predictHDX_dirs = featurization_results["predictHDX_dirs"]
    n_reps = len(predictHDX_dirs)

    # Dictionary to store results
    results = {}

    for expo_idx, gamma_list in enumerate(gamma_lists):
        exponent = -(expo_idx + 1)  # -1, -2, -3
        results[exponent] = {}

        for gamma in gamma_list:
            print(f"Benchmarking with gamma = {gamma}x10^{exponent}...")

            # For each replicate
            iteration_counts = []
            iteration_times = []
            total_times = []

            for rep in range(1, n_reps + 1):
                predictHDX_dir = predictHDX_dirs[rep - 1]

                # Load intrinsic rates for this replicate
                rep_name = f"train_BPTI_500_{rep}"
                rates_file = os.path.join(predictHDX_dir, "Intrinsic_rates.dat")
                VDX.load_intrinsic_rates(rates_file, calc_name=rep_name)

                # Time the reweighting process
                start_time = time.time()
                opt_gamma, df, cr_bc_bh = VDX.reweight_HDX(
                    expt_name=expt_name,
                    # calc_name="BPTI_500",
                    gamma_range=(
                        int(gamma * 10),
                        int(gamma * 10) + 1,
                    ),  # Narrow range to force specific gamma
                    predictHDX_dir=predictHDX_dir,
                    train=True,
                    rep=rep,
                )
                end_time = time.time()

                total_time = end_time - start_time
                total_times.append(total_time)

                # Get iteration count from the output file
                iter_file = os.path.join(predictHDX_dir, "reweighting_per_iteration_output.dat")
                if os.path.exists(iter_file):
                    with open(iter_file, "r") as f:
                        lines = f.readlines()
                        # Skip header lines and count iterations
                        iteration_count = sum(1 for line in lines if not line.startswith("#"))
                        iteration_counts.append(iteration_count)
                        if iteration_count > 0:
                            iteration_times.append(total_time / iteration_count)

            # Calculate statistics
            mean_total_time = np.mean(total_times)
            std_total_time = np.std(total_times)

            if iteration_counts:
                mean_iteration_count = np.mean(iteration_counts)
                mean_iteration_time = np.mean(iteration_times)
                std_iteration_time = np.std(iteration_times)
            else:
                mean_iteration_count = 0
                mean_iteration_time = 0
                std_iteration_time = 0

            results[exponent][gamma] = {
                "total_times": total_times,
                "mean_total_time": mean_total_time,
                "std_total_time": std_total_time,
                "iteration_counts": iteration_counts,
                "mean_iteration_count": mean_iteration_count,
                "iteration_times": iteration_times,
                "mean_iteration_time": mean_iteration_time,
                "std_iteration_time": std_iteration_time,
            }

            print(f"  Mean total time: {mean_total_time:.6f} s")
            print(f"  Mean iterations: {mean_iteration_count:.2f}")
            print(f"  Mean time per iteration: {mean_iteration_time:.6f} s")

    return results


def plot_iteration_benchmark_results(results, out_dir=None):
    """Plot the benchmark results."""
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "BPTI_500_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    # Plot total time vs gamma for each exponent
    plt.figure(figsize=(12, 8))

    for exponent, exponent_results in results.items():
        gammas = list(exponent_results.keys())
        mean_times = [exponent_results[g]["mean_total_time"] for g in gammas]
        std_times = [exponent_results[g]["std_total_time"] for g in gammas]

        label = f"10^{exponent}"
        plt.errorbar(gammas, mean_times, yerr=std_times, fmt="o-", label=label)

    plt.xlabel("Gamma Coefficient")
    plt.ylabel("Total Time (s)")
    plt.title("MaxENT Total Time vs. Gamma Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "maxENT_total_time.png"))
    plt.close()

    # Plot time per iteration vs gamma for each exponent
    plt.figure(figsize=(12, 8))

    for exponent, exponent_results in results.items():
        gammas = list(exponent_results.keys())
        mean_times = [exponent_results[g]["mean_iteration_time"] for g in gammas]
        std_times = [exponent_results[g]["std_iteration_time"] for g in gammas]

        label = f"10^{exponent}"
        plt.errorbar(gammas, mean_times, yerr=std_times, fmt="o-", label=label)

    plt.xlabel("Gamma Coefficient")
    plt.ylabel("Time per Iteration (s)")
    plt.title("MaxENT Time per Iteration vs. Gamma Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "maxENT_iteration_time.png"))
    plt.close()

    # Plot number of iterations vs gamma for each exponent
    plt.figure(figsize=(12, 8))

    for exponent, exponent_results in results.items():
        gammas = list(exponent_results.keys())
        mean_iters = [exponent_results[g]["mean_iteration_count"] for g in gammas]

        label = f"10^{exponent}"
        plt.plot(gammas, mean_iters, "o-", label=label)

    plt.xlabel("Gamma Coefficient")
    plt.ylabel("Number of Iterations")
    plt.title("MaxENT Iterations vs. Gamma Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "maxENT_iteration_count.png"))
    plt.close()


def benchmark_BPTI_500_full():
    """Full benchmark for BPTI_500."""
    # Set up paths
    hdx_path, segs_path, rates_path, top_paths, traj_paths, sim_name, expt_name, test_names = (
        pre_process_main_BPTI()
    )

    # Use only the first trajectory for benchmarking
    top_path = top_paths[0]
    traj_path = traj_paths[0]

    # Create output directory
    out_dir = os.path.join(os.getcwd(), "BPTI_500_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-featurize the data
    featurization_results, VDX = pre_featurize_BPTI(
        hdx_path, segs_path, top_path, traj_path, out_dir=out_dir, n_reps=5
    )

    # Define gamma values for benchmarking
    gamma_lists = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # 10^-1
        [0.01, 0.02, 0.03, 0.04, 0.05],  # 10^-2
        [0.001, 0.002, 0.003, 0.004, 0.005],  # 10^-3
    ]

    # Benchmark maxENT iteration speed
    iteration_results = benchmark_maxENT_iterations(
        VDX, featurization_results, hdx_path, segs_path, rates_path, expt_name, gamma_lists
    )

    # Plot benchmark results
    plot_iteration_benchmark_results(iteration_results, out_dir=out_dir)

    # Save results
    import pickle

    with open(os.path.join(out_dir, "benchmark_results.pkl"), "wb") as f:
        pickle.dump(
            {
                "featurization_results": featurization_results,
                "iteration_results": iteration_results,
            },
            f,
        )

    print("\nBenchmark completed successfully!")
    print(f"Results saved to: {out_dir}")

    return featurization_results, iteration_results


if __name__ == "__main__":
    start_time = time.time()
    benchmark_BPTI_500_full()
    end_time = time.time()

    total_time = end_time - start_time
    print(f"\nTotal benchmark time: {total_time:.2f} s ({total_time / 60:.2f} min)")
