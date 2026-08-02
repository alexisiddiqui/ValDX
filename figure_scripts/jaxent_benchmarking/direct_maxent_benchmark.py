#!/usr/bin/env python
"""
Script to directly benchmark the MaxEnt reweighting process without using ValDXer.
This circumvents any issues with the ValDXer wrapper by directly using the MaxEnt class.
"""

import sys

sys.path.append("/home/alexi/Documents/ValDX")

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from HDXer.reweighting import MaxEnt


def benchmark_maxent_direct(times, gamma_values=None, exponents=None):
    """
    Benchmark MaxEnt reweighting directly using the HDXer.reweighting.MaxEnt class.

    Parameters:
    -----------
    times : list
        Experimental timepoints
    gamma_values : list of lists
        Lists of gamma coefficient values for each exponent
    exponents : list
        List of exponents (powers of 10) to test
    """
    if gamma_values is None:
        gamma_values = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 10^-1
            [0.01, 0.02, 0.03, 0.04, 0.05],  # 10^-2
            [0.001, 0.002, 0.003, 0.004, 0.005],  # 10^-3
        ]

    if exponents is None:
        exponents = [-1, -2, -3]  # 10^-1, 10^-2, 10^-3

    # Set up paths - use relative paths that exist in the repo
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(data_dir, "bpti_input")
    output_dir = os.path.join(data_dir, "maxent_benchmark_results")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for contacts, hbonds, kint file, and experimental data
    # These will be created directly from the trajectory if they don't exist
    contacts_prefix = "Contacts_chain_0_res_"
    hbonds_prefix = "Hbonds_chain_0_res_"
    kint_file = os.path.join(input_dir, "BPTI_Intrinsic_rates.dat")
    exp_file = os.path.join(input_dir, "BPTI_expt_dfracs.dat")

    # Generate synthetic data for testing if files don't exist
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print("Generating synthetic data for testing...")
        generate_synthetic_data(
            input_dir,
            contacts_prefix,
            hbonds_prefix,
            kint_file,
            exp_file,
            n_residues=53,
            n_frames=500,
        )

    folder_list = [input_dir]

    # Results dictionary
    results = {}

    for idx, exp in enumerate(exponents):
        results[exp] = {}
        gamma_list = gamma_values[idx]

        for gamma_coef in gamma_list:
            gamma = gamma_coef * (10**exp)
            print(f"Benchmarking with gamma = {gamma_coef}x10^{exp}...")

            # Set up MaxEnt object
            maxent_outprefix = os.path.join(output_dir, f"maxent_g{gamma_coef}e{exp}_")

            # Start timing
            start_time = time.time()

            # Create and run MaxEnt
            me = MaxEnt(do_reweight=True, do_params=True)

            try:
                me.run(
                    gamma=gamma,
                    data_folders=folder_list,
                    kint_file=kint_file,
                    exp_file=exp_file,
                    times=times,
                    out_prefix=maxent_outprefix,
                    random_initial=False,
                )
                end_time = time.time()
                total_time = end_time - start_time

                # Read iteration count from output file
                iter_file = f"{maxent_outprefix}per_iteration_output.dat"
                iteration_count = 0
                if os.path.exists(iter_file):
                    with open(iter_file, "r") as f:
                        lines = f.readlines()
                        # Skip header lines and count iterations
                        iteration_count = sum(1 for line in lines if not line.startswith("#"))

                # Store results
                results[exp][gamma_coef] = {
                    "total_time": total_time,
                    "iteration_count": iteration_count,
                    "time_per_iteration": total_time / max(1, iteration_count),
                }

                print(f"  Total time: {total_time:.6f} s")
                print(f"  Iterations: {iteration_count}")
                print(
                    f"  Time per iteration: {results[exp][gamma_coef]['time_per_iteration']:.6f} s"
                )

            except Exception as e:
                print(f"Error running MaxEnt with gamma={gamma}: {str(e)}")
                results[exp][gamma_coef] = {
                    "total_time": None,
                    "iteration_count": None,
                    "time_per_iteration": None,
                    "error": str(e),
                }

    return results, output_dir


def generate_synthetic_data(
    output_dir, contacts_prefix, hbonds_prefix, kint_file, exp_file, n_residues=58, n_frames=500
):
    """Generate synthetic data for MaxEnt benchmarking."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate random contacts and hbonds
    for i in range(n_residues):
        contacts = np.random.random((n_frames,)) * 10  # Random number of contacts
        hbonds = np.random.random((n_frames,)) * 5  # Random number of hydrogen bonds

        np.savetxt(os.path.join(output_dir, f"{contacts_prefix}{i + 1}"), contacts)
        np.savetxt(os.path.join(output_dir, f"{hbonds_prefix}{i + 1}"), hbonds)

    # Generate random intrinsic rates
    kints = np.random.random(n_residues) * 10
    with open(kint_file, "w") as f:
        f.write("# Residue kint\n")
        for i, k in enumerate(kints):
            f.write(f"{i + 1} {k}\n")

    # Generate random experimental dfracs
    times = [0.167, 1, 10]  # Example timepoints
    exp_dfracs = np.random.random((n_residues, len(times))) * 0.8  # Random deuteration fractions

    with open(exp_file, "w") as f:
        f.write("# Experimental deuteration fractions\n")
        f.write(f"# Res {' '.join([str(t) for t in times])}\n")

        for i in range(n_residues):
            line = f"{i + 1} " + " ".join([f"{d:.6f}" for d in exp_dfracs[i]])
            f.write(line + "\n")


def plot_benchmark_results(results, output_dir):
    """Plot the benchmark results."""
    # Plot total time vs gamma for each exponent
    plt.figure(figsize=(12, 8))

    for exponent, exponent_results in results.items():
        gammas = sorted(
            [g for g in exponent_results.keys() if exponent_results[g]["total_time"] is not None]
        )

        if not gammas:
            continue

        times = [exponent_results[g]["total_time"] for g in gammas]

        label = f"10^{exponent}"
        plt.plot(gammas, times, "o-", label=label)

    plt.xlabel("Gamma Coefficient")
    plt.ylabel("Total Time (s)")
    plt.title("MaxENT Total Time vs. Gamma Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "maxENT_direct_total_time.png"))
    plt.close()

    # Plot time per iteration vs gamma for each exponent
    plt.figure(figsize=(12, 8))

    for exponent, exponent_results in results.items():
        gammas = sorted(
            [
                g
                for g in exponent_results.keys()
                if exponent_results[g]["time_per_iteration"] is not None
            ]
        )

        if not gammas:
            continue

        times = [exponent_results[g]["time_per_iteration"] for g in gammas]

        label = f"10^{exponent}"
        plt.plot(gammas, times, "o-", label=label)

    plt.xlabel("Gamma Coefficient")
    plt.ylabel("Time per Iteration (s)")
    plt.title("MaxENT Time per Iteration vs. Gamma Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "maxENT_direct_iteration_time.png"))
    plt.close()

    # Plot number of iterations vs gamma for each exponent
    plt.figure(figsize=(12, 8))

    for exponent, exponent_results in results.items():
        gammas = sorted(
            [
                g
                for g in exponent_results.keys()
                if exponent_results[g]["iteration_count"] is not None
            ]
        )

        if not gammas:
            continue

        iters = [exponent_results[g]["iteration_count"] for g in gammas]

        label = f"10^{exponent}"
        plt.plot(gammas, iters, "o-", label=label)

    plt.xlabel("Gamma Coefficient")
    plt.ylabel("Number of Iterations")
    plt.title("MaxENT Iterations vs. Gamma Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "maxENT_direct_iteration_count.png"))
    plt.close()


if __name__ == "__main__":
    # Experimental timepoints
    times = [0.167, 1, 10]

    # Run benchmark
    start_time = time.time()
    results, output_dir = benchmark_maxent_direct(times)
    end_time = time.time()

    # Plot results
    plot_benchmark_results(results, output_dir)

    total_time = end_time - start_time
    print(f"\nTotal benchmark time: {total_time:.2f} s ({total_time / 60:.2f} min)")

    # Save results to file
    import json

    # Convert results to a serializable format
    serializable_results = {}
    for exp, exp_results in results.items():
        serializable_results[str(exp)] = {}
        for gamma, gamma_results in exp_results.items():
            serializable_results[str(exp)][str(gamma)] = {
                k: v for k, v in gamma_results.items() if k != "error" or v is not None
            }

    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)
