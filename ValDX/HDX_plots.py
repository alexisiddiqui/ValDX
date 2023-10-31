# Plotting functions - make them universal so other scripts can use them

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np  
import pandas as pd
import os
import MDAnalysis as mda

from ValDX.helpful_funcs import *

def plot_dfracs(args: list, data: pd.DataFrame, times: list, segs: pd.DataFrame, save=False, save_dir=None):
    """Plot HDX deuterated fractions for each time point.
    
    Args: 
        args (list): list of strings indicating which data to plot. 
            For ex: 'expt', 'pred', 'reweighted'
        data (pd.DataFrame): dataframe containing data to plot using dfs from helpful_funcs.py
        times (list): list of times to plot
        segs (pd.DataFrame): dataframe containing peptide segment information

        # not implemented yet
        save (bool): whether to save the figure
        save_dir (str): directory to save the figure in

    """

    fig, axs = plt.subplots(len(times), 1, figsize=(12, 24))
    for i, (ax, t) in enumerate(zip(axs, times)):
        for arg in args:
            df = data[arg].copy()
            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ax.plot(xs, df.iloc[:, i], label=arg)
            ax.set_title(f'Labeling time = {t} min')
            ax.legend(loc='upper right')
            ax.set_xticks(xs)
            ax.set_xlim(xs[0], xs[-1])
            ax.set_xticklabels(segs.iloc[:, 1], rotation=90)
            ax.set_ylim(0, 1)

    fig.text(0.5, 0.095, 'Residue', ha='center', fontsize=22)
    fig.text(0.05, 0.5, 'HDX deuterated fractions', va='center', rotation='vertical', fontsize=22)

def plot_peptide_redundancy(top: mda.Universe, segs: pd.DataFrame, save=False, save_dir=None):
    """Plot peptide redundancy.

    Plots the number of times each residue appears in the peptide segments.
    """
    residues = segs.copy()
    # print(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # print(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    print(all_resis)

    # must ensure that the topology numbers match EXACTLY with the experimental data

    # get residue numbers from mda
    resnums = [resi.resid for resi in top.residues]

    # for all resiudes in top, find number of times it appears in all_resis
    resi_counts = []
    for resi in resnums:
        resi_counts.append(all_resis.count(resi))
    print(resi_counts)

    # plot bar chart of resi counts
    _, ax = plt.subplots(figsize=(20, 8.5))

    ax.bar(resnums, resi_counts)

    ax.set_title('Expt HDX peptide redundancy')
    ax.set_xlim(resnums[0], resnums[-1])
    # set y ticks to be integers
    ax.set_yticks(np.arange(0, max(resi_counts)+1, 1.0))
    # set x ticks to be integers
    ax.set_xticks(resnums)
    ax.set_xlabel('Residue Number')
    ax.set_ylabel('Residue Frequency')

    plt.show()


def plot_heatmap_compare(args: list, data: pd.DataFrame, top: mda.Universe, segs: pd.DataFrame, times: list):

    residues = segs
    # print(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # print(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # print(all_resis)
    expt_resis = set(all_resis)
    # must ensure that the topology numbers match EXACTLY with the experimental data
    resnums = [resi.resid for resi in top.residues]


    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'Time'])
    for arg in args:
        print(arg)

        df = data[arg].copy()
        df["resis"] = residues["resis"]
        df = df.explode("resis")
        df = df.groupby("resis").mean().reset_index()
        df["Residue"] = df.resis
        df = df.drop(columns=["resis"])

        # convert to long format
        df = df.melt(id_vars=["Residue"], var_name="Time", value_name=arg)
        # print(df)
        plot_df = pd.merge(plot_df, df)
            
        print(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    missing_df = pd.DataFrame([(residue, time) for residue in missing_resis for time in times], columns=['Residue', 'Time'])

    plot_df = pd.concat([plot_df, missing_df])

    print(plot_df)

    # break

    fig, axes = plt.subplots(1, len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        print(arg)
        ax = axes[i]
        # print(ax)
        # print(plot_df)
        data = plot_df.pivot(index="Time", columns="Residue", values=arg)
        sns.heatmap(data, ax=ax, cmap='crest')

        # Overlay for NaN values
        nan_mask = data.isna()
        sns.heatmap(nan_mask, ax=ax, mask=~nan_mask, cmap=['#808080'], cbar=False)
        
        ax.set_title(arg)
        ax.set_xlabel('Residue Number')
        ax.set_ylabel('Time (min)')
        ax.set_xticks(resnums)

    plt.suptitle('BPTI HDX deuterated fractions heatmap', fontsize=22)
    plt.tight_layout()
    plt.show()

        


def plot_heatmap_errors(args: list, data: pd.DataFrame, top: mda.Universe, segs: pd.DataFrame, times: list, expt_index=0):

    residues = segs
    # print(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # print(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # print(all_resis)
    expt_resis = set(all_resis)
    # must ensure that the topology numbers match EXACTLY with the experimental data
    resnums = [resi.resid for resi in top.residues]

    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'Time'])
    expt = data[args[expt_index]].copy()
    expt["resis"] = residues["resis"]
    expt = expt.explode("resis")
    expt = expt.groupby("resis").mean().reset_index()
    expt["Residue"] = expt.resis
    expt = expt.drop(columns=["resis"])

    # convert to long format
    expt = expt.melt(id_vars=["Residue"], var_name="Time", value_name='expt')

    args = [a for a in args if a != args[expt_index]]

    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'Time'])
    for arg in args:
        print(arg)

        print(arg)
        df = data[arg].copy
        df["resis"] = residues["resis"]
        df = df.explode("resis")
        df = df.groupby("resis").mean().reset_index()
        df["Residue"] = df.resis
        df = df.drop(columns=["resis"])

        # convert to long format
        df = df.melt(id_vars=["Residue"], var_name="Time", value_name=arg)

        # subtract expt from df
        df[arg] = expt['expt'] - df[arg]

        # print(df)
        plot_df = pd.merge(plot_df, df)
            
        print(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    missing_df = pd.DataFrame([(residue, time) for residue in missing_resis for time in times], columns=['Residue', 'Time'])

    plot_df = pd.concat([plot_df, missing_df])

    print(plot_df)

    # break


    fig, axes = plt.subplots(1, len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        print(arg)
        ax = axes[i]
        # print(ax)
        # print(plot_df)
        data = plot_df.pivot(index="Time", columns="Residue", values=arg)
        sns.heatmap(data, ax=ax, cmap='vlag')

        # Overlay for NaN values
        nan_mask = data.isna()
        sns.heatmap(nan_mask, ax=ax, mask=~nan_mask, cmap=['#808080'], cbar=False)
        ax.set_title(arg)
        ax.set_xlabel('Residue Number')
        ax.set_ylabel('Time (min)')
        ax.set_xticks(resnums)

    plt.suptitle('BPTI HDX deuterated fractions deviation heatmap', fontsize=22)
    plt.tight_layout()
    plt.show()


def plot_peptide_dfracs(args: list, data: pd.DataFrame, times: list, segs: pd.DataFrame, save=False, save_dir=None):
        
    residues = segs
    # print(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # print(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # print(all_resis)
    expt_resis = set(all_resis)
    # must ensure that the topology numbers match EXACTLY with the experimental data
    resnums = [resi.resid for resi in top.residues]

    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'Time'])
    for arg in args:
        print(arg)

        print(arg)
        df = data[arg].copy()
        df["Peptide"] = df.index
        df["resis"] = residues["resis"]
        df = df.explode("resis")
        # print(df)
        # df = df.groupby("resis").mean().reset_index()
        df["Residue"] = df.resis
        df = df.drop(columns=["resis"])

        # convert to long format
        df = df.melt(id_vars=["Residue","Peptide"], var_name="Time", value_name=arg)
        # print(df)

        plot_df= pd.merge(plot_df, df)
            
        print(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    pep_nos = plot_df.Peptide.unique()
    missing_df = pd.DataFrame([(residue, time, p, np.nan) for residue in missing_resis for time in times for p in pep_nos], columns=['Residue', 'Time', 'Peptide', 'nan'])

    plot_df = pd.concat([plot_df, missing_df])

    print(missing_df.values)

    cmap = ListedColormap(['#808080', 'none'])



    fig, axes = plt.subplots(len(times), len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        for j, t in enumerate(times):

            ax = axes[j, i]

            print(arg)
                # print(ax)
            # print(plot_df)
            data = plot_df[plot_df['Time'] == t].pivot(index="Peptide", columns="Residue", values=arg)
            # data = data.pivot(index="Peptide", columns="Residue", values=arg)
            sns.heatmap(data, ax=ax, cmap='crest')

            # add grey values for all residues in msising_resis

            # Prepare and plot the overlay
            overlay_data = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
            for missing in missing_resis:
                if missing in overlay_data.columns:
                    overlay_data[missing] = 1
            sns.heatmap(overlay_data, cmap=cmap, cbar=False, ax=ax)



            ax.set_title(f'{arg} Df at {t} min')
            ax.set_xlabel('Residue Number')
            ax.set_ylabel('Peptide Number')
            ax.set_xticks(resnums)

    plt.suptitle('BPTI HDX deuterated fractions heatmap', fontsize=22)
    plt.tight_layout()
    plt.show()


def plot_peptide_dfracs_errors(args: list, data: pd.DataFrame, top: mda.Universe, times: list, segs: pd.DataFrame, save=False, save_dir=None, expt_index=0):
        
    residues = segs
    # print(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # print(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # print(all_resis)
    expt_resis = set(all_resis)
    # must ensure that the topology numbers match EXACTLY with the experimental data
    resnums = [resi.resid for resi in top.residues]

    expt = data[args[expt_index]].copy()

    expt["Peptide"] = expt.index

    expt["resis"] = residues["resis"]
    expt = expt.explode("resis")
    expt["Residue"] = expt.resis
    expt = expt.drop(columns=["resis"])

    # convert to long format
    expt = expt.melt(id_vars=["Residue","Peptide"], var_name="Time", value_name='expt')
    print(expt)
    expt_arg = args[expt_index]
    args = [a for a in args if a != args[expt_index]]

    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'Time'])
    for arg in args:
        print(arg)

        if arg in ['single', 'pred', 'average', 'average_closest', 'reweighted']:
            print(arg)
            df = data[arg].copy()
            df["Peptide"] = df.index
            df["resis"] = residues["resis"]
            df = df.explode("resis")
            # print(df)
            # df = df.groupby("resis").mean().reset_index()
            df["Residue"] = df.resis
            df = df.drop(columns=["resis"])

            # convert to long format
            df = df.melt(id_vars=["Residue","Peptide"], var_name="Time", value_name=arg)
            # print(df)
            # print(df[arg])
            df[arg] = expt[expt_arg] - df[arg]
            plot_df= pd.merge(plot_df, df)
            
        # print(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    pep_nos = plot_df.Peptide.unique()
    missing_df = pd.DataFrame([(residue, time, p, np.nan) for residue in missing_resis for time in times for p in pep_nos], columns=['Residue', 'Time', 'Peptide', 'nan'])

    plot_df = pd.concat([plot_df, missing_df])

    # print(missing_df.values)

    cmap = ListedColormap(['#808080', 'none'])



    fig, axes = plt.subplots(len(times), len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        for j, t in enumerate(times):

            ax = axes[j, i]

            print(arg)
                # print(ax)
            # print(plot_df)
            data = plot_df[plot_df['Time'] == t].pivot(index="Peptide", columns="Residue", values=arg)
            # data = data.pivot(index="Peptide", columns="Residue", values=arg)
            sns.heatmap(data, ax=ax, cmap='vlag', center=0, vmin=-1, vmax=1)

            # add grey values for all residues in msising_resis

            # Prepare and plot the overlay
            overlay_data = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
            for missing in missing_resis:
                if missing in overlay_data.columns:
                    overlay_data[missing] = 1
            sns.heatmap(overlay_data, cmap=cmap, cbar=False, ax=ax)

            ax.set_title(f'{arg} Df at {t} min')
            ax.set_xlabel('Residue Number')
            ax.set_ylabel('Peptide Number')
            ax.set_xticks(resnums)

    plt.suptitle('BPTI HDX deuterated fractions emp error heatmap', fontsize=22)
    plt.tight_layout()
    plt.show()

