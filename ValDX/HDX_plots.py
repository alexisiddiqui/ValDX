# Plotting functions - make them universal so other scripts can use them
import matplotlib
# matplotlib.use('TkAgg')  # Example for using the TkAgg backend

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np  
import pandas as pd
import os
import math
import MDAnalysis as mda
from sklearn.metrics import mean_squared_error
from icecream import ic
import time
import datetime

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

    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_dfracs.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)

def plot_peptide_redundancy(top: mda.Universe, segs: pd.DataFrame, save=False, save_dir=None):
    """Plot peptide redundancy.

    Plots the number of times each residue appears in the peptide segments.
    """
    residues = segs.copy()
    # ic(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # ic(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    ic(all_resis)

    # must ensure that the topology numbers match EXACTLY with the experimental data

    # get residue numbers from mda
    resnums = [resi.resid for resi in top.residues]

    # for all resiudes in top, find number of times it appears in all_resis
    resi_counts = []
    for resi in resnums:
        resi_counts.append(all_resis.count(resi))
    ic(resi_counts)

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


    if save is True and save_dir is not None:
        save_name = "peptide_redundancy.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_heatmap_compare(args: list, data: pd.DataFrame, top: mda.Universe, segs: pd.DataFrame, times: list):

    residues = segs
    # ic(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # ic(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # ic(all_resis)
    expt_resis = set(all_resis)
    # must ensure that the topology numbers match EXACTLY with the experimental data
    resnums = [resi.resid for resi in top.residues]


    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'time'])
    for arg in args:
        ic(arg)

        df = data[arg].copy()
        df["resis"] = residues["resis"]
        df = df.explode("resis")
        df = df.groupby("resis").mean().reset_index()
        df["Residue"] = df.resis
        df = df.drop(columns=["resis"])

        # convert to long format
        df = df.melt(id_vars=["Residue"], var_name="time", value_name=arg)
        # ic(df)
        plot_df = pd.concat([plot_df, df], ignore_index=True, axis=1)
            
        ic(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    missing_df = pd.DataFrame([(residue, time) for residue in missing_resis for time in times], columns=['Residue', 'time'])

    plot_df = pd.concat([plot_df, missing_df])

    ic(plot_df)

    # break

    fig, axes = plt.subplots(1, len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        ic(arg)
        ax = axes[i]
        # ic(ax)
        # ic(plot_df)
        data = plot_df.pivot(index="time", columns="Residue", values=arg)
        sns.heatmap(data, ax=ax, cmap='crest')

        # Overlay for NaN values
        nan_mask = data.isna()
        sns.heatmap(nan_mask, ax=ax, mask=~nan_mask, cmap=['#808080'], cbar=False)
        
        ax.set_title(arg)
        ax.set_xlabel('Residue Number')
        ax.set_ylabel('time (min)')
        ax.set_xticks(resnums)

    plt.suptitle('BPTI HDX deuterated fractions heatmap', fontsize=22)
    plt.tight_layout()


    plt.show()
    plt.close()
        


def plot_heatmap_errors(args: list, data: pd.DataFrame, top: mda.Universe, segs: pd.DataFrame, times: list, expt_index=0):

    residues = segs
    # ic(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # ic(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # ic(all_resis)
    expt_resis = set(all_resis)
    # must ensure that the topology numbers match EXACTLY with the experimental data
    resnums = [resi.resid for resi in top.residues]

    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'time'])
    expt = data[args[expt_index]].copy()
    expt["resis"] = residues["resis"]
    expt = expt.explode("resis")
    expt = expt.groupby("resis").mean().reset_index()
    expt["Residue"] = expt.resis
    expt = expt.drop(columns=["resis"])

    # convert to long format
    expt = expt.melt(id_vars=["Residue"], var_name="time", value_name='expt')

    args = [a for a in args if a != args[expt_index]]

    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'time'])
    for arg in args:
        ic(arg)

        ic(arg)
        df = data[arg].copy
        df["resis"] = residues["resis"]
        df = df.explode("resis")
        df = df.groupby("resis").mean().reset_index()
        df["Residue"] = df.resis
        df = df.drop(columns=["resis"])

        # convert to long format
        df = df.melt(id_vars=["Residue"], var_name="time", value_name=arg)

        # subtract expt from df
        df[arg] = expt['expt'] - df[arg]

        # ic(df)
        plot_df = pd.merge(plot_df, df)
            
        ic(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    missing_df = pd.DataFrame([(residue, time) for residue in missing_resis for time in times], columns=['Residue', 'time'])

    plot_df = pd.concat([plot_df, missing_df])

    ic(plot_df)

    # break


    fig, axes = plt.subplots(1, len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        ic(arg)
        ax = axes[i]
        # ic(ax)
        # ic(plot_df)
        data = plot_df.pivot(index="time", columns="Residue", values=arg)
        sns.heatmap(data, ax=ax, cmap='vlag')

        # Overlay for NaN values
        nan_mask = data.isna()
        sns.heatmap(nan_mask, ax=ax, mask=~nan_mask, cmap=['#808080'], cbar=False)
        ax.set_title(arg)
        ax.set_xlabel('Residue Number')
        ax.set_ylabel('time (min)')
        ax.set_xticks(resnums)

    plt.suptitle('BPTI HDX deuterated fractions deviation heatmap', fontsize=22)
    plt.tight_layout()

    plt.show()
    plt.close()

def plot_peptide_dfracs(args: list, data: pd.DataFrame, times: list, top: mda.Universe, segs: pd.DataFrame, save=False, save_dir=None):
        
    residues = segs
    # ic(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # ic(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # ic(all_resis)
    expt_resis = set(all_resis)
    # must ensure that the topology numbers match EXACTLY with the experimental data
    resnums = [resi.resid for resi in top.residues]

    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'time'])
    for arg in args:
        ic(arg)

        ic(arg)
        df = data[arg].copy()
        df["Peptide"] = df.index
        df["resis"] = residues["resis"]
        df = df.explode("resis")
        # ic(df)
        # df = df.groupby("resis").mean().reset_index()
        df["Residue"] = df.resis
        df = df.drop(columns=["resis"])

        # convert to long format
        df = df.melt(id_vars=["Residue","Peptide"], var_name="time", value_name=arg)
        # ic(df)

        plot_df= pd.merge(plot_df, df)
            
        ic(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    pep_nos = plot_df.Peptide.unique()
    missing_df = pd.DataFrame([(residue, time, p, np.nan) for residue in missing_resis for time in times for p in pep_nos], columns=['Residue', 'time', 'Peptide', 'nan'])

    plot_df = pd.concat([plot_df, missing_df])

    ic(missing_df.values)

    cmap = ListedColormap(['#808080', 'none'])



    fig, axes = plt.subplots(len(times), len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        for j, t in enumerate(times):

            ax = axes[j, i]

            ic(arg)
                # ic(ax)
            # ic(plot_df)
            data = plot_df[plot_df['time'] == t].pivot(index="Peptide", columns="Residue", values=arg)
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




    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_peptide_dfracs.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
        plt.close()



def plot_peptide_dfracs_errors(args: list, data: pd.DataFrame, top: mda.Universe, times: list, segs: pd.DataFrame, save=False, save_dir=None, expt_index=0):
        
    residues = segs
    # ic(residues)
    # convert residue start and end to list of residues contained
    residues["resis"] = residues.apply(lambda x: list(range(x.ResStr, x.ResEnd+1)), axis=1)
    # ic(residues)
    all_resis = []
    for resis in residues.resis:
        all_resis.extend(resis)

    # ic(all_resis)
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
    expt = expt.melt(id_vars=["Residue","Peptide"], var_name="time", value_name='expt')
    ic(expt)
    expt_arg = args[expt_index]
    args = [a for a in args if a != args[expt_index]]

    # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
    # plot df contains the product of all possible combinations of residue and time
    plot_df = pd.DataFrame([(residue, time) for residue in expt_resis for time in times], columns=['Residue', 'time'])
    for arg in args:
        ic(arg)

        if arg in ['single', 'pred', 'average', 'average_closest', 'reweighted']:
            ic(arg)
            df = data[arg].copy()
            df["Peptide"] = df.index
            df["resis"] = residues["resis"]
            df = df.explode("resis")
            # ic(df)
            # df = df.groupby("resis").mean().reset_index()
            df["Residue"] = df.resis
            df = df.drop(columns=["resis"])

            # convert to long format
            df = df.melt(id_vars=["Residue","Peptide"], var_name="time", value_name=arg)
            # ic(df)
            # ic(df[arg])
            df[arg] = expt[expt_arg] - df[arg]
            plot_df= pd.merge(plot_df, df)
            
        # ic(plot_df)

    # find missing residues
    missing_resis = set(resnums) - expt_resis

    pep_nos = plot_df.Peptide.unique()
    missing_df = pd.DataFrame([(residue, time, p, np.nan) for residue in missing_resis for time in times for p in pep_nos], columns=['Residue', 'time', 'Peptide', 'nan'])

    plot_df = pd.concat([plot_df, missing_df])

    # ic(missing_df.values)

    cmap = ListedColormap(['#808080', 'none'])



    fig, axes = plt.subplots(len(times), len(args), figsize=(12*len(args), 12))

    for i, arg in enumerate(args):
        for j, t in enumerate(times):

            ax = axes[j, i]

            ic(arg)
                # ic(ax)
            # ic(plot_df)
            data = plot_df[plot_df['time'] == t].pivot(index="Peptide", columns="Residue", values=arg)
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


    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_peptide_dfracs_errors.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
        plt.close()





def plot_dfracs_error(args: list, data: pd.DataFrame, RMSF: list or np.ndarray, top: mda.Universe, times: list, segs: pd.DataFrame, save=False, save_dir=None, expt_index: int=0, abs=False):
        
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fract
            
        data (pd.DataFrame): dataframe containing data to plot using dfs from helpful_funcs.py
        RMSF (list or np.ndarray): RMSF of each residue from MD simulation
        top (mda.Universe): topology of MD simulation
        times (list): list of times to plot
        segs (pd.DataFrame): dataframe containing peptide segment information
        save (bool): whether to save the figure
        save_dir (str): directory to save the figure in
        expt_index (int): index of expt in args
        abs (bool): whether to plot absolute error or not
    """
    if 'expt' == args:
        raise ValueError('expt must be included in args')
    
    expt = data[args[expt_index]].copy()

    args = [arg for arg in args if arg != args[expt_index]]
# calculate standard deviation of each residue from xtal structure bfactors

    # residues = segs
    # ic(residues)

    #create set of residue numbers from resstr 
    # resnums = {residues.iloc[i, 0] for i in range(residues.shape[0])}


    fig, axs = plt.subplots(len(times), 1, figsize=(12, 24))
    for i, (ax, t) in enumerate(zip(axs, times)):
        # plot RMSF
        xs = np.arange(0, expt.iloc[:, 1].shape[0])
        ys = RMSF
        label = 'Scaled RMSF (Abs)'
        # scale to 0-1
        ys = (ys - ys.min())/(ys.max() - ys.min())

        # add -ys to plot as bar chart
        xs = np.concatenate((xs, xs))
        ys = np.concatenate((ys, -ys))

        # plot as bar chart
        ax.bar(xs, ys, label=label, color='gray', alpha=0.3)

        for arg in args:
            # if arg in ['expt', 'pred', 'reweighted']:
            df = data[arg].copy()
            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ys = df.iloc[:, i]
                # calculate absolute difference between ys and expt at each residue
            if abs:
                difference = np.abs(ys - expt.iloc[:, i])
            else:
                difference = ys - expt.iloc[:, i]
            label = arg + ' difference from expt'
            # line, = ax.plot(xs, difference, label=label)
            # line_color = line.get_color()
            ax.axhline(0, color='gray', linestyle='--')  # This line adds a horizontal line at y=0

            ax.plot(xs, difference, label=label)
            ax.set_title(f'Labeling time = {t} min')
            ax.legend(loc='upper right')
            ax.set_xticks(xs)
            ax.set_xlim(xs[0], xs[-1])
            ax.set_xticklabels(segs.iloc[:, 1], rotation=90)
            ax.set_ylim(-1, 1)
            # else:
            #     ic("Incorrect argument given. Please choose one or more of the following: 'expt' 'pred' 'reweighted'")
    fig.text(0.5, 0.095, 'Residue', ha='center', fontsize=22)
    fig.text(0.05, 0.5, 'HDX df absolute error from expt', va='center', rotation='vertical', fontsize=22)



    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_dfracs_error.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_dfracs_compare(args: list, data: pd.DataFrame, times: list, save=False, save_dir=None, expt_index: int=0, key: str='calc_name'):   
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fractions
        data (pd.DataFrame): dataframe containing data to plot using dfs from helpful_funcs.py
        times (list): list of times (flaots) to plot
        save (bool): whether to save the figure
        save_dir (str): directory to save the figure in
        expt_index (int): index of expt in args

    """
    ic("plot_dfracs_compare")
    ic(data)
    # expt = data[args[expt_index]].copy()
    expt = data.loc[data[key]==args[expt_index]].copy()
    ic(expt)
    all_diff_data = []
    # plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    expt_means = []  # List to store mean experimental values at each time step
    ic("ARGUMENTS")
    ic(args)
    ic(data.columns)
    ic(data[key].values)
    ic(data)
    for i, t in enumerate(times):
        expt_mean_at_t = np.mean(expt.iloc[:, i])
        expt_means.append(expt_mean_at_t)

        for arg in args:
            ic(arg)
            df = data.loc[data[key]==arg].copy()
            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ys = df.iloc[:, i].to_list()
            peptides = df['peptide'].values.astype(int)
            ic(*peptides)

            ic(ys)
            exs = expt.iloc[:, i].to_list()
            exs = [exs[pep] for pep in peptides]
            ic(exs)
            # need to account for nan values
            difference = [np.abs(y - ex) for y, ex in zip(ys, exs)]
            ic(difference)
            ic(ys)
            ic(len(ys))
            ic(len(difference))
            ic(len(peptides))
            # Storing differences with corresponding time and argument in the DataFrame
            for j, d in enumerate(difference):
                all_diff_data.append({'time': t, 'difference': d, 'type': arg, 'values': ys[j]})

    # Convert list of dictionaries to DataFrame
    df_differences = pd.DataFrame(all_diff_data).dropna()
    ic(df_differences)
    # Plotting the violin plot
    sns.boxplot(x='time', y='values', hue='type', data=df_differences)
    plt.plot(range(0,len(times)), expt_means, color='black', label='expt mean', linestyle='-', marker='o')

    plt.title('HDX df empirical error from expt')
    plt.xlabel('Labeling time (min)')
    plt.ylabel('HDX Protection Factor')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    
    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_dfracs_compare.png"
        save_path = os.path.join(save_dir, save_name)
        


    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_dfracs_compare.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_dfracs_compare_abs(args: list, data: pd.DataFrame, times: list, save=False, save_dir=None, expt_index: int=0, key: str='calc_name'):   
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fractions
        data (pd.DataFrame): dataframe containing data to plot using dfs from helpful_funcs.py
        times (list): list of times (flaots) to plot
        save (bool): whether to save the figure
        save_dir (str): directory to save the figure in
        expt_index (int): index of expt in args

    """
    ic("plot_dfracs_compare_abs")
    ic(data)
    # expt = data[args[expt_index]].copy()
    expt = data.loc[data[key]==args[expt_index]].copy()
    ic(expt)
    all_diff_data = []
    # plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    expt_means = []  # List to store mean experimental values at each time step
    ic("ARGUMENTS")
    ic(args)
    ic(data.columns)
    ic(data[key].values)
    ic(data)
    for i, t in enumerate(times):
        expt_mean_at_t = np.mean(expt.iloc[:, i])
        expt_means.append(expt_mean_at_t)

        for arg in args:
            ic(arg)
            df = data.loc[data[key]==arg].copy()
            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ys = df.iloc[:, i].to_list()
            peptides = df['peptide'].values.astype(int)
            ic(*peptides)

            ic(ys)
            exs = expt.iloc[:, i].to_list()
            exs = [exs[pep] for pep in peptides]
            ic(exs)
            # need to account for nan values
            difference = [np.abs(y - ex) for y, ex in zip(ys, exs)]

            ic(difference)
            ic(ys)
            ic(len(ys))
            ic(len(difference))
            ic(len(peptides))
            # Storing differences with corresponding time and argument in the DataFrame
            for j, d in enumerate(difference):
                all_diff_data.append({'time': t, 'difference': d, 'type': arg, 'values': ys[j]})

    # Convert list of dictionaries to DataFrame
    df_differences = pd.DataFrame(all_diff_data).dropna()
    ic(df_differences)
    # Plotting the violin plot
    sns.boxplot(x='time', y='difference', hue='type', data=df_differences)
    # plt.plot(range(0,4), expt_means, color='black', label='expt mean', linestyle='-', marker='o')

    plt.title('HDX df abs error from expt')
    plt.xlabel('Labeling time (min)')
    plt.ylabel('HDX Protection Factor')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')


    if save_dir is not None:
        save_name = " ".join(args) + "_plot_dfracs_compare_abs.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()

        plt.close()    



def plot_dfracs_compare_MSE(args: list, data: pd.DataFrame, times: list, save=False, save_dir=None, expt_index: int=0, key: str='calc_name'):   
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fractions
        data (pd.DataFrame): dataframe containing data to plot using dfs from helpful_funcs.py
        times (list): list of times (flaots) to plot
        save (bool): whether to save the figure
        save_dir (str): directory to save the figure in
        expt_index (int): index of expt in args
    """
    ic("plot_dfracs_compare_MSE")
    ic(data)
    # expt = data[args[expt_index]].copy()
    expt = data.loc[data[key]==args[expt_index]].copy()
    ic(expt)
    all_diff_data = []
    # plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    expt_means = []  # List to store mean experimental values at each time step
    ic("ARGUMENTS")
    ic(args)
    ic(data.columns)
    ic(data[key].values)
    ic(data)
    for i, t in enumerate(times):
        expt_mean_at_t = np.mean(expt.iloc[:, i])
        expt_means.append(expt_mean_at_t)

        for arg in args:
            ic(arg)
            df = data.loc[data[key]==arg].copy()
            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ys = df.iloc[:, i].to_list()
            peptides = df['peptide'].values.astype(int)
            ic(*peptides)

            ic(ys)
            exs = expt.iloc[:, i].to_list()
            exs = [exs[pep] for pep in peptides]
            ic(exs)
            # need to account for nan values
            difference = [np.abs(y - ex) for y, ex in zip(ys, exs)]

            difference_sq = [d**2 for d in difference]

            mse = np.nanmean(difference_sq)


            ic(difference)
            ic(ys)
            ic(len(ys))
            ic(len(difference))
            ic(len(peptides))
            # Storing differences with corresponding time and argument in the DataFrame
            if "train" in arg:
                arg_type = "Train"
            elif "val" in arg:
                arg_type = "Val"
            else:
                arg_type = arg

            all_diff_data.append({'time': t, 'mse': mse, 'Type': arg_type, str(key): arg})
    ic(all_diff_data)
    # Convert list of dictionaries to DataFrame
    df_differences = pd.DataFrame(all_diff_data).dropna()
    ic(df_differences)
    # Plotting the violin plot
    sns.boxplot(x='time', y='mse', hue='Type', data=df_differences)
    # plt.plot(range(0,4), expt_means, color='black', label='expt mean', linestyle='-', marker='o')

    plt.title('HDX df mse from expt')
    plt.xlabel('Labeling time (min)')
    plt.ylabel('HDX Protection Factor')
    # plt.ylim(-0.005, 0.005)
    plt.legend(loc='upper right')


    # df_differences = df_differences.drop(columns=['Type'])
    if save and save_dir is not None:
        # save plot
        save_name = " ".join(args) + "_plot_dfracs_compare_MSE.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
        plt.close()

    return df_differences


def plot_dfracs_compare_hist(args: list, data: pd.DataFrame, times: list,  save=False, save_dir=None, expt_index: int=0):
        
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fractions
    """

    expt = data[args[expt_index]].copy()
    all_diff_data = []
    expt_means = []  # List to store mean experimental values at each time step

    for i, t in enumerate(times):
        expt_mean_at_t = np.mean(expt.iloc[:, i])
        expt_means.append(expt_mean_at_t)
        for arg in args:

            df = data[arg].copy()

            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ys = df.iloc[:, i]
            # ic(ys)
            # Calculate absolute difference between ys and expt at each residue
            difference = np.abs(ys - expt.iloc[:, i])
            
            # Storing differences with corresponding time and argument in the DataFrame
            for j, d in enumerate(difference):
                all_diff_data.append({'time': t, 'difference': d, 'type': arg, 'values': ys[j]})

    # Convert list of dictionaries to DataFrame
    df_differences = pd.DataFrame(all_diff_data)

    unique_times = df_differences['time'].unique()
    fig, axes = plt.subplots(nrows=len(unique_times), figsize=(12, 6 * len(unique_times)))
    # If only one time point, make axes iterable
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Iterate over each time and the corresponding axis
    for idx,_ in enumerate(zip(axes, unique_times)):
        ax, t = _
        # Filter data for the current time
        df_time = df_differences[df_differences['time'] == t]
        
        # Create a list to store histogram data for each type
        histograms = []

        for arg in args:
            # Filter data by type (e.g., 'expt', 'pred', 'reweighted')
            type_data = df_time[df_time['type'] == arg]['values']
            
            # Create histogram for the current type and append it to the histograms list
            histograms.append(type_data)

        # Plot histograms on the current axis
        ax.hist(histograms, bins=10, label=args, histtype='bar' ,alpha=0.5)

        ax.axvline(x=expt_means[idx], color='black', label='expt mean', linestyle='-')
        ax.set_title(f'HDX df empirical error from expt at time {t} min')
        ax.set_xlabel('HDX Protection Factor')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')

    plt.tight_layout()


    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_dfracs_compare_hist.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_dfracs_compare_hist_errors(args: list, data: pd.DataFrame, times: list,  save=False, save_dir=None, expt_index: int=0):
        
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fractions
    """

    expt = data[args[expt_index]].copy()

    args = [arg for arg in args if arg != args[expt_index]]

    all_diff_data = []


    expt_means = []  # List to store mean experimental values at each time step

    for i, t in enumerate(times):
        expt_mean_at_t = np.mean(expt.iloc[:, i])
        expt_means.append(expt_mean_at_t)
        for arg in args:

            df = data[arg].copy()

            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ys = df(arg).iloc[:, i]
            # ic(ys)
            # Calculate absolute difference between ys and expt at each residue
            difference = np.abs(ys - expt.iloc[:, i])
            
            # Storing differences with corresponding time and argument in the DataFrame
            for j, d in enumerate(difference):
                all_diff_data.append({'time': t, 'difference': d, 'type': arg, 'values': ys[j]})

    # Convert list of dictionaries to DataFrame
    df_differences = pd.DataFrame(all_diff_data)

    unique_times = df_differences['time'].unique()
    fig, axes = plt.subplots(nrows=len(unique_times), figsize=(12, 6 * len(unique_times)))
    # If only one time point, make axes iterable
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Iterate over each time and the corresponding axis
    for idx,_ in enumerate(zip(axes, unique_times)):
        ax, t = _
        # Filter data for the current time
        df_time = df_differences[df_differences['time'] == t]
        
        # Create a list to store histogram data for each type
        histograms = []

        for arg in args:
            # Filter data by type (e.g., 'expt', 'pred', 'reweighted')
            type_data = df_time[df_time['type'] == arg]['difference']
            
            # Create histogram for the current type and append it to the histograms list
            histograms.append(type_data)

        # Plot histograms on the current axis
        ax.hist(histograms, bins=10, label=args, alpha=0.3)

        # ax.axvline(x=expt_means[idx], color='black', label='expt mean', linestyle='-')
        ax.set_title(f'HDX df empirical error from expt at time {t} min')
        ax.set_xlabel('HDX Protection Factor error')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')

    plt.tight_layout()


    if save is True and save_dir is not None:
        save_name = " ".join(args) + "_plot_dfracs_compare_hist_errors.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_paired_errors(args: list, data: pd.DataFrame, times: list, save=False, save_dir=None, expt_index: int=0, key: str='calc_name'):
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fractions
    """
    # if 'expt' == args:
    #     return ValueError('expt must be included in args')
    ic("plotting paired errors")
    ic(data)
    expt = data.loc[data[key]==args[expt_index]].copy()
    fig, axes = plt.subplots(nrows=len(times), ncols=1, figsize=(8 , 8* (len(times)-1)))
    # give each arg a different matplotlib marker
    markers = ['o', 's', 'v', 'D', 'P', 'X', 'd', 'p', 'h', '8', '1', '2', '3', '4']

    # Assuming times is a predefined list of time points
    for i, t in enumerate(times):
        
        # Extracting experimental data for the current time point
        expt_values = expt.iloc[:, i].copy().to_list()
        ic("expt values")
        ic(expt_values)
        # Creating pairwise plots for the current time
        
        for j, arg in enumerate(args):
            ax = axes[i]
            df = data.loc[data[key]==arg].copy()

            # Extracting data for the current argument
            arg_values = data.loc[data[key]==arg].iloc[:, i].copy()
            peptides = df["peptide"].values.astype(int)

            ic(f"{arg} values")
            ic(arg_values)
            
            # indexes = peptides.values.astype(int)

            ic(peptides)

            arg_expt_values = [expt_values[index]for index in peptides]


            assert len(arg_values) == len(arg_expt_values)
            
            # go over values and remove from both lists if either nan
            R_expt_values = []
            R_arg_values = []

            for expt_value, arg_value in zip(arg_expt_values, arg_values):
                if not np.isnan(expt_value) and not np.isnan(arg_value):
                    R_expt_values.append(expt_value)
                    R_arg_values.append(arg_value)
            ic("Values to compute R values")
            ic(R_expt_values)
            ic(R_arg_values)

            assert len(R_expt_values) == len(R_arg_values)

            # calculate pearson correlation coefficient R^2
            R = np.corrcoef(R_expt_values, R_arg_values)[0,1]
            ic(R)
            
            if arg == args[expt_index]:
                # plot line y=x for reference
                ax.plot(expt_values, expt_values, label='Identity line', linestyle='--', color='grey', alpha=0.1)
                ax.scatter(expt_values, arg_values, label=f'{arg} vs. {args[expt_index]}, R={R:.2f}', color='grey', marker="*")
            else:
                ax.scatter(R_expt_values, R_arg_values, label=f'{arg} vs. {args[expt_index]}, R={R:.2f}', marker=markers[j], alpha=0.5)
            
            # Plotting on the corresponding axis
            ax.set_xlabel('Experimental Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()
            ax.set_title(f'Pairwise comparison at time {t} min')
        
    plt.tight_layout()



    if save and save_dir is not None:
        # save plot
        save_name = " ".join(args) + "_plot_paired_errors.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()
        plt.close()



def plot_lcurve(calc_name, RW_range: tuple, RW_dir: str, prefix: str, gamma: float=None, save=False, save_dir=None):
    li = []
    i, j = RW_range
    # convert gamma (float) to standard form (a*10^b)
    if gamma is not None:
        gamma_exponent = math.floor(math.log10(gamma))
        gamma_coefficient = gamma / 10**gamma_exponent
            

    for i in np.arange(-4, 1): # Select the range of gamma (i in j*10^i)
        for j in np.arange(1, 10): # Select the range of gamma (j in j*10^i)
            # Read files containing work values from the smallest to the biggest gamma
            try:
                work_path = os.path.join(RW_dir, f'{prefix}{j}x10^{i}work.dat')
                df = pd.read_csv(work_path, comment='#', header=None, sep='\s+')
                ic(f"Reading {work_path} ...")
                li.append(df)
            except Exception as e:
                ic(f"Error reading {work_path}: {e}")
    ic(li)
    works = pd.concat(li, axis=0, ignore_index=True) 
    works.columns = ['gamma', 'MSE', 'RMSE', 'work']
    # calculate the value where the tangent of the point is at 45 degrees to the x axis
    # this is the optimal value of gamma
    x = works['MSE'].values.tolist()
    y = works['work'].values.tolist()
    ic("MSE, work", x, y)
    # calculate the the angle made between each point and the next and the x axis

    # if only one value - return this value
    if len(x) == 1:
        return works['gamma'].values[0], works


    ### Optimiser ###
    # TODO change this to a more robust method - change this to find the kink in the curve
    # instead of calcing angle - rotate the curve and find the point where the gradient is 1
    # 
    angles = []
    for i in range(len(x)-1):
        angles.append(np.arctan((y[i+1]-y[i])/(x[i+1]-x[i])))

    # find the index of the angle closest to 45 degrees
    closest = min(angles, key=lambda x:abs(x-math.pi/4))
    ###

    # compute regressionline
    m, b = np.polyfit(x, y, 1)


    dists = []
    for i in range(len(x)):
        regression_coord = (m*x[i] + b, x[i])
        curve_coord = (y[i], x[i])
        # calculate the displacement between the regression line and the curve
        dist = math.dist(curve_coord,regression_coord)
        ic(dist)
        # deternine if the displacement is positive or negative
        if y[i] < (m*x[i] + b):
            dist = dist * -1

        dists.append(dist)

    ic(dists)
    # remove positive values
    dists = [d if d < 0 else 0 for d in dists]
    ic(dists)
    dists= [abs(d) for d in dists]
    ic(dists)
    # find the index of the largest absolute value
    closest = max(dists)

    ic(closest)

    # find the value of gamma at this index
    # closest_gamma = works['gamma'][angles.index(closest)]
    closest_gamma = works['gamma'][dists.index(closest)]
    ic(closest_gamma)

    plt.figure(figsize=(11, 8.5))
    plt.plot(x, m*np.array(x) + b, color='black', linewidth=3, markersize=10)
    plt.plot(x, y, color='teal', linewidth=3, markersize=10, marker='o')

    if gamma is not None:
        gamma_x = works[works['gamma'] == gamma]['MSE'].values[0]
        gamma_y = works[works['gamma'] == gamma]['work'].values[0]
        ic(gamma_x, gamma_y)
        plt.annotate(f"Gamma = {gamma}", xy=(gamma_x, gamma_y), xytext=(gamma_x, gamma_y),
                        arrowprops=dict(facecolor='black', shrink=0.05), size=16 )
    closest_x = works[works['gamma'] == closest_gamma]['MSE'].values[0]
    closest_y = works[works['gamma'] == closest_gamma]['work'].values[0]
    ic("Closest x y: ", closest_x, closest_y)
    plt.annotate(f"Optimal Gamma = {closest_gamma}", xy=(closest_x, closest_y), xytext=(closest_x, closest_y),
                 arrowprops=dict(facecolor='red', shrink=0.05), size=16 )
    title = f'Decision curve for {calc_name}'
    plt.title(title)
    plt.xlabel('MSE to target data')
    plt.ylabel('W$_{app}$ / kJ mol$^{-1}$')



    if (save is True) and (save_dir is not None):
        save_name = f'{calc_name}_decision_plot.png'
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()
        plt.close()


    return closest_gamma, works

def plot_gamma_distribution(train_gammas: list, val_gammas: list, calc_name: str, save=False, save_dir=None):
    plt.figure(figsize=(11, 8.5))
    plt.hist(train_gammas, bins=5, alpha=0.5, label='train')
    plt.hist(val_gammas, bins=5, alpha=0.5, label='val')
    plt.legend(loc='upper right')
    plt.title(f'Gamma distribution for {calc_name}')
    plt.xlabel('Gamma')
    plt.ylabel('Count')
    # plt.xlim(0,10)
    if (save is True) and (save_dir is not None):   
        save_name = f'{calc_name}_gamma_distribution.png'
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()
        plt.close()

    # if save:
    #     plt.savefig(f'{calc_name}_gamma_distribution.pdf', bbox_inches='tight')


# def plot_heatmap_trainval_compare(expt_names: list, 
#                                   expt_segs: pd.DataFrame,
#                                   top: mda.Universe,
#                                   train_names:list,
#                                   val_names:list,
#                                   times: list, 
#                                   data: pd.DataFrame, 
#                                   save:bool=False, save_dir:str=None,
#                                   key:str='calc_name'
#                                   ):
#     # def plot_peptide_dfracs(args: list, data: pd.DataFrame, times: list, top: mda.Universe, segs: pd.DataFrame, save=False, save_dir=None):
        
#     residues = expt_segs
#     ic(residues)
#     peptides = residues['peptide'].to_list()
#     # convert residue start and end to list of residues contained
#     residues["Residue"] = residues.apply(lambda x: tuple(range(x.ResStr, x.ResEnd+1)), axis=1)
#     # # ic(residues)
#     all_resis = []
#     for resis in residues['Residue']:
#         all_resis.extend(resis)

#     # residues["Residue"] = residues['resis']
#     # ic(residues)

#     # residues= residues.drop(columns=["resis"])
#     ic(residues)
#     # ic(all_resis)
#     expt_resis = set(all_resis)
#     # must ensure that the topology numbers match EXACTLY with the experimental data
#     resnums = [resi.resid for resi in top.residues]

#     # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
#     # plot df contains the product of all possible combinations of residue and time
#     args = [expt_names[0], *train_names, *val_names]

#     data = pd.merge(data, residues.drop(columns=[key, "ResStr", "ResEnd"]), on='peptide')
#     ic(data)

#     # plot_df = pd.DataFrame([([residue,residue], time, p) for residue in expt_resis for time in times for p in peptides], columns=['Residue', 'time', 'peptide'])
#     plot_df = pd.DataFrame(columns=["peptide", "Residue" "time"])

#     for arg in args:
#         ic(arg)

#         df = data[data[key]==arg].copy()
#         ic(df)
#         df = df.drop(columns=[key])
#         df = df.explode("Residue")

#         # convert to long format
#         df = df.melt(id_vars=["peptide","Residue"], var_name="time", value_name=arg)
#         ic(df)

#         # plot_df= pd.concat([plot_df, df],ignore_index=True,axis=1)
#         # plot_df= pd.merge(plot_df, df, on=["peptide", "time"])
#         plot_df= pd.concat([plot_df, df], ignore_index=True, keys=["peptide", "time", "Residue"])
# # 
#         ic(plot_df)

#     # merge residue numbers in 

#     # plot_df = pd.merge(plot_df, residues.drop(columns=[key, "ResStr", "ResEnd"]))
#     ic("plotting df")
#     ic(plot_df.to_string())
#     # plot_df = pd.merge(plot_df, residues.drop(columns=[key, "ResStr", "ResEnd"]), on='peptide')
#     # ic(plot_df)

#     # find missing residues
#     missing_resis = set(resnums) - expt_resis

#     # pep_nos = residues["peptide"].to_list()
#     # missing_df = pd.DataFrame([(residue, time, p, np.nan) for residue in missing_resis for time in times for p in pep_nos], columns=['Residue', 'time', 'Peptide', 'nan'])

#     # plot_df = pd.concat([plot_df, missing_df])

#     # ic(missing_df.values)

#     nan_cmap = ListedColormap(['#808080', 'none'])

#     compare_cmaps = ('Greys', 'crest', 'flare')
    

#     args = list(zip(expt_names, train_names, val_names)) # zipped list of replicates to plot together

#     for i, arg in enumerate(args):

#         fig, axes = plt.subplots(len(times), len(arg), figsize=(8*len(arg), 12))

#         for idx, a in enumerate(arg):
#             for j, t in enumerate(times):

#                 ax = axes[j, idx]
#                 # select values for which a is not nan
#                 a_df = plot_df[plot_df[a].notna()]
#                 ic(a_df)
#                 peptides = set(a_df["peptide"].to_list())
#                 ic(peptides)
#                 residues = set(a_df["Residue"].to_list())
#                 ic(residues)
#                 a_df = plot_df[plot_df[a].notna() & (plot_df['time'] == t)]
#                 a_df['Residue'] = a_df['Residue'].astype(int)  # Ensure Residue is int type


#                 ic(arg, a)
#                 # for p in peptides:
#                 #     a_df = a_df[a_df["peptide"] == p]
#                 #     ic(a_df)
#                         # ic(ax)
#                 # ic(plot_df)
#                 # data = a_df[a_df['time'] == t].pivot(index="peptide", columns="Residue", values=a)
#                 heatmap_data = a_df.pivot(index="peptide", columns="Residue", values=a)

#                 ic(heatmap_data)
#                 # data = data.pivot(index="Peptide", columns="Residue", values=arg)
#                 sns.heatmap(heatmap_data, ax=ax, cmap=compare_cmaps[idx], vmin=0, vmax=1)


#                 ax.set_title(f'{a} Df at {t} min')
#                 ax.set_xlabel('Residue Number')
#                 ax.set_ylabel('Peptide Number')
#                 ax.set_xticks(sorted(resnums))

#                 ic(sorted(resnums))
#                 ic(min(resnums), max(resnums))            
#                 ax.set_xlim(min(resnums), max(resnums))
#                 ax.set_yticks(sorted(peptides, reverse=True))

#                 ic(sorted(peptides, reverse=True))
#                 ic(min(peptides), max(peptides))
#                 ax.set_ylim(min(peptides), max(peptides))

#                 ax.set_xticklabels(resnums, rotation=90, fontsize=5)
#                 ax.set_yticklabels(peptides, rotation=0, fontsize=5)

#         plt.suptitle('BPTI HDX deuterated fractions heatmap', fontsize=22)
#         plt.tight_layout()
#         plt.show()



def plot_R_agreement_trainval(expt_name: str, 
                                  expt_segs: pd.DataFrame,
                                  top: mda.Universe,
                                  train_names:list,
                                  val_names:list,
                                  times: list, 
                                  data: pd.DataFrame, 
                                  save:bool=False, save_dir:str=None,
                                  key:str='calc_name'
                                  ):
    
    residues = expt_segs
    peptides = residues['peptide'].to_list()
    # convert residue start and end to list of residues contained
    residues["Residue"] = residues.apply(lambda x: tuple(range(x.ResStr, x.ResEnd+1)), axis=1)
    # # ic(residues)
    all_resis = [resi for sublist in residues['Residue'] for resi in sublist]
    expt_resis = set(all_resis)
    resnums = [resi.resid for resi in top.residues]

    args = [*train_names, *val_names]

    ic("plotting paired trainval agreement")
    ic(data)
    expt = data[data[key]==expt_name].copy()
    df = pd.DataFrame(columns=["time", "R", "calc_name"])

    # Assuming times is a predefined list of time points
    for i, t in enumerate(times):
        
        # Extracting experimental data for the current time point
        expt_values = expt.iloc[:, i].copy().to_list()
        ic("expt values")
        ic(expt_values)
        # Creating pairwise plots for the current time
        
        for j, arg in enumerate(args):

            j_df = data.loc[data[key]==arg].copy()

            # Extracting data for the current argument
            arg_values = data.loc[data[key]==arg].iloc[:, i].copy()
            peptides = j_df["peptide"].values.astype(int)

            ic(f"{arg} values")
            ic(arg_values)
            
            # indexes = peptides.values.astype(int)

            ic(peptides)

            arg_expt_values = [expt_values[index]for index in peptides]


            assert len(arg_values) == len(arg_expt_values)
            
            # go over values and remove from both lists if either nan
            R_expt_values = []
            R_arg_values = []

            for expt_value, arg_value in zip(arg_expt_values, arg_values):
                if not np.isnan(expt_value) and not np.isnan(arg_value):
                    R_expt_values.append(expt_value)
                    R_arg_values.append(arg_value)
            ic("Values to compute R values")
            ic(R_expt_values)
            ic(R_arg_values)

            assert len(R_expt_values) == len(R_arg_values)

            # calculate pearson correlation coefficient R^2
            R = np.corrcoef(R_expt_values, R_arg_values)[0,1]
            ic(R)
            
            df = pd.concat([df, pd.DataFrame([[t, R, arg]], columns=["time", "R", "calc_name"])])

    # plot as box
    ic("df")
    ic(df)
    plot_df = pd.DataFrame(columns=["time", "Type", "R"])

    for t in times:
        for train, val in zip(train_names, val_names):
            ic(train, val)
            ic(t)
            # Extracting R values for the current train and val at time t
            train_R = df.loc[(df["calc_name"] == train) & (df["time"] == t)]
            val_R = df.loc[(df["calc_name"] == val) & (df["time"] == t)]
            train_R = train_R["R"].values
            val_R = val_R["R"].values
            ic(train_R)
            ic(val_R)
            # concat to plot_df
            plot_df = pd.concat([plot_df, pd.DataFrame({"time": t, "Type": "Train", "R": train_R, str(key):train})], ignore_index=True)
            plot_df = pd.concat([plot_df, pd.DataFrame({"time": t, "Type": "Val", "R": val_R, str(key):val})], ignore_index=True)
    ic("plot_df")
    plot_df = plot_df.dropna()
    ic(plot_df)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="time", y="R", hue="Type")
    plt.title("R Value Distributions for Train and Val Over time")
    plt.xlabel("time")
    plt.ylabel("R Value")
    plt.yticks(np.arange(-0.1, 1.05, 0.1))
    plt.ylim(-0.2, 1.1)
    if save is True and save_dir is not None:
        # save plot
        save_name = " ".join(args) + "_plot_R_agreement_trainval.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()


        plt.close()

    # plot_df.drop(columns=["Type"])
    return plot_df

def plot_heatmap_trainval_compare(expt_names: list, 
                                  expt_segs: pd.DataFrame,
                                  top: mda.Universe,
                                  train_names:list,
                                  val_names:list,
                                  times: list, 
                                  data: pd.DataFrame, 
                                  save:bool=False, save_dir:str=None,
                                  key:str='calc_name'
                                  ):
    residues = expt_segs
    peptides = residues['peptide'].to_list()
    residues["Residue"] = residues.apply(lambda x: tuple(range(x.ResStr, x.ResEnd+1)), axis=1)
    all_resis = [resi for sublist in residues['Residue'] for resi in sublist]
    expt_resis = set(all_resis)
    resnums = [resi.resid for resi in top.residues]

    args = [expt_names[0], *train_names, *val_names]
    data = pd.merge(data, residues.drop(columns=[key, "ResStr", "ResEnd"]), on='peptide')

    plot_df = pd.DataFrame(columns=["peptide", "Residue", "time"])
    for arg in args:
        df = data[data[key] == arg].copy()
        df = df.drop(columns=[key])
        df = df.explode("Residue")
        df = df.melt(id_vars=["peptide", "Residue"], var_name="time", value_name=arg)
        plot_df = pd.concat([plot_df, df], ignore_index=True)

    missing_resis = set(resnums) - expt_resis
    nan_cmap = ListedColormap(['#808080', 'none'])
    compare_cmaps = ('cividis_r', 'crest', 'flare')
    args = list(zip(expt_names, train_names, val_names))

    fig, axes = plt.subplots(len(train_names)*len(times), len(args[0]), figsize=(8*len(args[0]), 3*len(times)*len(train_names)))
    for i, arg in enumerate(args):
        
        for idx, a in enumerate(arg):
            for j, t in enumerate(times):
                z= (i*len(times)) + j
                ax = axes[z, idx]
                a_df = plot_df[plot_df[a].notna() & (plot_df['time'] == t)]
                #average over peptides
                a_df['Residue'] = a_df.loc[:, 'Residue'].astype(int)
                a_df["peptide"] = a_df.loc[:, "peptide"].astype(int)
                ic(a_df.to_string())
                ic(a_df.peptide.value_counts())
                # Pivot and plot heatmap
            # try:
                # heatmap_data = a_df.pivot(index="peptide", columns="Residue", vaes=a)   
            # except UserWarning:          
                ic("Averaging over peptides and resiudes")
                a_df = a_df.groupby(["peptide", "Residue"]).mean().reset_index()
                heatmap_data = a_df.pivot(index="peptide", columns="Residue", values=a)
            # finally:
                heatmap_data = heatmap_data.reindex(index=peptides, columns=resnums)

                ic(heatmap_data)
                sns.heatmap(heatmap_data, ax=ax, cmap=compare_cmaps[idx], vmin=0, vmax=1)


                # Prepare and plot the overlay
                overlay_data = pd.DataFrame(np.nan, index=heatmap_data.index, columns=heatmap_data.columns)
                for missing in missing_resis:
                    if missing in overlay_data.columns:
                        overlay_data[missing] = 1

                ic(overlay_data)
                sns.heatmap(overlay_data, cmap=nan_cmap, cbar=False, ax=ax)

                # Set numerical ticks
                ax.set_xticks(range(len(resnums)))
                ax.set_xticklabels(resnums, rotation=90, fontsize=5)
                ax.set_yticks(range(len(peptides)))
                ax.set_yticklabels(peptides, rotation=0, fontsize=5)

                ax.set_title(f'{a} Df at {t} min')
                ax.set_xlabel('Residue Number')
                ax.set_ylabel('Peptide Number')

    plt.suptitle('BPTI HDX deuterated fractions heatmap', fontsize=22)
    plt.tight_layout()
    plt.show()


    plt.close()




def plot_heatmap_trainval_compare_error(expt_names: list, 
                                  expt_segs: pd.DataFrame,
                                  top: mda.Universe,
                                  train_names:list,
                                  val_names:list,
                                  times: list, 
                                  data: pd.DataFrame, 
                                  save:bool=False, save_dir:str=None,
                                  key:str='calc_name'
                                  ):
    # def plot_peptide_dfracs(args: list, data: pd.DataFrame, times: list, top: mda.Universe, segs: pd.DataFrame, save=False, save_dir=None):
        
#     residues = expt_segs
#     ic(residues)
#     peptides = residues['peptide'].to_list()
#     # convert residue start and end to list of residues contained
#     residues["Residue"] = residues.apply(lambda x: tuple(range(x.ResStr, x.ResEnd+1)), axis=1)
#     # # ic(residues)
#     all_resis = []
#     for resis in residues['Residue']:
#         all_resis.extend(resis)

#     # residues["Residue"] = residues['resis']
#     # ic(residues)

#     # residues= residues.drop(columns=["resis"])
#     ic(residues)
#     # ic(all_resis)
#     expt_resis = set(all_resis)
#     # must ensure that the topology numbers match EXACTLY with the experimental data
#     resnums = [resi.resid for resi in top.residues]

#     # fig, axs = plt.subplots(1, len(args), figsize=(12*len(args), 24))
#     # plot df contains the product of all possible combinations of residue and time
#     expt_data = data[data[key]==expt_names[0]].copy()
    
#     args = [*train_names, *val_names]

#     data = pd.merge(data, residues.drop(columns=[key, "ResStr", "ResEnd"]), on='peptide')
#     ic(data)

#     # plot_df = pd.DataFrame([([residue,residue], time, p) for residue in expt_resis for time in times for p in peptides], columns=['Residue', 'time', 'peptide'])
#     plot_df = pd.DataFrame(columns=["peptide", "Residue" "time"])

#     for arg in args:
#         ic(arg)

#         df = data[data[key]==arg].copy()
#         ic(df)
#         arg_peptides = df["peptide"].values.astype(int)
#         # calculate absolute difference between ys and expt at each residue
#         for t in times:
#             expt_t = expt_data[t].to_list()
#             df_t = df[t].to_list()

#             expt_t = [expt_t[pep] for pep in arg_peptides]

#             difference = [(ex-y) for ex, y in zip(expt_t, df_t)]

#             ic(difference)
#             # add difference to df
#             df[t] = difference




#         df = df.drop(columns=[key])
#         df = df.explode("Residue")

#         # convert to long format
#         df = df.melt(id_vars=["peptide","Residue"], var_name="time", value_name=arg)
#         ic(df)

#         # plot_df= pd.concat([plot_df, df],ignore_index=True,axis=1)
#         # plot_df= pd.merge(plot_df, df, on=["peptide", "time"])
#         plot_df= pd.concat([plot_df, df], ignore_index=True, keys=["peptide", "time", "Residue"])
# # 
#         ic(plot_df)

#     # merge residue numbers in 

#     # plot_df = pd.merge(plot_df, residues.drop(columns=[key, "ResStr", "ResEnd"]))
#     ic("plotting df")
#     ic(plot_df.to_string())
#     # plot_df = pd.merge(plot_df, residues.drop(columns=[key, "ResStr", "ResEnd"]), on='peptide')
#     # ic(plot_df)

#     # find missing residues
#     missing_resis = set(resnums) - expt_resis

#     # pep_nos = residues["peptide"].to_list()
#     # missing_df = pd.DataFrame([(residue, time, p, np.nan) for residue in missing_resis for time in times for p in pep_nos], columns=['Residue', 'time', 'Peptide', 'nan'])

#     # plot_df = pd.concat([plot_df, missing_df])

#     # ic(missing_df.values)

#     nan_cmap = ListedColormap(['#808080', 'none'])

#     compare_cmaps = ('RdBu', 'PRGn')
    

#     args = list(zip(train_names, val_names)) # zipped list of replicates to plot together

#     for i, arg in enumerate(args):

#         fig, axes = plt.subplots(len(times), len(arg), figsize=(8*len(arg), 12))

#         for idx, a in enumerate(arg):
#             for j, t in enumerate(times):

#                 ax = axes[j, idx]
#                 # select values for which a is not nan
#                 a_df = plot_df[plot_df[a].notna()]
#                 ic(a_df)
                
                
#                 ic(arg, a)
#                     # ic(ax)
#                 # ic(plot_df)
#                 data = a_df[a_df['time'] == t].pivot(index="peptide", columns="Residue", values=a)
#                 ic(data)
#                 # data = data.pivot(index="Peptide", columns="Residue", values=arg)
#                 sns.heatmap(data, ax=ax, cmap=compare_cmaps[idx])

#                 # add grey values for all residues in msising_resis

#                 # Prepare and plot the overlay
#                 overlay_data = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
#                 for missing in missing_resis:
#                     if missing in overlay_data.columns:
#                         overlay_data[missing] = 1
#                 sns.heatmap(overlay_data, cmap=nan_cmap, cbar=False, ax=ax)



#                 ax.set_title(f'{a} Df at {t} min')
#                 ax.set_xlabel('Residue Number')
#                 ax.set_ylabel('Peptide Number')
#                 ax.set_xticks(resnums)

#         plt.suptitle(f'BPTI HDX deuterated fractions error from {expt_names[0]} heatmap', fontsize=22)
#         plt.tight_layout()
#         plt.show()









    residues = expt_segs
    peptides = residues['peptide'].to_list()
    residues["Residue"] = residues.apply(lambda x: tuple(range(x.ResStr, x.ResEnd+1)), axis=1)
    all_resis = [resi for sublist in residues['Residue'] for resi in sublist]
    expt_resis = set(all_resis)
    resnums = [resi.resid for resi in top.residues]
    expt_data = data[data[key]==expt_names[0]].copy()

    args = [expt_names[0], *train_names, *val_names]
    data = pd.merge(data, residues.drop(columns=[key, "ResStr", "ResEnd"]), on='peptide')

    plot_df = pd.DataFrame(columns=["peptide", "Residue", "time"])
    for arg in args:
        df = data[data[key] == arg].copy()

        arg_peptides = df["peptide"].values.astype(int)
        # calculate absolute difference between ys and expt at each residue
        for t in times:
            expt_t = expt_data[t].to_list()
            df_t = df[t].to_list()
            expt_t = [expt_t[pep] for pep in arg_peptides]
            difference = [(ex-y) for ex, y in zip(expt_t, df_t)]
            ic(difference)
            # add difference to df
            df[t] = difference


        df = df.drop(columns=[key])
        df = df.explode("Residue")
        df = df.melt(id_vars=["peptide", "Residue"], var_name="time", value_name=arg)
        plot_df = pd.concat([plot_df, df], ignore_index=True)

    missing_resis = set(resnums) - expt_resis
    nan_cmap = ListedColormap(['#808080', 'none'])
    compare_cmaps = ('RdBu', 'PRGn')
    args = list(zip(train_names, val_names))
    fig, axes = plt.subplots(len(train_names)*len(times), len(args[0]), figsize=(8*len(args[0]), 3*len(times)*len(train_names)))

    for i, arg in enumerate(args):
        # fig, axes = plt.subplots(len(times), len(arg), figsize=(8*len(arg), 12))

        for idx, a in enumerate(arg):
            for j, t in enumerate(times):
                z= (i*len(times)) + j
                ax = axes[z, idx]
                a_df = plot_df[plot_df[a].notna() & (plot_df['time'] == t)]
                a_df['Residue'] = a_df.loc[:, 'Residue'].astype(int)
                a_df["peptide"] = a_df.loc[:, "peptide"].astype(int)
                ic(a_df)
                # Pivot and plot heatmap
            # try:
                # heatmap_data = a_df.pivot(index="peptide", columns="Residue", values=a)   
            # except UserWarning:          
                ic("Averaging over peptides and resiudes")
                a_df = a_df.groupby(["peptide", "Residue"]).mean().reset_index()
                heatmap_data = a_df.pivot(index="peptide", columns="Residue", values=a)
            # finally:
                heatmap_data = heatmap_data.reindex(index=peptides, columns=resnums)

                ic(heatmap_data)
                sns.heatmap(heatmap_data, ax=ax, cmap=compare_cmaps[idx], vmin=-1, center=0, vmax=1)


                # Prepare and plot the overlay
                overlay_data = pd.DataFrame(np.nan, index=heatmap_data.index, columns=heatmap_data.columns)
                for missing in missing_resis:
                    if missing in overlay_data.columns:
                        overlay_data[missing] = 1

                ic(overlay_data)
                sns.heatmap(overlay_data, cmap=nan_cmap, cbar=False, ax=ax)

                # Set numerical ticks
                ax.set_xticks(range(len(resnums)))
                ax.set_xticklabels(resnums, rotation=90, fontsize=5)
                ax.set_yticks(range(len(peptides)))
                ax.set_yticklabels(peptides, rotation=0, fontsize=5)

                ax.set_title(f'{a} Df at {t} min')
                ax.set_xlabel('Residue Number')
                ax.set_ylabel('Peptide Number')

    plt.suptitle('BPTI HDX deuterated fractions heatmap', fontsize=22)
    plt.tight_layout()
    plt.show()
    plt.close()


def split_benchmark_plot_MSE_by_name(df,
                        save=False,
                        save_dir=None):
    ic("plotting benchmark MSE by name")
    # Create a FacetGrid, using 'name' for each subplot
    g = sns.FacetGrid(df, col="name", col_wrap=6, height=4, aspect=1.5)
    g.fig.suptitle('MSE over Time by Type for each Named Split Mode')

    # Create boxplots
    g = g.map(sns.boxplot, "time", "mse", "Type", palette="Set3")

    # Adding some additional options for better visualization
    g.add_legend(title='Type')
    g.set_axis_labels("Time", "MSE")
    g.set_titles("{col_name}")

    # Adjust the arrangement of the plots
    plt.subplots_adjust(top=0.9)
    names = df["name"].unique()
    if save is True and save_dir is not None:
        # save plot
        save_name = " ".join(names) + "_plot_MSE.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_plot_MSE_by_protein_split(df,
                        save=False,
                        save_dir=None):
    ic("plotting benchmark MSE by protein and split")
    # Create a FacetGrid, using 'name' for each subplot
    g = sns.FacetGrid(df, col="protein", col_wrap=3, height=4, aspect=1.5)
    g.fig.suptitle('MSE over Time by Type for each protein and Split Mode')

    # Create boxplots
    g = g.map(sns.boxplot, "dataset", "mse", "split_type", palette="Set3")

    # Adding some additional options for better visualization
    g.add_legend(title='Type')
    g.set_axis_labels("Time", "MSE")
    g.set_titles("{col_name}")

    # Adjust the arrangement of the plots
    plt.subplots_adjust(top=0.9)
    proteins = df["protein"].unique()

    if save is True and save_dir is not None:
        # save plot
        save_name = " ".join(proteins) + "_plot_MSE_protein_split.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_plot_MSE_by_split_protein(df,
                        save=False,
                        save_dir=None):
    ic("plotting benchmark MSE by split and protein")
    # Create a FacetGrid, using 'name' for each subplot
    g = sns.FacetGrid(df, col="split_type", col_wrap=3, height=4, aspect=1.5)
    g.fig.suptitle('MSE over Time by Type for each protein and Split Mode')

    # Create boxplots
    g = g.map(sns.boxplot, "protein", "mse", "dataset", palette="Set3")

    # Adding some additional options for better visualization
    g.add_legend(title='Type')
    g.set_axis_labels("Time", "MSE")
    g.set_titles("{col_name}")

    # Adjust the arrangement of the plots
    plt.subplots_adjust(top=0.9)
    splits = df["split_type"].unique()
    if save is True and save_dir is not None:
        # save plot
        save_name = " ".join(splits) + "_plot_MSE_split_protein.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_plot_MSE_by_split(df,
                                      save=False,
                                      save_dir=None):
    ic("plotting benchmark MSE by split")
    plt.figure(figsize=(10, 6))  # Adjust the size of the figure here

    sns.boxplot(data=df, x="dataset", y="mse", hue="split_type", palette="Set2")

    # Adding some additional options for better visualization
    plt.xlabel("dataset")
    plt.ylabel("MSE")
    plt.title("MSE by Dataset colour by Split Type")
    plt.legend(title="Dataset")

    if save is True and save_dir is not None:
        # save plot
        save_name = "plot_MSE_by_class.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_plot_MSE_by_protein(df,
                                        save=False,
                                        save_dir=None):
    ic("plotting benchmark MSE by protein")
    plt.figure(figsize=(10, 6))  # Adjust the size of the figure here

    sns.boxplot(data=df, x="dataset", y="mse", hue="protein", palette="Set2")

    # Adding some additional options for better visualization
    plt.xlabel("Dataset")
    plt.ylabel("MSE")
    plt.title("MSE by Dataset Coloured by Protein")
    plt.legend(title="Protein")

    if save is True and save_dir is not None:
        # save plot
        save_name = "plot_MSE_by_protein.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_BV_scatterplot(df,
                                   save=False,
                                   save_dir=None):
    ic("plotting benchmark BV scatter")
    # Constants
    bc, bh = 0.35, 2.0

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6)) # figsize can be adjusted as needed

    # First plot on the left
    sns.scatterplot(data=df, x="Bc", y="Bh", hue="protein", palette="Set2", ax=axs[0])
    axs[0].axhline(y=bh, color='grey', linestyle='--')
    axs[0].axvline(x=bc, color='grey', linestyle='--')
    axs[0].set_xlabel("Bc")
    axs[0].set_ylabel("Bh")
    axs[0].set_title("BV Constants by Protein")
    axs[0].legend(title="Protein")

    # Second plot on the right
    sns.scatterplot(data=df, x="Bc", y="Bh", hue="split_type", palette="Set1", ax=axs[1])
    axs[1].axhline(y=bh, color='grey', linestyle='--')
    axs[1].axvline(x=bc, color='grey', linestyle='--')
    axs[1].set_xlabel("Bc")
    axs[1].set_ylabel("Bh")
    axs[1].set_title("BV Constants by Split Type")
    axs[1].legend(title="Split Type")

    # Adjust layout
    plt.tight_layout()
    # [ax.set_yscale('log') for ax in axs]
    # Show plot
    if save is True and save_dir is not None:
            # save plot
            save_name = "plot_BV_scatter.png"
            save_path = os.path.join(save_dir, save_name)
            plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_BV_boxplot_by_protein(df,
                                        save=False,
                                        save_dir=None):
    ic("plotting benchmark BV boxplot by protein")
        # Constants
    bc, bh = 0.35, 2.0

    # Setup figure
    fig, axs = plt.subplots(2, 1, figsize=(20, 12))  # Adjust figsize as needed
    fig.suptitle('BV Constant Distribution for each Protein')

    # Plot Bh
    sns.boxplot(x='protein', y='Bh', hue='protein', data=df, palette="Set3", ax=axs[0])
    axs[0].axhline(y=bh, color='red', linestyle='--')
    axs[0].set_title('Bh by Protein')
    axs[0].set_xlabel('Protein')
    axs[0].set_ylabel('Bh')

    # Plot Bc
    sns.boxplot(x='protein', y='Bc', hue='protein', data=df, palette="Set3", ax=axs[1])
    axs[1].axhline(y=bc, color='blue', linestyle='--')
    axs[1].set_title('Bc by Protein')
    axs[1].set_xlabel('Protein')
    axs[1].set_ylabel('Bc')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rectangle [left, bottom, right, top] as needed

    if save is True and save_dir is not None:
            # save plot
            save_name = "plot_BV_by_protein.png"
            save_path = os.path.join(save_dir, save_name)
            plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_BV_boxplot_by_split_type(df,
                                        save=False,
                                        save_dir=None):
    ic("plotting benchmark BV boxplot by split type")
        # Constants
    bc, bh = 0.35, 2.0

    # Setup figure
    fig, axs = plt.subplots(2, 1, figsize=(20, 12))  # Adjust figsize as needed
    fig.suptitle('BV Constant Distribution for each Protein')

    # Plot Bh
    sns.boxplot(x='split_type', y='Bh', hue='split_type', data=df, palette="Set3", ax=axs[0])
    axs[0].axhline(y=bh, color='red', linestyle='--')
    axs[0].set_title('Bh by Split Mode')
    axs[0].set_xlabel('Split Mode')
    axs[0].set_ylabel('Bh')

    # Plot Bc
    sns.boxplot(x='split_type', y='Bc', hue='split_type', data=df, palette="Set3", ax=axs[1])
    axs[1].axhline(y=bc, color='blue', linestyle='--')
    axs[1].set_title('Bc by Split Mode')
    axs[1].set_xlabel('Split Mode')
    axs[1].set_ylabel('Bc')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rectangle [left, bottom, right, top] as needed

    if save is True and save_dir is not None:
            # save plot
            save_name = "plot_BV_by_split_mode.png"
            save_path = os.path.join(save_dir, save_name)
            plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()

def split_benchmark_BV_boxplot_by_protein_by_split_type(df,
                                        save=False,
                                        save_dir=None):
    ic("plotting benchmark BV boxplot by protein by split type")
        # Constants
    bc, bh = 0.35, 2.0

    # Setup figure
    fig, axs = plt.subplots(2, 1, figsize=(20, 12))  # Adjust figsize as needed
    fig.suptitle('BV Constants by Protein for each Split Mode')

    # Plot Bh
    sns.boxplot(x='protein', y='Bh', hue='split_type', data=df, palette="Set3", ax=axs[0])
    axs[0].axhline(y=bh, color='red', linestyle='--')
    axs[0].set_title('Bh by split_type')
    axs[0].set_xlabel('Protein')
    axs[0].set_ylabel('Bh')

    # Plot Bc
    sns.boxplot(x='protein', y='Bc', hue='split_type', data=df, palette="Set3", ax=axs[1])
    axs[1].axhline(y=bc, color='blue', linestyle='--')
    axs[1].set_title('Bc by split_type')
    axs[1].set_xlabel('Protein')
    axs[1].set_ylabel('Bc')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rectangle [left, bottom, right, top] as needed

    if save is True and save_dir is not None:
        save_name = "plot_BV_by_protein_by_split_mode.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)

    else:
        # Show plot
        plt.show()
        plt.close()


def split_benchmark_BV_boxplot_by_split_type_by_protein(df,
                                        save=False,
                                        save_dir=None):
    ic("plotting benchmark BV boxplot by split type by protein")
        # Constants
    bc, bh = 0.35, 2.0

    # Setup figure
    fig, axs = plt.subplots(2, 1, figsize=(20, 12))  # Adjust figsize as needed
    fig.suptitle('BV Constants by Split for each Protein')

    # Plot Bh
    sns.boxplot(x='split_type', y='Bh', hue='protein', data=df, palette="Set3", ax=axs[0])
    axs[0].axhline(y=bh, color='red', linestyle='--')
    axs[0].set_title('Bh by Protein')
    axs[0].set_xlabel('Split Mode')
    axs[0].set_ylabel('Bh')

    # Plot Bc
    sns.boxplot(x='split_type', y='Bc', hue='protein', data=df, palette="Set3", ax=axs[1])
    axs[1].axhline(y=bc, color='blue', linestyle='--')
    axs[1].set_title('Bc by Protein')
    axs[1].set_xlabel('Split Mode')
    axs[1].set_ylabel('Bc')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rectangle [left, bottom, right, top] as needed

    if save is True and save_dir is not None:
        save_name = "plot_BV_by_split_mode_by_protein.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png', dpi=300)
    else:
        # Show plot
        plt.show()
        plt.close()


def plot_cluster_weights(projected_data, cluster_labels, new_cluster_centers, new_cluster_weights, save, save_dir):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with 1 row and 2 columns for subplots

    # Plot the first scatter plot (PCA of CA atoms from Clustered Trajectory)
    axs[0].set_title("PCA of CA atoms from Trajectory")
    axs[0].scatter(projected_data[:, 0], projected_data[:, 1], c=cluster_labels, s=50, cmap='viridis')

    # Plot the second scatter plot (PCA of Cluster Centers from Clustered Trajectory)
    axs[1].set_title("PCA of Cluster Centers from Clustered Trajectory")
    axs[1].scatter(new_cluster_centers[:, 0], new_cluster_centers[:, 1], c='red', s=100*new_cluster_weights, alpha=0.5)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    if save is True and save_dir is not None:
        time= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name = f"cluster_weights_{time}.png"
        save_path = os.path.join(save_dir, save_name)        
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()
        plt.close()
