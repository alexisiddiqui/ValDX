# Plotting functions - make them universal so other scripts can use them

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np  
import pandas as pd
import os
import math
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


def plot_peptide_dfracs(args: list, data: pd.DataFrame, times: list, top: mda.Universe, segs: pd.DataFrame, save=False, save_dir=None):
        
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
    # print(residues)

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
            #     print("Incorrect argument given. Please choose one or more of the following: 'expt' 'pred' 'reweighted'")
    fig.text(0.5, 0.095, 'Residue', ha='center', fontsize=22)
    fig.text(0.05, 0.5, 'HDX df absolute error from expt', va='center', rotation='vertical', fontsize=22)





def plot_dfracs_compare(args: list, data: pd.DataFrame, times: list, save=False, save_dir=None, expt_index: int=0, key: str='calc_name'):   
    """Plot HDX deuterated fractions for each time point.
    
    Args:
        *args: 
            'expt' - experimental HDX deuterated fractions
            'pred' - computed HDX deuterated fractions
            'reweighted' - reweighted HDX deuterated fractions
        data (pd.DataFrame): dataframe containing data to plot using dfs from helpful_funcs.py
        times (list): list of times to plot
        save (bool): whether to save the figure
        save_dir (str): directory to save the figure in
        expt_index (int): index of expt in args

    """
    print("plot_dfracs_compare")
    print(data)
    # expt = data[args[expt_index]].copy()
    expt = data.loc[data[key]==args[expt_index]].copy()
    print(expt)
    all_diff_data = []
    # plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    expt_means = []  # List to store mean experimental values at each time step
    print("ARGUMENTS")
    print(args)
    print(data.columns)
    print(data[key].values)
    print(data)
    for i, t in enumerate(times):
        expt_mean_at_t = np.mean(expt.iloc[:, i])
        expt_means.append(expt_mean_at_t)

        for arg in args:
            print(arg)
            df = data.loc[data[key]==arg].copy()
            xs = np.arange(0, df.iloc[:, 1].shape[0])
            ys = df.iloc[:, i].to_list()
            peptides = df['peptide'].values.astype(int)
            print(*peptides)

            print(ys)
            exs = expt.iloc[:, i].to_list()
            exs = [exs[pep] for pep in peptides]
            print(exs)
            # need to account for nan values
            difference = [np.abs(y - ex) for y, ex in zip(ys, exs)]



            print(difference)
            print(ys)
            print(len(ys))
            print(len(difference))
            print(len(peptides))
            # Storing differences with corresponding time and argument in the DataFrame
            for j, d in enumerate(difference):
                all_diff_data.append({'time': t, 'difference': d, 'type': arg, 'values': ys[j]})

    # Convert list of dictionaries to DataFrame
    df_differences = pd.DataFrame(all_diff_data).dropna()
    print(df_differences)
    # Plotting the violin plot
    sns.boxplot(x='time', y='values', hue='type', data=df_differences)
    plt.plot(range(0,4), expt_means, color='black', label='expt mean', linestyle='-', marker='o')

    plt.title('HDX df empirical error from expt')
    plt.xlabel('Labeling time (min)')
    plt.ylabel('HDX Protection Factor')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.show()
    fig.text(0.5, 0.095, 'Residue', ha='center', fontsize=22)
    fig.text(0.05, 0.5, 'HDX df compare to expt', va='center', rotation='vertical', fontsize=22)




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
            # print(ys)
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
    plt.show()
            #     print("Incorrect argument given. Please choose one or more of the following: 'expt' 'pred' 'reweighted'")
    fig.text(0.5, 0.095, 'Residue', ha='center', fontsize=22)
    fig.text(0.05, 0.5, 'HDX df compare to expt', va='center', rotation='vertical', fontsize=22)





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
            # print(ys)
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
    plt.show()
            #     print("Incorrect argument given. Please choose one or more of the following: 'expt' 'pred' 'reweighted'")
    fig.text(0.5, 0.095, 'Residue', ha='center', fontsize=22)
    fig.text(0.05, 0.5, 'HDX df compare to expt', va='center', rotation='vertical', fontsize=22)



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
    print("plotting paired errors")
    print(data)
    expt = data.loc[data[key]==args[expt_index]].copy()
    fig, axes = plt.subplots(nrows=len(times), ncols=1, figsize=(12 , 6* (len(times)-1)))

    # Assuming times is a predefined list of time points
    for i, t in enumerate(times):
        
        # Extracting experimental data for the current time point
        expt_values = expt.iloc[:, i].copy().to_list()
        print("expt values")
        print(expt_values)
        # Creating pairwise plots for the current time
        
        for j, arg in enumerate(args):
            ax = axes[i]
            df = data.loc[data[key]==arg].copy()

            # Extracting data for the current argument
            arg_values = data.loc[data[key]==arg].iloc[:, i].copy()
            peptides = df["peptide"].values.astype(int)

            print(f"{arg} values")
            print(arg_values)
            
            # indexes = peptides.values.astype(int)

            print(peptides)

            arg_expt_values = [expt_values[index]for index in peptides]


            assert len(arg_values) == len(arg_expt_values)
            
            # go over values and remove from both lists if either nan
            R_expt_values = []
            R_arg_values = []

            for expt_value, arg_value in zip(arg_expt_values, arg_values):
                if not np.isnan(expt_value) and not np.isnan(arg_value):
                    R_expt_values.append(expt_value)
                    R_arg_values.append(arg_value)
            print("Values to compute R values")
            print(R_expt_values)
            print(R_arg_values)

            assert len(R_expt_values) == len(R_arg_values)

            # calculate pearson correlation coefficient R^2
            R = np.corrcoef(R_expt_values, R_arg_values)[0,1]
            print(R)
            
            if arg == args[expt_index]:
                # plot line y=x for reference
                ax.plot(expt_values, expt_values, label='expt', linestyle='--', color='black')
                ax.scatter(arg_values, expt_values, label=f'{arg} vs. {args[expt_index]}, R={R:.2f}', color='grey')
            else:
                ax.scatter(R_arg_values, R_expt_values, label=f'{arg} vs. {args[expt_index]}, R={R:.2f}')
            
            # Plotting on the corresponding axis
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Experimental Values')
            ax.legend()
            ax.set_title(f'Pairwise comparison at time {t} min')
        
    plt.tight_layout()
    plt.show()

    fig.text(0.5, 0.04, 'Experimental Value', ha='center', fontsize=22)
    fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=22)





def plot_lcurve(calc_name, RW_range: tuple, RW_dir: str, prefix: str, gamma: int=None, save=False, save_dir=None):
    li = []
    i, j = RW_range
    for i in np.arange(-3, 1): # Select the range of gamma (i in j*10^i)
        for j in np.arange(1, 10): # Select the range of gamma (j in j*10^i)
            # Read files containing work values from the smallest to the biggest gamma
            try:
                work_path = os.path.join(RW_dir, f'{prefix}{j}x10^{i}work.dat')
                df = pd.read_csv(work_path, comment='#', header=None, sep='\s+')
                print(f"Reading {work_path} ...")
                li.append(df)
            except FileNotFoundError:
                pass
    print(li)
    works = pd.concat(li, axis=0, ignore_index=True) 
    works.columns = ['gamma', 'MSE', 'RMSE', 'work']
    # calculate the value where the tangent of the point is at 45 degrees to the x axis
    # this is the optimal value of gamma
    x = works['MSE'].values.tolist()
    y = works['work'].values.tolist()
    print("MSE, work", x, y)
    # calculate the the angle made between each point and the next and the x axis

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
        print(dist)
        # deternine if the displacement is positive or negative
        if y[i] < (m*x[i] + b):
            dist = dist * -1

        dists.append(dist)

    print(dists)
    # remove positive values
    dists = [d if d < 0 else 0 for d in dists]
    print(dists)
    dists= [abs(d) for d in dists]
    print(dists)
    # find the index of the largest absolute value
    closest = max(dists)

    print(closest)

    # find the value of gamma at this index
    # closest_gamma = works['gamma'][angles.index(closest)]
    closest_gamma = works['gamma'][dists.index(closest)]
    print(closest_gamma)

    plt.figure(figsize=(11, 8.5))
    plt.plot(x, m*np.array(x) + b, color='black', linewidth=3, markersize=10)
    plt.plot(x, y, color='teal', linewidth=3, markersize=10, marker='o')

    if gamma is not None:
        gamma = gamma * 10**-3
        gamma_x = works[works['gamma'] == gamma]['MSE'].values[0]
        gamma_y = works[works['gamma'] == gamma]['work'].values[0]
        print(gamma_x, gamma_y)
        plt.annotate(f"Gamma = {gamma}", xy=(gamma_x, gamma_y), xytext=(gamma_x, gamma_y),
                        arrowprops=dict(facecolor='black', shrink=0.05), size=16 )
    closest_x = works[works['gamma'] == closest_gamma]['MSE'].values[0]
    closest_y = works[works['gamma'] == closest_gamma]['work'].values[0]
    print("Closest x y: ", closest_x, closest_y)
    plt.annotate(f"Optimal Gamma = {closest_gamma}", xy=(closest_x, closest_y), xytext=(closest_x, closest_y),
                 arrowprops=dict(facecolor='red', shrink=0.05), size=16 )
    title = f'Decision curve for {calc_name}'
    plt.title(title)
    plt.xlabel('MSE to target data')
    plt.ylabel('W$_{app}$ / kJ mol$^{-1}$')
    plt.show()
    # if save:
    #     plt.savefig(f'{calc_name}_decision_plot.pdf', bbox_inches='tight')

    # TODO convert closest gamma float to integer exponent and coefficient

    # for now just multiply by 10^3
    # 
    closest_gamma = closest_gamma * 10**3    
    

    return int(closest_gamma), works

def plot_gamma_distribution(train_gammas: list, val_gammas: list, calc_name: str, save=False, save_dir=None):
    plt.figure(figsize=(11, 8.5))
    plt.hist(train_gammas, bins=5, alpha=0.5, label='train')
    plt.hist(val_gammas, bins=5, alpha=0.5, label='val')
    plt.legend(loc='upper right')
    plt.title(f'Gamma distribution for {calc_name}')
    plt.xlabel('Gamma')
    plt.ylabel('Count')
    # if save:
    #     plt.savefig(f'{calc_name}_gamma_distribution.pdf', bbox_inches='tight')

