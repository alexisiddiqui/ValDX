### Class to contain the HDX experiment

# inputs:
# settings
# experimental data
# matched topology and peptide-residue segments
# simulation MD data 

from ValDX.VDX_Settings import Settings
from ValDX.Experiment_ABC import Experiment
from ValDX.helpful_funcs import segs_to_df, dfracs_to_df, segs_to_file
from ValDX.HDX_plots import plot_lcurve, plot_gamma_distribution, plot_dfracs_compare, plot_paired_errors
from HDXer.reweighting import MaxEnt

import pandas as pd
import time
import os 
import pickle
import glob
import subprocess
import numpy as np

class ValDXer(Experiment):
    def __init__(self, settings: Settings, name=None):
        super().__init__(settings, name)
        if name is not None:
                self.name = name
        else:
             self.name = self.settings.name
 
        self.load_HDXer()
        self.generate_directory_structure(overwrite=False)
    
    def load_HDX_data(self, HDX_path=None, SEG_path=None, calc_name: str=None, experimental=False):
        """
        Load HDX data and Matched Residue Segments paths: Experiment and Predicted.
        Does not check if the paths are valid. 
        May fail if incorrect format during prepare HDX data.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")
        if HDX_path is None and experimental is False:
            raise ValueError(f"Please provide a path to the experimental {calc_name} HDX data.")
        if HDX_path and SEG_path is not None:
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], "SEG": [SEG_path], "calc_name": [calc_name], "experimental": [experimental]})
        elif HDX_path is None:
            paths_to_add = pd.DataFrame({"SEG": [SEG_path], "calc_name": [calc_name], "experimental": [experimental]})
        elif SEG_path is None:
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], "calc_name": [calc_name], "experimental": [experimental]})
        
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)

        hdx, segs  = self.prepare_HDX_data(calc_name=calc_name)

        if hdx and segs is None:
            raise ValueError(f"Unable to prepare any HDX data for {calc_name}.")

    def load_structures(self, top_path, traj_paths:str=None, calc_name: str=None):
        """
        Load structures: topology and trajectories (Optional).
        If no trajectories are provided, the topology is used to generate a single frame trajectory.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")

        if traj_paths is None:
            traj_paths = top_path

        paths_to_add = pd.DataFrame({"top": [top_path], "traj": [traj_paths], "calc_name": [calc_name]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)

        top, traj = self.prepare_structures(calc_name=calc_name)

    def load_HDXer(self, HDXer_env=None, HDXer_path=None):
        """
        Load HDXer: HDXer environment and HDXer executable.
        """

        if HDXer_env is not None:
            self.HDXer_env = HDXer_env
            self.HDXer_path = os.environ["HDXER_PATH"]
        elif HDXer_path is not None:
            self.HDXer_path = HDXer_path

        test_HDX_command = ['python', self.HDXer_path, '-h']

        try:
            subprocess.run(test_HDX_command, env=self.HDXer_env, check=True)
            return True
        except subprocess.CalledProcessError:
            print("HDXer failed to run. Check the HDXer environment and executable path.")
            return False


    def predict_HDX(self, calc_name: str=None, mode: str=None, rep: int=None, train: bool=True):
        """
        Predict HDX data from MD trajectories.
        HDX data from calc_name
        Segs data from rep_name

        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")
        if train:
            rep_name = "_".join(["train", calc_name, str(rep)])
        else:
            rep_name = "_".join(["val", calc_name, str(rep)])

        # folder should exist
        _, out_dir = self.generate_directory_structure(calc_name=rep_name, gen_only=True)

        calc_hdx = os.path.join(self.HDXer_path, "HDXer", "calc_hdx.py")
        top = self.paths.loc[self.paths["calc_name"] == calc_name, "top"]
        trajs = self.paths.loc[self.paths["calc_name"] == calc_name, "traj"]
        log = os.path.join(out_dir, self.settings.logfile_name[0] + rep_name + self.settings.logfile_name[1])
        if train:
            segs = self.train_segs.loc[self.train_segs["calc_name"] == calc_name, "path"]
        else:
            segs = self.val_segs.loc[self.val_segs["calc_name"] == calc_name, "path"]
        out_prefix = os.path.join(out_dir, "_".join([self.settings.outname, rep_name]))

        if self.load_HDXer():
            ### how do we add times
            calc_hdx_command = ["python",
                                calc_hdx,
                                "-t", trajs,
                                "-p", top,
                                "-m", self.settings.HDX_method,
                                "-log", log,
                                "-out", out_prefix,
                                "-segs", segs,
                                "-mopt", self.settings.HDXer_mopt]
                                

            subprocess.run(calc_hdx_command, env=self.HDXer_env, check=True)

            ### read HDX data into df
            df = dfracs_to_df(out_prefix + "_segment_average_fractions.dat", names=self.settings.times)

            df["calc_name"] = [rep_name for i in range(len(df))]

            self.load_intrinsic_rates(out_prefix + "_intrinsic_rates.dat", calc_name=calc_name)
            
            return df, rep_name
        

    def reweight_HDX(self, expt_name: str=None, calc_name: str=None, gamma_range: tuple=(2,10), train: bool=True, rep: int=None, train_gamma: float=None):
        """
        Reweight HDX data from previsously predicted HDX data performed by predict_HDX().
        """
        if expt_name or calc_name is None:
            raise ValueError("Please provide a Calculation AND Experimental name for the structures.")
        if not train and train_gamma is None:
            raise ValueError("If validating, must provide a gamma to from a previous training run")

        if train:
            rep_name = "_".join(["train", calc_name, str(rep)])
        else:
            rep_name = "_".join(["val", calc_name, str(rep)])

        predictHDX_dir = os.path.join(self.settings.data_dir, self.name, rep_name)
        expt = self.paths.loc[self.paths["calc_name"] == expt_name, "HDX"]
        rates = self.paths.loc[self.paths["calc_name"] == calc_name, "rates"]
        times = self.settings.times
        exponent = self.settings.RW_exponent

        # how do we do this for validation data? I guess this is is simply a procedure - does it 
        if train:
            for r in range(gamma_range):
                reweight_object = MaxEnt(do_reweight=self.settings.RW_do_reweighting, 
                                         do_params=self.settings.RW_do_params, 
                                         stepfactor=self.settings.RW_stepfactor)
                
                reweight_object.run(gamma=self.settings.RW_basegamma*r, 
                                    data_folders=predictHDX_dir, 
                                    kint_file=rates, 
                                    exp_file=expt, 
                                    times=times, 
                                    restart_interval=self.settings.RW_restart_interval, 
                                    out_prefix=self.settings.RW_outprefix+f"{r}x10^{exponent}")
                
            # plot L-curve - return closest gamma value
            opt_gamma =  plot_lcurve(calc_name=rep_name, RW_range=gamma_range, gamma=train_gamma, RW_dir=predictHDX_dir, prefix=self.settings.RW_outprefix)
            # read in reweighted data using opt_gamma if train
            if train:
                RW_path = os.path.join(predictHDX_dir, self.settings.RW_outprefix+f"{opt_gamma}x10^{exponent}_reweighted_segment_average_fractions.dat")
            elif train_gamma is not None:
                RW_path = os.path.join(predictHDX_dir, self.settings.RW_outprefix+f"{train_gamma}x10^{exponent}_reweighted_segment_average_fractions.dat")

            reweighted_df = dfracs_to_df(path=RW_path, names=self.settings.times)
            reweighted_df["calc_name"] = [rep_name for i in range(len(reweighted_df))]

            return opt_gamma, reweighted_df


    def run_VDX(self, calc_name: str=None, expt_name: str=None, mode: str=None, n_reps: int=None, random_seeds: list=None):
        
        if random_seeds is None:
            random_seeds = [i for i in range(self.settings.random_seed, self.settings.random_seed + n_reps)]

        train_gammas = []
        val_gammas = []

        train_dfs = pd.DataFrame()
        val_dfs = pd.DataFrame()

        ## change functions to export dataframes which are then passed into evaulate HDX
        for rep in range(n_reps):

            _, train_rep_name, val_rep_name = self.split_segments(calc_name=calc_name, rep=rep, random_seed=random_seeds[rep])

            # train HDX
            train_opt_gamma, train_df = self.train_HDX(calc_name=train_rep_name, expt_name=expt_name, mode=mode, rep=rep)
            train_gammas.append(train_opt_gamma)

            # validation HDX
            val_opt_gamma, val_df = self.validate_HDX(calc_name=val_rep_name, expt_name=expt_name, mode=mode, rep=rep, train_gamma=train_opt_gamma)
            val_gammas.append(val_opt_gamma)  

            train_dfs = pd.concat([train_dfs, train_df], ignore_index=True)
            val_dfs = pd.concat([val_dfs, val_df], ignore_index=True)

        # evaluate HDX train vs val - how do we actually compare both? I guess we just take the average across the reps - how do we account for peptides?
        self.evaluate_HDX(train_dfs=train_dfs, val_dfs=val_dfs, data=self.HDX_data, expt_name=expt_name, calc_name=calc_name, mode=mode, rep=rep, train_gammas=train_gammas, val_gammas=val_gammas, n_reps=n_reps)
        # plot optimal gamma distributions

        

    def train_HDX(self, calc_name: str=None, expt_name: str=None, mode: str=None, rep: int=None):
        ### need to rethink how to do train - val split for the names - each train rep needs to be in its own folder - reweighting uses an entire directory

        # train_opt_gammas = []

        # for rep in range(n_reps):

        rep_df, _ = self.predict_HDX(calc_name=calc_name, mode=mode, rep=rep, train=True)

        # add df to HDX_data
        self.HDX_data = pd.concat([self.HDX_data, rep_df], ignore_index=True)

        gamma, df = self.reweight_HDX(expt_name=expt_name, calc_name=calc_name, train=True, rep=rep)

        # train_opt_gammas.append(gamma)

        # mean_opt_gamma = np.mean(train_opt_gammas)

        return gamma, df

    def validate_HDX(self, calc_name: str=None, expt_name: str=None, mode: str=None, rep: int=None, train_gamma: float=None):
        # compare both training and validation data to the experimental data
        # show the averages and the distributions of the errors

        # val_opt_gammas = []

        # for rep in range(n_reps):

        rep_df, _ = self.predict_HDX(calc_name=calc_name, mode=mode, rep=rep, train=False)

        # add df to HDX_data
        self.HDX_data = pd.concat([self.HDX_data, rep_df], ignore_index=True)

        gamma, df = self.reweight_HDX(expt_name=expt_name, calc_name=calc_name, train=False, rep=rep)

            # val_opt_gammas.append(gamma)

        # val_mean_opt_gamma = np.mean(val_opt_gammas)

        return gamma, df
    

    def evaluate_HDX(self, train_dfs: list(pd.DataFrame), val_dfs: list(pd.DataFrame), data: pd.DataFrame=None, expt_name: str=None, calc_name: str=None, mode: str=None, rep: int=None, train_gammas: float=None, val_gammas: float=None, n_reps: int=None):
        

        plot_gamma_distribution(calc_name=calc_name, train_gammas=train_gammas, val_gammas=val_gammas)

        # plot the individual runs from train and val
        train_rep_names = ["_".join(["train", calc_name, str(rep)]) for rep in range(n_reps)]
        val_rep_names = ["_".join(["val", calc_name, str(rep)]) for rep in range(n_reps)]

        args = [expt_name, *train_rep_names]
        plot_dfracs_compare(args, data=self.HDX_data, times= self.settings.times)

        args = [expt_name, *val_rep_names]
        plot_dfracs_compare(args, data=self.HDX_data, times= self.settings.times)

        # average replicate data together - then plot
        train_avg_name = "_".join(["train", calc_name, "avg"])
        val_avg_name = "_".join(["val", calc_name, "avg"])

        # first we just group them together and compare overall distributions
        
        # set all calc names in train and val dfs to be calc_name
        train_merge_df = pd.concat(train_dfs, ignore_index=True)
        train_merge_df["calc_name"] = "_".join(["train", calc_name, "all"])
        val_merge_df = pd.concat(val_dfs, ignore_index=True)
        val_merge_df["calc_name"] = "_".join(["val", calc_name, "all"])

        avg_df = pd.concat([train_merge_df, val_merge_df], ignore_index=True)

        args = [expt_name, train_avg_name,  val_avg_name]
        plot_dfracs_compare(args, data=avg_df, times= self.settings.times)

        expt_segs = self.segs[["calc_name"] == expt_name]

        for r in range(n_reps):
            train_segs_df = self.train_segs.loc[self.train_segs["calc_name"] == train_rep_names[r]]
            val_segs_df = self.val_segs.loc[self.val_segs["calc_name"] == val_rep_names[r]]

            # add correct (original) peptide numbers to train and val data df
            train_dfs[r]["peptide"] = train_segs_df["peptide"]
            val_dfs[r]["peptide"] = val_segs_df["peptide"]

        train_dfs = pd.concat(train_dfs, ignore_index=True)
        val_dfs = pd.concat(val_dfs, ignore_index=True)

        train_avg_df = train_dfs.groupby(["peptide", "time"]).mean()
        train_avg_df["calc_name"] = [train_avg_name]*len(train_avg_df)
        val_avg_df = val_dfs.groupby(["peptide", "time"]).mean()
        val_avg_df["calc_name"] = [val_avg_name]*len(val_avg_df)

        train_peptides = set(train_avg_df["peptide"].values)
        val_peptides = set(val_avg_df["peptide"].values)
        expt_peptides = set(expt_segs["peptide"].values)

        train_intersection = train_peptides.intersection(expt_peptides)
        val_intersection = val_peptides.intersection(expt_peptides)

        train_coverage = 100*len(train_intersection)/len(expt_peptides)
        val_coverage = 100*len(val_intersection)/len(expt_peptides)

        print(f"Train coverage: {train_coverage:.2f}")
        print(f"Val coverage: {val_coverage:.2f}")
        
        # add to HDX_data
        self.HDX_data = pd.concat([self.HDX_data, train_avg_df, val_avg_df], ignore_index=True)
        # plot train and val against expt data with compare plot
        args = [expt_name, train_avg_name, val_avg_name]
        plot_dfracs_compare(args, data=self.HDX_data, times= self.settings.times)
        # plot train and val averages against expt data with paired plot
        plot_paired_errors(args, data=self.HDX_data, times= self.settings.times)



# rates are required for reweighting???
    def load_intrinsic_rates(self, path, calc_name: str=None):
        """
        Load intrinsic rates: intrinsic rates from HDXer.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")

        paths_to_add = pd.DataFrame({"int_rates": [path], "calc_name": [calc_name]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)

