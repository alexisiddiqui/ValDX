### Class to contain the HDX experiment

# inputs:
# settings
# experimental data
# matched topology and peptide-residue segments
# simulation MD data 
import pandas as pd
import time
import os 
import pickle
import glob
import subprocess
import numpy as np
import concurrent.futures

from HDXer.reweighting import MaxEnt

from ValDX.VDX_Settings import Settings
from ValDX.Experiment_ABC import Experiment
from ValDX.helpful_funcs import  conda_to_env_dict, segs_to_df, dfracs_to_df, segs_to_file, run_MaxEnt, restore_trainval_peptide_nos, add_nan_values
from ValDX.HDX_plots import plot_lcurve, plot_gamma_distribution, plot_dfracs_compare, plot_paired_errors



class ValDXer(Experiment):
    def __init__(self, 
                 settings: Settings, 
                 name=None):
        super().__init__(settings, name=None)
        if name is not None:
                self.name = name
        else:
             self.name = self.settings.name
        self.settings.data_dir = os.path.join(os.getcwd(), self.settings.data_dir)
        self.load_HDXer()
        self.generate_directory_structure(overwrite=False)
    
    def load_HDX_data(self, 
                      HDX_path:str=None, 
                      SEG_path:str=None, 
                      calc_name: str=None, 
                      experimental=False):
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
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], 
                                         "SEG": [SEG_path], 
                                         "calc_name": [calc_name], 
                                         "experimental": [experimental]})
        elif HDX_path is None:
            paths_to_add = pd.DataFrame({"SEG": [SEG_path], 
                                         "calc_name": [calc_name], 
                                         "experimental": [experimental]})
        elif SEG_path is None:
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], 
                                         "calc_name": [calc_name], 
                                         "experimental": [experimental]})
        
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)
        hdx, segs  = self.prepare_HDX_data(calc_name=calc_name)

        if hdx is None and segs is None:
            raise ValueError(f"Unable to prepare any HDX data for {calc_name}.")

    def load_structures(self, 
                        top_path, 
                        traj_paths: str=None, 
                        calc_name: str=None):
        """
        Load structures: topology and trajectories (Optional).
        If no trajectories are provided, the topology is used to generate a single frame trajectory.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")

        if traj_paths is None:
            traj_paths = top_path

        paths_to_add = pd.DataFrame({"top": [top_path], 
                                     "traj": [traj_paths], 
                                     "calc_name": [calc_name]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)
        print(self.paths)
        top, traj = self.prepare_structures(calc_name=calc_name)

    def load_HDXer(self, 
                   HDXer_env: dict=None, 
                   HDXer_path: str=None):
        """
        Load HDXer: HDXer environment and HDXer executable.
        """

        if HDXer_env is not None:
            self.HDXer_env = HDXer_env
            self.HDXer_path = os.environ["HDXER_PATH"]
        elif HDXer_path is not None:
            self.HDXer_path = HDXer_path

        calc_hdx = os.path.join(self.HDXer_path, "HDXer", "calc_hdx.py")


        test_HDX_command = ['python', calc_hdx, '-h']

        env_path = conda_to_env_dict(self.HDXer_env)
        try:
            _ = subprocess.run(test_HDX_command, 
                           env=env_path, 
                           check=True,
                           capture_output=True)
            return True
        except subprocess.CalledProcessError:
            print("HDXer failed to run. Check the HDXer environment and executable path.")
            return False


    def predict_HDX(self, 
                    calc_name: str=None, 
                    # mode: str=None, 
                    rep: int=None, 
                    train: bool=True):
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
        _, out_dir = self.generate_directory_structure(calc_name=rep_name, 
                                                       gen_only=True)
        print(out_dir)
        calc_hdx = os.path.join(self.HDXer_path, "HDXer", "calc_hdx.py")
        print(calc_name)
        top = self.paths.loc[self.paths["calc_name"] == calc_name, "top"].dropna().values[0]
        trajs = self.paths.loc[self.paths["calc_name"] == calc_name, "traj"].dropna().values[0]
        log = os.path.join(out_dir, self.settings.logfile_name[0] + rep_name + self.settings.logfile_name[1])
        if train:
            segs = self.train_segs.loc[self.train_segs["calc_name"] == rep_name, "path"].dropna().values[0]
        else:
            segs = self.val_segs.loc[self.val_segs["calc_name"] == rep_name, "path"].dropna().values[0]
        out_prefix = os.path.join(out_dir, "_".join([self.settings.outname, rep_name]))
        print(out_prefix)
        if self.load_HDXer():
            ### how do we add times
            calc_hdx_command = ["python",
                                calc_hdx,
                                "-t", *trajs,
                                "-p", top,
                                "-m", self.settings.HDX_method,
                                "-log", log,
                                "-out", out_prefix, 
                                "-seg", segs,
                                "-mopt", self.settings.HDXer_mopt]
                                
            calc_hdx_command  =  " ".join(calc_hdx_command)
            # calc_hdx_command.extend(["-t", traj] for traj in trajs)
            print(calc_hdx_command)
            # print(" ".join(calc_hdx_command))
            env_path = conda_to_env_dict(self.HDXer_env)

            subprocess.run(calc_hdx_command, 
                           env=env_path, 
                           shell=True,
                           check=True,
                           cwd=out_dir)

            ### read HDX data into df

            df = dfracs_to_df(out_prefix + "segment_average_fractions.dat", 
                              names=self.settings.times)

            df["calc_name"] = [rep_name for i in range(len(df))]
            out_prefix = os.path.join(os.getcwd(), out_prefix)
            self.load_intrinsic_rates(out_prefix + "Intrinsic_rates.dat", 
                                      calc_name=rep_name)
            
            return df, rep_name
        


    

    def reweight_HDX(self, 
                     expt_name: str=None, 
                     calc_name: str=None, 
                     gamma_range: tuple=None, 
                     train: bool=True, 
                     rep: int=None, 
                     train_gamma: float=None):
        """
        Reweight HDX data from previsously predicted HDX data performed by predict_HDX().
        """
        print(expt_name, calc_name, train, rep)
        if gamma_range is None:
            gamma_range = self.settings.gamma_range
        if (expt_name or calc_name) is None:
            raise ValueError("Please provide a Calculation AND Experimental name for the structures.")
        if not train and train_gamma is None:
            raise ValueError("If validating, must provide a gamma to from a previous training run")

        if train:
            rep_name = "_".join(["train", calc_name, str(rep)])
        else:
            rep_name = "_".join(["val", calc_name, str(rep)])

        predictHDX_dir = os.path.join(self.settings.data_dir, self.name, rep_name)
        # change this to use the training HDX data
        # expt = self.paths.loc[self.paths["calc_name"] == expt_name, "HDX"].dropna().values[0]
        if train:
            expt = self.train_HDX_data.loc[self.train_HDX_data["calc_name"] == rep_name, "path"].dropna().values[0]
        else:
            expt = self.val_HDX_data.loc[self.val_HDX_data["calc_name"] == rep_name, "path"].dropna().values[0]
        rates = self.paths.loc[self.paths["calc_name"] == rep_name, "int_rates"].dropna().values[0]
        print(expt)
        print(predictHDX_dir)
        print(rates)
        times = self.settings.times
        exponent = self.settings.RW_exponent

        # package all args except gamma into a dictionary
        args = {
            "do_reweight": self.settings.RW_do_reweighting, 
            "do_params": self.settings.RW_do_params, 
            "stepfactor": self.settings.RW_stepfactor, 
            "basegamma": self.settings.RW_basegamma, 
            "predictHDX_dir": [predictHDX_dir], 
            "kint_file": rates, 
            "exp_file": expt, 
            "times": times, 
            "restart_interval": self.settings.RW_restart_interval,
            "out_prefix": os.path.join(predictHDX_dir, self.settings.RW_outprefix),
            "exponent": exponent,
            }


        # how do we do this for validation data? I guess this is is simply a procedure - does it 
        if train is not None:
            
            try:
                print("Trying concurrent.futures")
                args_r = [(args, r) for r in range(*gamma_range)]
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    executor.map(run_MaxEnt, args_r)

            except UserWarning("Concurrent.futures failed. Trying without concurrent.futures"):
                for r in range(*gamma_range):
                    print(f"Reweighting {rep_name} with gamma = {r}x10^{args['exponent']}")
                    ### concurrent.futures
                    run_MaxEnt(args_r)
                    ### cnocurrent.futures       

            if args["do_reweight"] is False:
                RW_path = os.path.join(predictHDX_dir,
                                        self.settings.RW_outprefix+f"{gamma_range[0]}x10^{exponent}final_segment_fractions.dat")
                reweighted_df = dfracs_to_df(path=RW_path,
                                            names=self.settings.times)
                reweighted_df["calc_name"] = [rep_name] * len(reweighted_df)
                opt_gamma = 0
                return opt_gamma, reweighted_df

            # plot L-curve - return closest gamma value
            opt_gamma, _ =  plot_lcurve(calc_name=rep_name, 
                                     RW_range=gamma_range, 
                                     gamma=train_gamma, 
                                     RW_dir=predictHDX_dir, 
                                     prefix=self.settings.RW_outprefix)
            print(f"Optimal gamma for {rep_name} is {opt_gamma}x10^{exponent}")
            # read in reweighted data using opt_gamma if train
            if train:
                RW_path = os.path.join(predictHDX_dir, 
                                       self.settings.RW_outprefix+f"{opt_gamma}x10^{exponent}final_segment_fractions.dat")
            elif train_gamma is not None:
                RW_path = os.path.join(predictHDX_dir, 
                                       self.settings.RW_outprefix+f"{train_gamma}x10^{exponent}final_segment_fractions.dat")
            print(RW_path)
            reweighted_df = dfracs_to_df(path=RW_path, 
                                         names=self.settings.times)
            print(reweighted_df)
            reweighted_df["calc_name"] = [rep_name] * len(reweighted_df)
            print(reweighted_df)

            return opt_gamma, reweighted_df
        


    def run_VDX(self, 
                calc_name: str=None, 
                expt_name: str=None, 
                mode: str=None, # not implemented yet
                n_reps: int=None, 
                random_seeds: list=None):
        
        if n_reps is None:
            n_reps = self.settings.replicates
        if random_seeds is None:
            random_seeds = [self.settings.random_seed+i for i in range(n_reps)]


        train_gammas = []
        val_gammas = []

        train_dfs = []
        val_dfs = []

        ## change functions to export dataframes which are then passed into evaulate HDX
        for rep in range(1,n_reps+1):

            _, train_rep_name, val_rep_name = self.split_segments(seg_name=expt_name,
                                                                  calc_name=calc_name, 
                                                                  rep=rep, 
                                                                  random_seed=random_seeds[rep-1])

            # train HDX
            train_opt_gamma, train_df = self.train_HDX(calc_name=calc_name, 
                                                       expt_name=expt_name, 
                                                       mode=mode, 
                                                       rep=rep)
            train_gammas.append(train_opt_gamma)

            # validation HDX
            val_opt_gamma, val_df = self.validate_HDX(calc_name=calc_name, 
                                                      expt_name=expt_name, 
                                                      mode=mode, 
                                                      rep=rep, 
                                                      train_gamma=train_opt_gamma)
            val_gammas.append(val_opt_gamma)  

            train_dfs.append(train_df)
            val_dfs.append(val_df)

        # evaluate HDX train vs val - how do we actually compare both? I guess we just take the average across the reps - how do we account for peptides?
        self.evaluate_HDX(train_dfs=train_dfs, 
                          val_dfs=val_dfs, 
                          expt_name=expt_name, 
                          calc_name=calc_name, 
                          train_gammas=train_gammas, 
                          val_gammas=val_gammas,
                          n_reps=n_reps)
        # plot optimal gamma distributions

        

    def train_HDX(self, 
                  calc_name: str=None, 
                  expt_name: str=None, 
                  mode: str=None, 
                  rep: int=None):
        ### need to rethink how to do train - val split for the names - each train rep needs to be in its own folder - reweighting uses an entire directory

        # train_opt_gammas = []

        # for rep in range(n_reps):

        rep_df, _ = self.predict_HDX(calc_name=calc_name, 
                                     rep=rep, 
                                     train=True)

        # add df to HDX_data

        gamma, df = self.reweight_HDX(expt_name=expt_name, 
                                      calc_name=calc_name, 
                                      train=True, 
                                      rep=rep)

        self.HDX_data = pd.concat([self.HDX_data, df], ignore_index=True)
        # train_opt_gammas.append(gamma)

        # mean_opt_gamma = np.mean(train_opt_gammas)

        return gamma, df

    def validate_HDX(self, 
                     calc_name: str=None, 
                     expt_name: str=None, 
                     mode: str=None, 
                     rep: int=None, 
                     train_gamma: float=None):
        # compare both training and validation data to the experimental data
        # show the averages and the distributions of the errors

        # val_opt_gammas = []

        # for rep in range(n_reps):

        rep_df, _ = self.predict_HDX(calc_name=calc_name, 
                                     rep=rep, 
                                     train=False)

        # add df to HDX_data
        gamma, df = self.reweight_HDX(expt_name=expt_name, 
                                      calc_name=calc_name, 
                                      train=False, 
                                      rep=rep, 
                                      train_gamma=train_gamma)

        self.HDX_data = pd.concat([self.HDX_data, df], ignore_index=True)
            # val_opt_gammas.append(gamma)

        # val_mean_opt_gamma = np.mean(val_opt_gammas)

        return gamma, df
    

    def evaluate_HDX(self, 
                     train_dfs: list[pd.DataFrame], 
                     val_dfs: list[pd.DataFrame], 
                     data: pd.DataFrame=None, 
                     expt_name: str=None, 
                     calc_name: str=None, 
                     mode: str=None, 
                     train_gammas: float=None, 
                     val_gammas: float=None, 
                     n_reps: int=None):
        
        times = self.settings.times

        plot_gamma_distribution(calc_name=calc_name, 
                                train_gammas=train_gammas, 
                                val_gammas=val_gammas)

        # plot the individual runs from train and val
        train_rep_names = ["_".join(["train", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
        val_rep_names = ["_".join(["val", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
        print(train_rep_names)
        print(val_rep_names)
        args = [expt_name, *train_rep_names]


        plot_dfracs_compare(args, 
                            data=self.HDX_data, 
                            times=self.settings.times)

        args = [expt_name, *val_rep_names]
        plot_dfracs_compare(args, 
                            data=self.HDX_data, 
                            times=self.settings.times)

    
        # 
        expt_segs = self.segs[self.segs["calc_name"] == expt_name].copy()

        # train_val_segs = pd.concat([self.train_segs, self.val_segs], ignore_index=True)

        # segs = pd.merge(expt_segs, train_val_segs, on=["ResStr", "ResEnd"], how="outer")

        merge_df = restore_trainval_peptide_nos(calc_name=calc_name,
                                    expt_name=expt_name,
                                    train_dfs=train_dfs,
                                    val_dfs=val_dfs,
                                    train_segs=self.train_segs,
                                    val_segs=self.val_segs,
                                    n_reps=n_reps,
                                    times=self.settings.times,
                                    expt_segs=expt_segs)


        expt_df = self.HDX_data[self.HDX_data["calc_name"] == expt_name]
        merge_df = pd.concat([expt_df, merge_df], ignore_index=True)
        # print(merge_df)
        args = [expt_name, *train_rep_names,  *val_rep_names]
        try:
            plot_dfracs_compare(args, 
                            data=merge_df, 
                            times=self.settings.times)
        except UserWarning:
            print("Unable to plot compare plot for merge_df")
        ####


        try:    
            plot_paired_errors(args,
                            data=merge_df, 
                            times=self.settings.times)
        except UserWarning:
            print("Unable to plot paired errors for nan_df")

        # return

        # Currently df contains values for the peptides in each train/val split 
        # we need to add nan values to the peptides which are not present in either split
        
        # first create df with all peptides

        nan_df = add_nan_values(calc_name=calc_name,
                                merge_df=merge_df,
                                n_reps=n_reps,
                                times=self.settings.times,
                                expt_segs=expt_segs)
            
        print("nan_df")
        print(nan_df)

        # add expt_df to nan_df
        nan_df = pd.concat([nan_df, expt_df], ignore_index=True)
        print("nan_df + expt_df")
        print(nan_df)

        # return


        args = [expt_name, *train_rep_names,  *val_rep_names]
        # this doesnt work - we need to either line up the peptides or just plot the averages
        try:
            plot_dfracs_compare(args, 
                            data=nan_df, 
                            times=self.settings.times)
        except UserWarning:
            print("Unable to plot compare plot for nan_df")
        ####


        try:    
            plot_paired_errors(args,
                            data=nan_df, 
                            times=self.settings.times)
        except UserWarning:
            print("Unable to plot paired errors for nan_df")

        # return
    
        expt_segs = self.segs[self.segs["calc_name"] == expt_name]

        train_avg_name = "_".join(["train", calc_name, "avg"])
        val_avg_name = "_".join(["val", calc_name, "avg"])

        train_df = merge_df[merge_df["calc_name"].isin(train_rep_names)]
        val_df = merge_df[merge_df["calc_name"].isin(val_rep_names)]

        train_avg_df = train_df.drop(columns="calc_name").groupby([*times,"peptide"]).mean().reset_index()
        val_avg_df = val_df.drop(columns="calc_name").groupby([*times,"peptide"]).mean().reset_index()

        train_avg_df["calc_name"] = [train_avg_name]*len(train_avg_df)
        val_avg_df["calc_name"] = [val_avg_name]*len(val_avg_df)

        avg_df = pd.concat([expt_df, train_avg_df, val_avg_df], ignore_index=True)
        
        # # return 

        # train_avg_df = pd.concat(train_avg_dfs, ignore_index=True)
        # val_avg_df = pd.concat(val_avg_dfs, ignore_index=True)
        # print(train_avg_df)
        # print(val_avg_df)


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
        # avg_df = pd.concat([expt, train_avg_df, val_avg_df], ignore_index=True)
        # plot train and val against expt data with compare plot
        args = [expt_name, train_avg_name, val_avg_name]
        try:
            plot_dfracs_compare(args,
                             data=avg_df, 
                             times=self.settings.times)
        except UserWarning:
            print("Unable to plot compare plot for avg_df")
        # plot train and val averages against expt data with paired plot
        try:
            plot_paired_errors(args, 
                           data=avg_df, 
                           times=self.settings.times)
        except UserWarning:
            print("Unable to plot paired errors for avg_df")



# rates are required for reweighting???
# should add a method to add rates to the df
    def load_intrinsic_rates(self, path, calc_name: str=None, experimental=False):
        """
        Load intrinsic rates: intrinsic rates from HDXer.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")

        paths_to_add = pd.DataFrame({"int_rates": [path], "calc_name": [calc_name], "experimental": [experimental]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True,)


     