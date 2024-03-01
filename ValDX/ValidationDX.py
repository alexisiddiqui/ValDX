### Class to contain the HDX experiment

# inputs:
# settings
# experimental data
# matched topology and peptide-residue segments
# simulation MD data 
import pandas as pd
import math
import time
import os 
import pickle
import glob
import copy
from copy import deepcopy
import subprocess
import numpy as np
import concurrent.futures
import MDAnalysis as mda
import seaborn as sns
import datetime
from typing import List, Tuple
from icecream import ic
# from .reweighting import MaxEnt

# import copy


from ValDX.VDX_Settings import Settings
from ValDX.Experiment_ABC import Experiment
from ValDX.helpful_funcs import  conda_to_env_dict, segs_to_df, dfracs_to_df, segs_to_file, run_MaxEnt, restore_trainval_peptide_nos, add_nan_values, kints_to_dict, merge_kint_dicts_into_df, calc_traj_LogP_byres, calc_dfrac_uptake_from_LogPf, cluster_traj_by_density, recluster_traj_by_weight
from ValDX.HDX_plots import plot_lcurve, plot_gamma_distribution, plot_dfracs_compare, plot_paired_errors, plot_heatmap_trainval_compare, plot_heatmap_trainval_compare_error, plot_R_agreement_trainval, plot_dfracs_compare_MSE, plot_dfracs_compare_abs, split_benchmark_plot_MSE_by_name, split_benchmark_plot_MSE_by_protein_split, split_benchmark_plot_MSE_by_split_protein, split_benchmark_plot_MSE_by_split, split_benchmark_plot_MSE_by_protein, split_benchmark_BV_scatterplot, split_benchmark_BV_boxplot_by_protein, split_benchmark_BV_boxplot_by_split_type, split_benchmark_BV_boxplot_by_protein_by_split_type, split_benchmark_BV_boxplot_by_split_type_by_protein


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
        self.HDXer_path = self.settings.HDXer_path
        self.HDXer_env = self.settings.HDXer_env
        self.load_HDXer()
        self.generate_directory_structure(overwrite=False)
        self.analysis = pd.DataFrame()
        # self.settings.plot_dir = os.path.join(self.settings.plot_dir, self.settings.name)
    
    def load_HDX_data(self, 
                      HDX_path: str=None, 
                      SEG_path: str=None, 
                      calc_name : str=None,
                      times: list=None, 
                      experimental=False):
        """
        Load HDX data and Matched Residue Segments paths: Experiment and Predicted.
        Does not check if the paths are valid. 
        May fail if incorrect format during prepare HDX data.
        """
        if times is not None:
            self.times = times

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
        else:
            return hdx, segs

    def load_structures(self, 
                        top_path, 
                        traj_paths: list=None, 
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


        # test_HDX_command = ['python', calc_hdx, '-h']

        # test_HDX_command = ['conda', 'run', '-n', 'HDXER_ENV', "python", calc_hdx, '-h']
        env_path = conda_to_env_dict(self.HDXer_env)
        print("calc_hdx")
        print(calc_hdx)
        test_HDX_command = f"""
        conda activate HDXER_ENV
        python {calc_hdx} -h
        """
        # print(" ".join(test_HDX_command))
        try:
            _ = subprocess.run(test_HDX_command, 
                            shell=True, executable="/bin/bash",
                        #    env=env_path, 
                           check=True,
                           capture_output=True)
            return True
        except:
            raise EnvironmentError("HDXer failed to run. Check the HDXer environment and executable path.")
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
        times_as_str_list = [str(time) for time in self.settings.times]
        times_as_str = ' '.join(times_as_str_list)

        python = "python"
        python = "conda run -n HDXER_ENV python"

        if self.load_HDXer():
            ### how do we add times
            calc_hdx_command = [python,
                                calc_hdx,
                                "-t", *trajs,
                                "-p", top,
                                "-m", self.settings.HDX_method,
                                "-log", log,
                                "-out", out_prefix, 
                                "-seg", segs,
                                "-mopt", self.settings.HDXer_mopt,
                                "--times", times_as_str,
                                "-str", str(self.settings.HDXer_stride)]
                                
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

            df = dfracs_to_df(out_prefix + "Segment_average_fractions.dat", 
                              names=self.settings.times)

            df["calc_name"] = [rep_name for i in range(len(df))]
            # out_prefix = os.path.join(os.getcwd(), out_prefix)
            self.load_intrinsic_rates(out_prefix + "Intrinsic_rates.dat", 
                                      calc_name=rep_name)
            
            return df, rep_name
        


    

    def reweight_HDX(self, 
                     expt_name: str=None, 
                     calc_name: str=None, 
                     gamma_range: tuple=None, 
                     train: bool=True, 
                     rep: int=None, 
                     weights: np.array=None,
                     train_gamma: float=None):
        """
        R   eweight HDX data from previsously predicted HDX data performed by predict_HDX().
        """
        print(expt_name, calc_name, train, rep)
        if gamma_range is None:
            gamma_range = self.settings.gamma_range
        if (expt_name or calc_name) is None:
            raise ValueError("Please provide a Calculation AND Experimental name for the structures.")
        if not train and train_gamma is None:
            raise ValueError("If validating, must provide a gamma to from a previous training run")
        if train_gamma is not None:
            train_gamma_exponent = math.floor(math.log10(train_gamma))
            train_gamma_coefficient = train_gamma / 10**train_gamma_exponent
            
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
        args_e = []
        # exponent = self.settings.RW_exponent
        RW_exponents = self.settings.RW_exponent
        if train is False:
            RW_exponents = [train_gamma_exponent]
            gamma_range = (int(train_gamma_coefficient), int(train_gamma_coefficient+1))

        for exponent in RW_exponents:
            print(f"REWIGHTING {rep_name} with Exponent: {exponent}")
            RW_basegamma = 10**exponent
            # package all args except gamma into a dictionary
            args = {
                "do_reweight": self.settings.RW_do_reweighting, 
                "do_params": self.settings.RW_do_params, 
                "stepfactor": self.settings.RW_stepfactor, 
                "basegamma": RW_basegamma, 
                "predictHDX_dir": [predictHDX_dir], # requires brackets 
                "kint_file": rates, 
                "exp_file": expt, 
                "times": times, 
                "restart_interval": self.settings.RW_restart_interval,
                "out_prefix": os.path.join(predictHDX_dir, self.settings.RW_outprefix),
                "exponent": exponent,
                "random_initial": self.settings.random_initialisation,
                "temp": self.settings.temp, 
                'iniweights': weights
                }
            args_e.append(args)
        print(args_e)
        # how do we do this for validation data? I guess this is is simply a procedure - does it 
        if train is not None:
            try:
                print("Trying concurrent.futures")
                args_r = [(args, r) for r in range(*gamma_range) for args in args_e]
                with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                    outputs_cr_bc_bh = list(executor.map(run_MaxEnt, args_r))

            except UserWarning("Concurrent.futures failed. Trying without concurrent.futures"):
                print("Running directly")
                outputs_cr_bc_bh = []
                for idx, r in enumerate(range(*gamma_range)):
                    print(f"Reweighting {rep_name} with gamma = {r}x10^{args['exponent']}")
                    ### concurrent.futures
                    print("not using concurrent.futures")
                    
                    output = run_MaxEnt(args_r[idx])
                    outputs_cr_bc_bh.append(output)
                    ### cnocurrent.futures       
            finally:
                print("Finished reweighting")
                print(outputs_cr_bc_bh)
                # add outpus to respective dfss


        if self.settings.RW_do_reweighting is False:
            print("RW_do_reweighting is False")
            RW_path = os.path.join(predictHDX_dir,
                                    self.settings.RW_outprefix+f"{gamma_range[0]}x10^{exponent}final_segment_fractions.dat")
            reweighted_df = dfracs_to_df(path=RW_path,
                                        names=self.settings.times)
            reweighted_df["calc_name"] = [rep_name] * len(reweighted_df)
            opt_gamma = gamma_range[0]*10**exponent
            # select the correct output 
            cr_bc_bh = outputs_cr_bc_bh[0]

            return opt_gamma, reweighted_df, cr_bc_bh
        # if train is False:
        #     print("not using concurrent.futures as train is False")
        #     r = train_gamma_coefficient
        #     run_MaxEnt((args_e[0], r))

        if train is not False and self.settings.RW_do_reweighting is True:
            # plot L-curve - return closest gamma value
            opt_gamma, _ =  plot_lcurve(calc_name=rep_name, 
                                        RW_range=gamma_range, 
                                        gamma=train_gamma, 
                                        RW_dir=predictHDX_dir, 
                                        prefix=self.settings.RW_outprefix)
            opt_gamma_exponent = math.floor(math.log10(opt_gamma))
            opt_gamma_coefficient = opt_gamma / 10**opt_gamma_exponent
        if train is False:
            opt_gamma = train_gamma
            opt_gamma_exponent = train_gamma_exponent
            opt_gamma_coefficient = train_gamma_coefficient

        print(f"Optimal gamma for {rep_name} is {opt_gamma_coefficient}x10^{opt_gamma_exponent}")
        # read in reweighted data using opt_gamma if train
        if train:
            RW_path = os.path.join(predictHDX_dir, 
                                    self.settings.RW_outprefix+f"{int(opt_gamma_coefficient)}x10^{opt_gamma_exponent}final_segment_fractions.dat")
        elif train_gamma is not None:
            RW_path = os.path.join(predictHDX_dir, 
                                    self.settings.RW_outprefix+f"{int(train_gamma_coefficient)}x10^{train_gamma_exponent}final_segment_fractions.dat")
        print(RW_path)
        reweighted_df = dfracs_to_df(path=RW_path, 
                                        names=self.settings.times)
        print(reweighted_df)
        reweighted_df["calc_name"] = [rep_name] * len(reweighted_df)
        print(reweighted_df)

        # select the correct output
        gamma_index =int(opt_gamma_coefficient - gamma_range[0])
        cr_bc_bh = outputs_cr_bc_bh[gamma_index]

        return opt_gamma, reweighted_df, cr_bc_bh
    



    def recalculate_dataset(self, traj, cr_bc_bh, dataset_name, segs, rates:dict, train=False):
        print(f"Recalculating {dataset_name}")
        times = self.settings.times

        stride = self.settings.HDXer_stride

        # segs = self.segs[self.segs["calc_name"] == segs_name].copy()

        df = segs.drop(columns=["calc_name"]).copy()

        print(df)
        # convert segs df to list of residues from resstr to resend
        segs["Residues"]= segs.apply(lambda x: np.array(range(x["ResStr"], x["ResEnd"]+1)), axis=1)
        residues = segs["Residues"].to_numpy()
        residues = np.concatenate(residues)
        residues = np.unique(residues)
        start_res = np.sort(residues)[0]

        print(f"Residues for recalculation: {residues}")
        print(residues)
        print(rates.keys())
        # filter residues that dont exist in rates using numpy
        residues = np.array([res for res in residues if res in rates.keys()])

        print(f"Residues for recalculation: {residues}")




        LogPf_by_res = calc_traj_LogP_byres(universe=traj,
                                            B_C=cr_bc_bh[1],
                                            B_H=cr_bc_bh[2],
                                            stride=stride,
                                            residues=residues,
                                            weights=cr_bc_bh[0])
        
        LogPfs_to_add = pd.DataFrame({"LogPf": [LogPf_by_res], "calc_name": [dataset_name], "Residues": [residues], "name": self.settings.name})
        
        self.LogPfs = pd.concat([self.LogPfs, LogPfs_to_add], ignore_index=True)

        if not train:
            print("LogPf_by_res shape")
            print(LogPf_by_res.shape)
            print(LogPf_by_res)

            dfracs_by_res_overtime = calc_dfrac_uptake_from_LogPf(LogPf_by_res,
                                                                kints=rates,
                                                                times=self.settings.times,
                                                                residues=residues)
            print("dfracs_by_res_overtime shape")
            print(dfracs_by_res_overtime.shape)
            print(dfracs_by_res_overtime)
            # reshape dfracs_by_res_overtime to be in the same format as the HDX data
            # this means converting residues back to segments
            
            df["calc_name"] = [dataset_name]*len(df)
            df["Residues"] = df.apply(lambda x: list(range(x["ResStr"], x["ResEnd"]+1)), axis=1)
            print(df)

            peptides = df["peptide"].to_list()
            print(peptides)
            for p in peptides:
                print(p)
                # collect all dfrac updates for each residue in the peptide
                peptide_residues = df.loc[(df["peptide"] == p), "Residues"].to_list()[0]
                print(peptide_residues)
                # convert to indices based on order in residues
                peptide_indices = [idx for idx, res in enumerate(residues) if res in peptide_residues]
                
                dfracs = dfracs_by_res_overtime[:, peptide_indices]
                print(dfracs.shape)
                # average the dfrac updates for each residue in the peptide
                dfracs = np.mean(dfracs, axis=1)

                for idx, t in enumerate(times):
                    print(idx, t, dfracs[idx])
                    # select column t and peptide p
                    df.loc[(df["peptide"] == p), t] = dfracs[idx]
        

            df = df.drop(columns=["Residues"])
            print("Dataframe being appending")
            print(df)
            self.HDX_data = pd.concat([self.HDX_data, df], ignore_index=True)

            return df
    

    def recalculate_test_and_val(self, cr_bc_bh, calc_name, expt_name, rep=None):
        """
        This method takes the current weights of the frames as well as the BV parameters Bc and Bh 
        and recalculates weighted HDX data from the ensemble across the enture protein.
        We must also recaculate the segments of the residues that appear in both the experimental and predicted data. # not implemented yet
        """
        rep_name = "_".join(["train", calc_name, str(rep)])
        val_name = "_".join(["val", calc_name, str(rep)])
        test_name = "_".join(["test", calc_name, str(rep)])
 
    
        # add weights to df
        weights_to_add = pd.DataFrame({"weights": [cr_bc_bh[0]], "calc_name": [rep_name]})
        self.weights = pd.concat([self.weights, weights_to_add], ignore_index=True)

        # add parameeters to df
        params_to_add = pd.DataFrame({"Bc": cr_bc_bh[1], "Bh": cr_bc_bh[2], "calc_name": [rep_name]})
        self.BV_constants = pd.concat([self.BV_constants, params_to_add], ignore_index=True)
        # calculate PFs from weights and parameters using BV model
        top, traj = self.prepare_structures(calc_name=calc_name)

        rates = self.rates[self.rates["calc_name"] == rep_name]["rates"].values[0]
        print(f"rates: {rates}")
        print(rates)

        train_segs = self.train_segs[self.train_segs["calc_name"] == rep_name].copy()

        self.recalculate_dataset(traj=traj,
                                cr_bc_bh=cr_bc_bh,
                                dataset_name=rep_name,
                                segs=train_segs,
                                rates=rates,
                                train=True)

        expt_segs = self.segs[self.segs["calc_name"] == expt_name].copy()

        no_weight_BV = ([None], 0.35, 2.0)
        no_weight_name = "_".join(["no_weight", calc_name, str(rep)])

        self.recalculate_dataset(traj=traj,
                                cr_bc_bh=no_weight_BV,
                                dataset_name=no_weight_name,
                                segs=expt_segs,
                                rates=rates,
                                train=True)                                 



        val_segs = self.val_segs[self.val_segs["calc_name"] == val_name].copy()

        val_df = self.recalculate_dataset(traj=traj,
                                        cr_bc_bh=cr_bc_bh,
                                        dataset_name=val_name,
                                        segs=val_segs,
                                        rates=rates)

        test_segs = self.segs[self.segs["calc_name"] == expt_name].copy()


        test_df = self.recalculate_dataset(traj=traj,
                                        cr_bc_bh=cr_bc_bh,
                                        dataset_name=test_name,
                                        segs=test_segs,
                                        rates=rates)
        print(val_df)

        return val_df, test_df

    def write_data_split_PDB(self, calc_name, expt_name, rep):
        """
        Write out the PDB files for the train, val and test splits
        """
        rep_name = "_".join(["train", calc_name, str(rep)])
        val_name = "_".join(["val", calc_name, str(rep)])
        top, traj = self.prepare_structures(calc_name=calc_name)
        print("Topology", top)
        top.add_TopologyAttr('bfactors')
        # write out the PDB files for the train, val and test splits
        train_segs = self.train_segs[self.train_segs["calc_name"] == rep_name].copy()
        val_segs = self.val_segs[self.val_segs["calc_name"] == val_name].copy()


        train_segs = self.train_segs[self.train_segs["calc_name"] == rep_name].copy()
        val_segs = self.val_segs[self.val_segs["calc_name"] == val_name].copy()

        train_segs["residues"] = train_segs.apply(lambda x: list(range(x["ResStr"], x["ResEnd"]+1)), axis=1)
        val_segs["residues"] = val_segs.apply(lambda x: list(range(x["ResStr"], x["ResEnd"]+1)), axis=1)

        train_residues = np.concatenate(train_segs["residues"].to_numpy())
        val_residues = np.concatenate(val_segs["residues"].to_numpy())

        train_residues = np.unique(train_residues)-1
        val_residues = np.unique(val_residues)-1


        train_top = top.copy()
        val_top = top.copy()
        # assign bfactor to residues in the train and val splits

        train_bfactors = np.zeros(len(train_top.residues))
        val_bfactors = np.zeros(len(val_top.residues))

        train_bfactors[train_residues] = 1
        val_bfactors[val_residues] = 1

        for idx,_ in enumerate(train_bfactors):
            # pick residue and then assign bfactor to all atoms in the residue
            train_res = train_top.residues[idx]
            val_res = val_top.residues[idx]

            for atom in train_res.atoms:
                atom.bfactor = train_bfactors[idx]
            for atom in val_res.atoms:
                atom.bfactor = val_bfactors[idx]

        name = self.settings.name

        mode = self.settings.split_mode

        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_pdb_name = "_".join(["train", str(rep), name, mode ,time]) + ".pdb"
        val_pdb_name = "_".join(["val", str(rep), name, mode ,time]) + ".pdb"

        out_dir = os.path.join(self.settings.results_dir, name)

        train_pdb_path = os.path.join(out_dir, train_pdb_name)
        val_pdb_path = os.path.join(out_dir, val_pdb_name)

        os.makedirs(out_dir, exist_ok=True)

        print(f"Writing train PDB to {train_pdb_path}")
        train_top.atoms.write(train_pdb_path)
        print(f"Writing val PDB to {val_pdb_path}")
        val_top.atoms.write(val_pdb_path)
        



    def run_VDX(self, 
                calc_name: str=None, 
                expt_name: str=None, 
                mode: str=None, # not implemented yet
                n_reps: int=None, 
                weights: np.array=None,
                random_seeds: list=None):
        print("Running VDX loop")

        if mode is not None:
            self.settings.split_mode = mode
        if n_reps is None:
            n_reps = self.settings.replicates
        if random_seeds is None:
            random_seeds = [self.settings.random_seed+i for i in range(n_reps)]

        print(f"Random seeds: {random_seeds}")
        train_gammas = []
        val_gammas = []

        train_dfs = []
        val_dfs = []
        test_dfs = []
        ## change functions to export dataframes which are then passed into evaulate HDX
        for rep in range(1,n_reps+1):

            _, train_rep_name, val_rep_name = self.split_segments(seg_name=expt_name,
                                                                  calc_name=calc_name, 
                                                                  rep=rep, 
                                                                  random_seed=random_seeds[rep-1])

            # train HDX
            train_opt_gamma, train_df, cr_bc_bh = self.train_HDX(calc_name=calc_name, 
                                                       expt_name=expt_name, 
                                                       mode=mode, 
                                                       weights=weights,
                                                       rep=rep)
            train_gammas.append(train_opt_gamma)

            # validation HDX
            val_opt_gamma, val_df,test_df = self.validate_HDX(calc_name=calc_name,
                                                        expt_name=expt_name,
                                                        mode=mode,
                                                        rep=rep,
                                                        train_gamma=train_opt_gamma,
                                                        cr_bc_bh=cr_bc_bh)
            
            if self.settings.plot:
                self.write_data_split_PDB(calc_name=calc_name,
                                            expt_name=expt_name,
                                            rep=rep)
            


            val_gammas.append(val_opt_gamma)  

            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)

        print("Finished running VDX loop")

        # evaluate HDX train vs val - how do we actually compare both? I guess we just take the average across the reps - how do we account for peptides?
        try:
            print("Evaluating HDX")
            self.evaluate_HDX(train_dfs=train_dfs, 
                          val_dfs=val_dfs, 
                          expt_name=expt_name, 
                          calc_name=calc_name, 
                          test_dfs=test_dfs,
                          train_gammas=train_gammas, 
                          val_gammas=val_gammas,
                          n_reps=n_reps)
            print("Finished evaluating HDX")
        # plot optimal gamma distributions
        except UserWarning:
            print("Unable to evaluate HDX")
        finally:
            print(self.BV_constants)
            return train_dfs, val_dfs, train_gammas, val_gammas
        

    def run_benchmark_ensemble(self,
                                system: str=None,
                                times: list=None,
                                expt_name: str=None,
                                n_reps: int=None,
                                split_modes: list=['r', 's', 'R3', 'Sp'],
                                random_seeds: list=None,
                                hdx_path: str=None,
                                segs_path: str=None,
                                traj_paths: str=None,
                                weights: np.array=None,
                                RW: bool=False,
                                optimise: bool=True,
                                top_path: str=None
                                ):
        print(self.settings.gamma_range)
  


        name_mapping = {
            "r": "naive_random",
            "s": "naive_sequential",
            "R3": "k_sequence",
            "xR": "no_loops",
            "Sp": "res_neighbours",
            "SR": "stratified_space"
            }
        
        if n_reps is None:
            n_reps = self.settings.replicates
        else:
            self.settings.replicates = n_reps
        if random_seeds is None:
            random_seeds = [self.settings.random_seed+i for i in range(n_reps)]

        # raw_run_outputs = {}
        analysis_dumps = {}
        analysis_df = pd.DataFrame()
        names = []
        save_paths = []

        if times is not None:
            self.settings.times = times
            self.times = times
        settings = deepcopy(self.settings)
        if RW and optimise:
            settings.RW_do_reweighting = True
            settings.RW_do_params = False
            settings.gamma_range = (3, 4)
        if not RW:
            settings.RW_do_reweighting = False
            settings.RW_do_params = True
        if not optimise:
            settings.RW_do_reweighting = False
            settings.RW_do_params = False
            # settings.gamma_range = (3, 4)
        name = deepcopy(settings.name)
        print(f"Running benchmark for {name}")
        split_names = [f"{mode}_{name_mapping[mode]}" for mode in split_modes]
        names = [f"{name}_{split_name}" for split_name in split_names]
        settings.times = times
        settings.results_dir = os.path.join(settings.results_dir, system, "Benchmark")
        for idx, mode in enumerate(split_modes):
            settings.name = names[idx]
            # split_name = split_names[idx]
            print(f"Running {mode} split mode")
            settings.split_mode = mode
            _VDX = ValDXer(settings=settings)
            _VDX.settings.plot = False
            _VDX.load_HDX_data(HDX_path=hdx_path,
                                SEG_path=segs_path,
                                calc_name=expt_name)
            _VDX.load_structures(top_path=top_path,
                                traj_paths=traj_paths,
                                calc_name=system)
            _ = _VDX.run_VDX(calc_name=system,
                             weights=weights,
                            expt_name=expt_name,
                            random_seeds=random_seeds)
            # raw_run_outputs[split_name] = run_outputs # we dont need the raw outputs
            analysis_dump, df, name = _VDX.dump_analysis()
            save_path = _VDX.save_experiment()
            print("Analysis Dump", analysis_dump)
            analysis_dumps.update(analysis_dump)
            analysis_df = pd.concat([analysis_df, df], ignore_index=True)
            names.append(name)
            save_paths.append(save_path)
        
        print("Concatenated",analysis_dumps)

        combined_analysis_dump = {}
        # repack the outputs into concatentated dataframes for each key
        key0 = names[0]
        print(analysis_dumps.keys())
        print()
        for key in analysis_dumps[key0].keys():
            test_dump = analysis_dumps[key0][key]
            print(key)
            print(type(test_dump))

            # print(test_dump)
            if isinstance(test_dump, pd.DataFrame):
                print("DF found")
                # print(key)
                # print(test_dump)
                combined_analysis_dump[key] = pd.concat([analysis_dumps[name][key] for name in names], ignore_index=True)

        combined_analysis_dump["analysis_df"] = analysis_df
        combined_analysis_dump["save_paths"] = {name: save_path for name, save_path in zip(names, save_paths)}

        print("Adding info to analysis dump")

        for key in combined_analysis_dump.keys():
            dump = combined_analysis_dump[key]
            print(key)
            print(type(dump))

            if isinstance(dump, pd.DataFrame):
                print("df found")
                # print(dump)
                try:
                    dump["name_name"] = dump["name"]+"_"+dump["calc_name"]
                    dump["protein"] = [i.split("_")[3] if len(i.split("_")) > 3 else "Experiment" for i in dump["name"]]
                    # dump["split_type"] = [i.split("_")[0] for i in dump["name"]]
                    dump["dataset"] = dump["calc_name"].apply(lambda x: x.split("_")[0])
                    dump["class"] = dump["dataset"] + "_" + dump["split_type"]
                except:
                    print("Failed to add info to analysis dump")
                    print(dump)
                    raise ValueError("Failed to add info to analysis dump")
                    # break
 
        # plot the results
        MSE_df = combined_analysis_dump["analysis_df"]
        print("MSE df")
        print(MSE_df)
        # MSE for each protein
    
        # split_benchmark_plot_MSE_by_protein(MSE_df)

        # # MSE for each split mode by protein
        # split_benchmark_plot_MSE_by_split_protein(MSE_df)
                
        # split_benchmark_plot_MSE_by_protein_split(MSE_df)

        if self.settings.save_figs:
            save_dir = os.path.join(settings.plot_dir, system, "Benchmark")
            try:
                os.removedirs(save_dir)
            except:
                pass
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None


        split_benchmark_plot_MSE_by_split(MSE_df,
                                          save=settings.save_figs,
                                        save_dir=save_dir,)


        if not RW:
            # BV Constants
            BV_df = combined_analysis_dump["BV_constants"]
            print(BV_df)
            # BV Constants difference by protein
            # split_benchmark_BV_boxplot_by_protein(BV_df)
            # BV Constants difference by split mode
            split_benchmark_BV_boxplot_by_split_type(BV_df,
                                                     save=settings.save_figs,
                                                        save_dir=save_dir)
            # # BV Constants by split mode
            # split_benchmark_BV_boxplot_by_split_type_by_protein(BV_df)
            # # BV Constants by protein
            # split_benchmark_BV_boxplot_by_protein_by_split_type(BV_df)
                    
            

        return combined_analysis_dump, names, save_paths



    def run_refine_ensemble(self,
                            system: str=None,
                            times: list=None,
                            expt_name: str=None,
                            n_reps: int=None,
                            split_mode: str='R3',
                            random_seeds: list=None,
                            hdx_path: str=None,
                            segs_path: str=None,
                            traj_paths: list=None,
                            top_path: str=None,
                            modal_cluster: bool=False):
        ### Currently only doing the mean - update to take the mode for cluster frac2

        self.settings.split_mode = split_mode
        if times is not None:
            self.settings.times = times
            self.times = times

        u = mda.Universe(top_path, *traj_paths)

        cluster_frames, iniweights, pca = cluster_traj_by_density(u, cluster_frac1=self.settings.cluster_frac1)
        

        # save clustered universe 
        clustered_traj_name = "_".join([system, "clustered", "cfrac1", str(self.settings.cluster_frac1), ".xtc"])
        clustered_traj_path = os.path.join(self.settings.data_dir, self.settings.name, clustered_traj_name)

        with mda.Writer(clustered_traj_path, u.trajectory.n_frames) as W:
            for ts in u.trajectory[cluster_frames]:
                W.write(u)
        
        clustered_universe = mda.Universe(top_path, clustered_traj_path)

        assert clustered_universe.trajectory.n_frames == len(cluster_frames)



        _ = self.run_benchmark_ensemble(system=system+"_clustered",
                                            times=times,
                                            expt_name=expt_name,
                                            n_reps=n_reps,
                                            random_seeds=random_seeds,
                                            hdx_path=hdx_path,
                                            optimise=False,
                                            segs_path=segs_path,
                                            traj_paths=[clustered_traj_path],
                                            weights=iniweights,
                                            top_path=top_path)


        # reweight trajectory
                
        settings = deepcopy(self.settings)
        settings.RW_do_reweighting = True
        settings.RW_do_params = False
        settings.name = "_".join([settings.name, system, "refine"])
        
        _VDX = ValDXer(settings=settings)

        _VDX.load_HDX_data(HDX_path=hdx_path,
                            SEG_path=segs_path,
                            calc_name=expt_name,
                            experimental=True)
        _VDX.load_structures(top_path=top_path,
                            traj_paths=[clustered_traj_path],
                            calc_name=system+"_clustered"+"_refine")
        _VDX.settings.random_initialisation = True
        _ = _VDX.run_VDX(calc_name=system+"_clustered"+"_refine",
                        expt_name=expt_name,
                        n_reps=n_reps)


        final_weights = _VDX.weights["weights"].values
        final_weights = np.array([np.array(w) for w in final_weights])
        # average weights
        avg_weights = np.mean(final_weights, axis=0)

        clustered_universe = mda.Universe(top_path, clustered_traj_path)
        # recluster to cluster frac2
        recluster_frames, final_cluster2_weights = recluster_traj_by_weight(clustered_universe=clustered_universe, cluster_weights=avg_weights, cluster_frac2=self.settings.cluster_frac2, pca_operator=pca)

        # print("Weights", final_cluster2_weights)
        # print("Weights shape", final_cluster2_weights.shape)
        # print("Weights sum", np.sum(final_cluster2_weights))


        reclustered_traj_name = "_".join([system, "reclustered", "cfrac2", str(self.settings.cluster_frac2), ".xtc"])
        reclustered_traj_path = os.path.join(self.settings.data_dir, self.settings.name, reclustered_traj_name)


        with mda.Writer(reclustered_traj_path, u.trajectory.n_frames) as W:
            for ts in u.trajectory[recluster_frames]:
                W.write(u)
        
        reclustered_universe = mda.Universe(top_path, reclustered_traj_path)

        assert reclustered_universe.trajectory.n_frames == len(recluster_frames)

        self.settings.random_initialisation = False

        _ = self.run_benchmark_ensemble(system=system+"_reclustered",
                                            times=times,
                                            expt_name=expt_name,
                                            n_reps=n_reps,
                                            random_seeds=random_seeds,
                                            hdx_path=hdx_path,
                                            optimise=False,
                                            segs_path=segs_path,
                                            traj_paths=[reclustered_traj_path],
                                            weights=final_cluster2_weights,
                                            top_path=top_path)



        # run BV Benchmark ensemble
        return self.run_benchmark_ensemble(system=system+"_reclustered-BVoptimised",
                                            times=times,
                                            expt_name=expt_name,
                                            n_reps=n_reps,
                                            random_seeds=random_seeds,
                                            hdx_path=hdx_path,
                                            RW=False,
                                            segs_path=segs_path,
                                            traj_paths=[reclustered_traj_path],
                                            weights=final_cluster2_weights,
                                            top_path=top_path)
        

    
    def train_HDX(self, 
                  calc_name: str=None, 
                  expt_name: str=None, 
                  mode: str=None, 
                  weights: np.array=None,
                  rep: int=None):
        ### need to rethink how to do train - val split for the names - each train rep needs to be in its own folder - reweighting uses an entire directory

        # train_opt_gammas = []

        # for rep in range(n_reps):

        rep_df, _ = self.predict_HDX(calc_name=calc_name, 
                                     rep=rep, 
                                     train=True)

        # add df to HDX_data

        gamma, df, cr_bc_bh = self.reweight_HDX(expt_name=expt_name, 
                                      calc_name=calc_name, 
                                      train=True, 
                                      weights=weights,
                                      rep=rep)

        self.HDX_data = pd.concat([self.HDX_data, df], ignore_index=True)
        # train_opt_gammas.append(gamma)

        # mean_opt_gamma = np.mean(train_opt_gammas)

        return gamma, df, cr_bc_bh

    def validate_HDX(self, 
                     calc_name: str=None, 
                     expt_name: str=None, 
                     mode: str=None, 
                     rep: int=None, 
                     train_gamma: float=None,
                     cr_bc_bh=None):
        # compare both training and validation data to the experimental data
        # show the averages and the distributions of the errors

        val_df, test_df = self.recalculate_test_and_val(cr_bc_bh=cr_bc_bh,
                                                        calc_name=calc_name,
                                                        expt_name=expt_name,
                                                        rep=rep)
        # plot in evaluate_HDX

        return train_gamma, val_df, test_df
    

    def evaluate_HDX(self, 
                     train_dfs: List[pd.DataFrame], 
                     val_dfs: List[pd.DataFrame], 
                     test_dfs: List[pd.DataFrame],
                     data: pd.DataFrame=None, 
                     expt_name: str=None, 
                     calc_name: str=None, 
                     mode: str=None, 
                     train_gammas: float=None, 
                     val_gammas: float=None, 
                     n_reps: int=None):
        

    

        if self.settings.save_figs:

            save_dir = os.path.join(self.settings.plot_dir, calc_name, 'Evaluate')
            try:
                os.removedirs(save_dir)
            except:
                pass
            os.makedirs(save_dir, exist_ok=True)

        else:
            save_dir = None

        times = self.settings.times

        if self.settings.plot: ###
            print("plotting gamma distributions")
            plot_gamma_distribution(calc_name=calc_name, 
                                    train_gammas=train_gammas, 
                                    val_gammas=val_gammas,
                                    save=self.settings.save_figs,
                                    save_dir=save_dir)

        # plot the individual runs from train and val
        train_rep_names = ["_".join(["train", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
        val_rep_names = ["_".join(["val", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
        # test_rep_names = ["_".join(["test", calc_name, str(rep)]) for rep in range(1,n_reps+1)]
        print(train_rep_names)
        print(val_rep_names)
        if self.settings.plot is None:

            args = [expt_name, *train_rep_names]
            print("plotting dfracs compare for train")
            plot_dfracs_compare(args, 
                                data=self.HDX_data, 
                                times=self.settings.times)

            args = [expt_name, *val_rep_names]
            print("plotting dfracs compare for val")
            plot_dfracs_compare(args, 
                                data=self.HDX_data, 
                                times=self.settings.times)

        # args = [expt_name, *test_rep_names]
        # plot_dfracs_compare(args, 
        #                     data=self.HDX_data, 
        #                     times=self.settings.times)

        # 

        print("Restoring trainval peptide numbers")
        expt_segs = self.segs[self.segs["calc_name"] == expt_name].copy()
        merge_df = restore_trainval_peptide_nos(calc_name=calc_name,
                                    expt_name=expt_name,
                                    train_dfs=train_dfs,
                                    val_dfs=val_dfs,
                                    # test_dfs=test_dfs,
                                    train_segs=self.train_segs,
                                    val_segs=self.val_segs,
                                    n_reps=n_reps,
                                    times=self.settings.times,
                                    expt_segs=expt_segs)

        print("Adding experimental data in to merge df")
        expt_df = self.HDX_data[self.HDX_data["calc_name"] == expt_name]
        merge_df = pd.concat([expt_df, merge_df], ignore_index=True)
        name = self.settings.name
        print("dumping data")
        data_to_dump = {
            "train_dfs": train_dfs,
            "val_dfs": val_dfs,
            "test_dfs": test_dfs,
            "expt_df": expt_df,
            "merge_df": merge_df,
            "expt_segs": expt_segs,
            "train_segs": self.train_segs,
            "val_segs": self.val_segs,
            "n_reps": n_reps,
            "times": self.settings.times,
            "expt_name": expt_name,
            "calc_name": calc_name,
            "train_rep_names": train_rep_names,
            "val_rep_names": val_rep_names,
            # "test_rep_names": test_rep_names,
            "HDX_data": self.HDX_data,
            "train_gammas": train_gammas,
            "val_gammas": val_gammas,
            "weights": self.weights,
            "BV_constants": self.BV_constants,
            "LogPfs": self.LogPfs,
        }
        # add to dictionary
        self.analysis_dump[name] = data_to_dump
        print("dumped data")
        ic(self.analysis_dump)
        # print(merge_df)
        args = [expt_name, *train_rep_names,  *val_rep_names]

        if self.settings.plot:
            try:
                print("plotting dfracs compare for merge_df")
                plot_dfracs_compare(args, 
                                data=merge_df, 
                                times=self.settings.times,
                                save=self.settings.save_figs,
                                save_dir=save_dir)
            except UserWarning:
                print("Unable to plot compare plot for merge_df")
            ####


            try:    
                print("plotting dfracs compare abs for merge_df")
                plot_paired_errors(args,
                                data=merge_df, 
                                times=self.settings.times,
                                save=self.settings.save_figs,
                                save_dir=save_dir)
            except UserWarning:
                print("Unable to plot paired errors for merge_df")


        top_path = self.paths.loc[self.paths["calc_name"] == calc_name, "top"].dropna().values[0]
        top = mda.Universe(top_path)

        expt_names = [expt_name] * n_reps
        train_names = train_rep_names
        val_names = val_rep_names
        
        # plot_heatmap_trainval_compare(expt_names=expt_names, 
        #                             train_names=train_names, 
        #                             val_names=val_names, 
        #                             expt_segs=expt_segs,
        #                             data=merge_df, 
        #                             times=self.settings.times, 
        #                             top=top)

        # plot_heatmap_trainval_compare_error(expt_names=expt_names, 
        #                             train_names=train_names, 
        #                             val_names=val_names, 
        #                             expt_segs=expt_segs,
        #                             data=merge_df, 
        #                             times=self.settings.times, 
        #                             top=top)
        print("plotting R agreement")
        plot_df = plot_R_agreement_trainval(expt_name=expt_name, 
                                train_names=train_names, 
                                val_names=val_names, 
                                expt_segs=expt_segs,
                                data=merge_df, 
                                times=self.settings.times, 
                                top=top,
                                save=self.settings.save_figs,
                                save_dir=save_dir)
        print("concat plot_df")
        self.analysis = pd.concat([self.analysis, plot_df], ignore_index=True)
        # return

        # Currently df contains values for the peptides in each train/val split 
        # we need to add nan values to the peptides which are not present in either split
        
        # first create df with all peptides
        print("plotting nan_df")
        nan_df = add_nan_values(calc_name=calc_name,
                                merge_df=merge_df,
                                n_reps=n_reps,
                                times=self.settings.times,
                                expt_segs=expt_segs)
            
        print("nan_df")
        ic(nan_df)

        # add expt_df to nan_df
        nan_df = pd.concat([nan_df, expt_df], ignore_index=True)
        print("nan_df + expt_df")
        ic(nan_df)



            # args = [expt_name, *train_rep_names,  *val_rep_names]
            # # this doesnt work - we need to either line up the peptides or just plot the averages
            # try:
            #     plot_dfracs_compare(args, 
            #                     data=nan_df, 
            #                     times=self.settings.times)
            # except UserWarning:
            #     print("Unable to plot compare plot for nan_df")
            # ####

        # plot abs error for train and val
        if self.settings.plot:
            try:
                print("plotting abs error for nan_df")
                plot_dfracs_compare_abs(args, 
                                data=nan_df, 
                                times=self.settings.times,
                                save=self.settings.save_figs,
                                save_dir=save_dir)
            except UserWarning:
                print("Unable to plot compare plot for nan_df")
        ####
        # plot MSE for train and val
        try:
            print("plotting MSE for nan_df")
            plot_df = plot_dfracs_compare_MSE(args, 
                            data=nan_df, 
                            times=self.settings.times,
                            save=self.settings.save_figs,
                            save_dir=save_dir)
            
            self.analysis = pd.concat([self.analysis, plot_df], ignore_index=True)
        except UserWarning:
            print("Unable to plot compare plot for nan_df")
        ####

        if self.settings.plot:
            print("plotting AVG df")
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
                print("plotting dfracs compare for avg_df")
                plot_dfracs_compare(args,
                                data=avg_df, 
                                times=self.settings.times,
                                save=self.settings.save_figs,
                                save_dir=save_dir)
            except UserWarning:
                print("Unable to plot compare plot for avg_df")
            # plot train and val averages against expt data with paired plot
            try:
                print("plotting paired errors for avg_df")
                plot_paired_errors(args, 
                            data=avg_df, 
                            times=self.settings.times,
                                save=self.settings.save_figs,
                                save_dir=save_dir)
            except UserWarning:
                print("Unable to plot paired errors for avg_df")

            # plot_heatmap_trainval_compare(expt_names=[expt_name], 
            #                             train_names=[train_avg_name], 
            #                             val_names=[val_avg_name], 
            #                             expt_segs=expt_segs,
            #                             data=avg_df, 
            #                             times=self.settings.times, 
            #                             top=top)

            # plot_heatmap_trainval_compare_error(expt_names=[expt_name], 
            #                             train_names=[train_avg_name], 
            #                             val_names=[val_avg_name], 
            #                             expt_segs=expt_segs,
            #                             data=avg_df, 
            #                             times=self.settings.times, 
            #                             top=top)

    def prepare_intrinsic_rates(self, calc_name: str=None):
        """
        Reads Intrinsic rates from HDXer output file and adds to rates df.
        """
        rates_path = self.paths.loc[self.paths["calc_name"] == calc_name, "int_rates"].dropna().values[0]
        rates_dict = kints_to_dict(rates_path)
        rates_to_add = pd.DataFrame({"rates": [rates_dict], "calc_name": [calc_name]})
        self.rates = pd.concat([self.rates, rates_to_add], ignore_index=True)



# rates are required for reweighting??? no they 
# should add a method to add rates to the df
    def load_intrinsic_rates(self, path, calc_name: str=None, experimental=False):
        """
        Load intrinsic rates: intrinsic rates from HDXer.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")

        paths_to_add = pd.DataFrame({"int_rates": [path], "calc_name": [calc_name], "experimental": [experimental]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True,)

        self.prepare_intrinsic_rates(calc_name=calc_name)
     
    def dump_analysis(self):
        """
        Dump analysis to csv file in results dir
        Also returns useful data
        """
        name = self.settings.name
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_name = time + "_analysis.csv"
        csv_dir = os.path.join(self.settings.results_dir, name)
        csv_path = os.path.join(csv_dir, csv_name)
        os.makedirs(csv_dir, exist_ok=True)
        self.analysis["name"] = [name]*len(self.analysis)
        self.analysis.to_csv(csv_path, index=False)
        print(f"Analysis dumped to {csv_path}")

        print(self.analysis_dump.keys())
        print(self.analysis_dump[name].keys())

        assert self.settings.split_mode is not None

        self.analysis["name"] = [name]*len(self.analysis)
        self.analysis["split_type"] = [self.settings.split_mode]*len(self.analysis)

        for key in self.analysis_dump[name].keys():
            dump = self.analysis_dump[name][key]
            print(f"Key: {key}")
            print(type(dump))
            if isinstance(dump, pd.DataFrame):
                # print("dump", dump)
                print(f"Adding {name} to df {key}")
                dump["name"] = [name]*len(dump)
                dump["split_type"] = [self.settings.split_mode]*len(dump)

        return self.analysis_dump, self.analysis, name
