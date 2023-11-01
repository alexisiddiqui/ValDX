### Class to contain the HDX experiment

# inputs:
# settings
# experimental data
# matched topology and peptide-residue segments
# simulation MD data 

from ValDX.VDX_Settings import Settings
from ValDX.Experiment_ABC import Experiment
from ValDX.helpful_funcs import segs_to_df, dfracs_to_df, segs_to_file
from ValDX.HDX_plots import plot_lcurve
from HDXer.reweighting import MaxEnt

import pandas as pd
import time
import os 
import pickle
import glob
import subprocess

class ValDX(Experiment):
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

        if HDX_path is None:
            paths_to_add = pd.DataFrame({"SEG": [SEG_path], "calc_name": [calc_name], "experimental": [experimental]})
        elif SEG_path is None:
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], "calc_name": [calc_name], "experimental": [experimental]})
        elif HDX_path and SEG_path is not None:
            paths_to_add = pd.DataFrame({"HDX": [HDX_path], "SEG": [SEG_path], "calc_name": [calc_name], "experimental": [experimental]})
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
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")
        if train:
            rep_name = "_".join(["train", calc_name, str(rep)])
        else:
            rep_name = "_".join(["val", calc_name, str(rep)])

        out_dir = os.path.join(self.settings.data_dir, self.name, calc_name)

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

            self.load_intrinsic_rates(out_prefix + "_intrinsic_rates.dat", calc_name=calc_name)
            
            return df
        

    def reweight_HDX(self, expt_name: str=None, calc_name: str=None, gamma_range: tuple=(2,10), train: bool=True, rep: int=None):
        """
        Reweight HDX data from previsously predicted HDX data performed by predict_HDX().
        """
        if expt_name or calc_name is None:
            raise ValueError("Please provide a Calculation AND Experimental name for the structures.")

        rep_name = "_".join(["train", calc_name, str(rep)])

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
                
            # plot L-curve return closest gamma value
            return plot_lcurve(calc_name=rep_name, RW_range=gamma_range, dir=predictHDX_dir, prefix=self.settings.RW_outprefix)
        

    def train_HDX(self, calc_name: str=None, expt_name: str=None, mode: str=None, n_reps: int=None, random_seeds: list=None):
        ### need to rethink how to do train - val split for the names - each train rep needs to be in its own folder - reweighting uses an entire directory

        if random_seeds is None:
            random_seeds = [i for i in range(self.settings.random_seed, self.settings.random_seed + n_reps)]

        train_name = "_".join(["train", calc_name])


        for rep in range(n_reps):
            seed = random_seeds[rep]

            rep_name = self.split_segments(calc_name=calc_name, random_seed=seed, train_frac=self.settings.train_frac, rep=rep)
            self.predict_HDX(calc_name=calc_name, mode=mode, rep=rep, train=True)
            self.reweight_HDX(expt_name=expt_name, calc_name=calc_name, train=True)






    def evaluate_HDX(self, calc_name: str=None, mode: str=None, n_reps: int=None, gamma: float=None):
        # compare both training and validation data to the experimental data
        # show the averages and the distributions of the errors
        pass

    ### Dont need to mess with this for now
    def load_intrinsic_rates(self, path, calc_name: str=None):
        """
        Load intrinsic rates: intrinsic rates from HDXer.
        """
        if calc_name is None:
            raise ValueError("Please provide a calculation name for the structures.")

        paths_to_add = pd.DataFrame({"int_rates": [path], "calc_name": [calc_name]})
        self.paths = pd.concat([self.paths, paths_to_add], ignore_index=True)

