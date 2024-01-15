### C

from abc import ABC, abstractmethod
from ValDX.VDX_Settings import Settings
from ValDX.helpful_funcs import segs_to_df, dfracs_to_df, segs_to_file, HDX_to_file
import pandas as pd
import numpy as np
import os
import time
import glob
import pickle
import shutil
import MDAnalysis as mda

class Experiment(ABC):
    def __init__(self,
                 settings: Settings, 
                 name=None):
        super().__init__()
        self.settings = settings
        self.times = self.settings.times

        if name is not None:
            self.name = name
        else:
            self.name = "ABC"
        self.calc_names = []
        self.paths = pd.DataFrame()
        self.rates = pd.DataFrame()
        self.structures = pd.DataFrame()
        self.segs = pd.DataFrame()
        self.train_segs = pd.DataFrame()
        self.val_segs = pd.DataFrame()
        self.HDX_data = pd.DataFrame()
        self.train_HDX_data = pd.DataFrame()
        self.val_HDX_data = pd.DataFrame()


    def prepare_HDX_data(self, 
                         calc_name: str=None): 
        """
        Prepares dataframes of HDX data from paths
        """
        print(f"Preparing HDX data for {calc_name}")
        try:
            path = self.paths.loc[self.paths['calc_name'] == calc_name]['HDX'].values[0]
            new_HDX_data = dfracs_to_df(path, 
                                        names=self.times)
            ### do we need this line??
            new_HDX_data['calc_name'] = calc_name
            ### 
            self.HDX_data = pd.concat([self.HDX_data, new_HDX_data], ignore_index=True)
        except:
            new_HDX_data = None
            raise Warning(f"No HDX data found for {calc_name}.")
        
        try:
            path = self.paths.loc[self.paths['calc_name'] == calc_name]['SEG'].values[0]
            new_segs_data = segs_to_df(path)
            ### do we need this line??
            new_segs_data['calc_name'] = calc_name
            ###
            self.segs = pd.concat([self.segs, new_segs_data], ignore_index=True)
        except:
            new_segs_data = None
            raise Warning(f"No residue segments found for {calc_name}.")
        
        return new_HDX_data, new_segs_data




    def split_segments(self, 
                       seg_name: str=None, 
                       calc_name: str=None, 
                       mode: str='r', 
                       random_seed: int=None, 
                       train_frac: float=None, 
                       rep: int=None):
        """
        splits segments into train and validation sets
        various modes:
        r - random split of sequences (default)
        ### Not implemented yet
        s - split by N-terminal and C-terminal
        x - spatial split across Xture
        R - Redundancy aware split
        ###
        seg_name is the name of the segments dir to split loaded by load_HDX -> prepare_HDX_data
        calc_name is the name of the calculation that the segments are being used for
        """
        if random_seed is None:
            random_seed = self.settings.random_seed
        if train_frac is None:
            train_frac = self.settings.train_frac
        if seg_name is None:
            seg_name = calc_name

        rep_name = "_".join([calc_name, str(rep)])
        train_rep_name = "_".join(["train", rep_name])
        val_rep_name = "_".join(["val", rep_name])


        if mode == 'r':
            train_segs = self.segs.loc[self.segs['calc_name'] == seg_name].sample(frac=train_frac, random_state=random_seed)
            val_segs = self.segs.loc[self.segs['calc_name'] == seg_name].drop(train_segs.index)
        else:
            raise ValueError(f"Mode {mode} not implemented yet.")

        # calc_name_ext = "_".join([calc_name, str(rep)])
        # calc_name = "_".join([calc_name, calc_name_ext])
        
        train_segs["calc_name"] = train_rep_name
        val_segs["calc_name"] = val_rep_name

        # save to file
        train_segs_name = "_".join(["train",self.settings.segs_name[0], calc_name, self.settings.segs_name[1]])
        val_segs_name = "_".join(["val",self.settings.segs_name[0], calc_name, self.settings.segs_name[1]])
        _, train_segs_dir = self.generate_directory_structure(calc_name=train_rep_name, overwrite=True)
        _, val_segs_dir = self.generate_directory_structure(calc_name=val_rep_name, overwrite=True)

        train_segs_path = os.path.join(train_segs_dir, train_segs_name)
        val_segs_path = os.path.join(val_segs_dir, val_segs_name)

        train_segs["path"] = train_segs_path
        val_segs["path"] = val_segs_path
        
        # print("train_segs")
        # print(train_segs.head())

        segs_to_file(train_segs_path, train_segs)
        print(f"Saved train {calc_name} segments to {train_segs_path}")
        print(f"Train Peptide numbers: {np.sort(train_segs['peptide'].values)}")
        segs_to_file(val_segs_path, val_segs)
        print(f"Saved val {calc_name} segments to {val_segs_path}")
        print(f"Val Peptide numbers: {np.sort(val_segs['peptide'].values)}")

        self.train_segs = pd.concat([self.train_segs, train_segs], ignore_index=True)
        self.val_segs = pd.concat([self.val_segs, val_segs], ignore_index=True)

        
        ### split HDX data based on segments
        train_HDX_data = self.HDX_data.loc[self.HDX_data['calc_name'] == seg_name].iloc[train_segs.index]
        val_HDX_data = self.HDX_data.loc[self.HDX_data['calc_name'] == seg_name].iloc[val_segs.index]

        train_HDX_data["calc_name"] = [train_rep_name]*len(train_HDX_data)
        val_HDX_data["calc_name"] = [val_rep_name]*len(val_HDX_data)

        train_HDX_name = "_".join([train_rep_name, "expt_dfracs.dat"])
        val_HDX_name = "_".join([val_rep_name, "expt_dfracs.dat"])

        train_HDX_path = os.path.join(train_segs_dir, train_HDX_name)
        val_HDX_path = os.path.join(val_segs_dir, val_HDX_name)

        train_HDX_data["path"] = train_HDX_path
        val_HDX_data["path"] = val_HDX_path

        # print(train_HDX_data.head())
        # print(val_HDX_data.path)

        # # sort by peptide number
        # train_HDX_data = train_HDX_data.sort_values(by=['peptide'])
        # val_HDX_data = val_HDX_data.sort_values(by=['peptide'])

        train_segs = train_segs.drop(columns=["calc_name", "path"]).copy()
        val_segs = val_segs.drop(columns=["calc_name", "path"]).copy()

        # merge segs and HDX on peptide number
        train_HDX_data = pd.merge(train_HDX_data, train_segs, on=['peptide'])
        val_HDX_data = pd.merge(val_HDX_data, val_segs, on=['peptide'])

        # reorder columns
        train_HDX_data = train_HDX_data[['ResStr','ResEnd', *self.settings.times, 'peptide', 'calc_name', 'path']]
        val_HDX_data = val_HDX_data[['ResStr','ResEnd', *self.settings.times, 'peptide', 'calc_name', 'path']]

        HDX_to_file(train_HDX_path, train_HDX_data)
        print(f"Saved train {calc_name} HDX data to {train_HDX_path}")
        HDX_to_file(val_HDX_path, val_HDX_data)
        print(f"Saved val {calc_name} HDX data to {val_HDX_path}")

        self.train_HDX_data = pd.concat([self.train_HDX_data, train_HDX_data], ignore_index=True)
        self.val_HDX_data = pd.concat([self.val_HDX_data, val_HDX_data], ignore_index=True)

        return calc_name, train_rep_name, val_rep_name

    def prepare_structures(self, 
                           calc_name: str=None):
        """
        Prepares MDA Universe object from topology and trajectory files.
        """
        # do we need iloc[0]??? only one top per calc name
        top_path = self.paths.loc[self.paths['calc_name'] == calc_name, 'top'].values[0]
        top = mda.Universe(top_path)
        # ensure that the traj unpacks all paths in the list
        traj_paths = self.paths.loc[self.paths['calc_name'] == calc_name, 'traj'].values[0]
        traj = mda.Universe(top_path,
                            *traj_paths)
                             
                            

        structures_to_add = pd.DataFrame({"top": [top], "traj": [traj], "calc_name": [calc_name]})

        self.structures = pd.concat([self.structures, structures_to_add], ignore_index=True)

        print(f"Structures loaded {calc_name}: ")
        print(f"{calc_name} Topology: {top}")
        print(f"{calc_name} Trajectory: {traj}")
        print(f"{calc_name} Traj: no frames {traj.trajectory.n_frames}")

        return top, traj


    def generate_directory_structure(self, calc_name: str=None, overwrite=False, gen_only=False):
        """
        Generates directory structure for the experiment.
        Used during init with no calc_name to generate the experiment directory. Overwrite = False.
        Used during predict HDX to gen_only the path. Overwrite = False.
        Used during split segements to create the train and val segments directories per replicate. Overwrite = True.
        """
        if calc_name is None:
            name = self.name
            exp_dir = os.path.join(self.settings.data_dir, name)

            exists = os.path.isdir(exp_dir)
            if overwrite:
                try:
                    os.removedirs(exp_dir)
                    print(f"Removing contents {exp_dir}")
                except:
                    pass
                try:
                    os.rmdir(exp_dir)
                    print(f"Removing contents {exp_dir}")
                except:
                    pass
                exp_dir = os.path.join(self.settings.data_dir, self.name)
                os.makedirs(exp_dir, exist_ok=True)

                return self.name, exp_dir
    
            count = 0
            while exists:
                name = self.name + str(count)
                print(f"Experiment name {self.name} already exists. Attempting to change name to {name}")
                count += 1
                exists = os.path.isdir(os.path.join(self.settings.data_dir, name))
            # when doesnt exist - set self.name to name
            self.name = name
            exp_dir = os.path.join(self.settings.data_dir, self.name)
            os.makedirs(exp_dir)

            return self.name, exp_dir

        elif calc_name is not None:
            name = calc_name
            calc_dir = os.path.join(self.settings.data_dir, self.name, name)

            exists = os.path.isdir(calc_dir)
            if overwrite and exists:
                shutil.rmtree(calc_dir)
                print(f"Removing contents {calc_dir}")
                # except:
                #     pass
    
            elif exists and not gen_only:
                raise ValueError(f"Calculation {calc_name} already exists. Please choose a different name.")

            calc_dir = os.path.join(self.settings.data_dir, self.name, calc_name)
            if exists and gen_only:
                return calc_name, calc_dir
            
            os.makedirs(calc_dir)
            return calc_name, calc_dir
            

    # @abstractmethod
    def prepare_config(self):
        """
        Prepares the configuration...for the environment setup.
        Includes HDXER env as well as the HDXER executable.
        """
        pass



    # @abstractmethod
    def save_experiment(self, save_name=None):
        """
        Writes the settings of the experiment (contents of class) to a pickle file. Logs???
        """
        if save_name is None:
            save_name = self.name
        unix_time = int(time.time())
        if save_name is not None:
            save_name = save_name+"_"+str(unix_time)+".pkl"
            save_path = os.path.join(self.settings.logs_dir, save_name)

            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
                print("Saving experiment to: ", save_path)
                return save_path

    # @abstractmethod
    # add something to select by name?
    def load_experiment(self, load_path=None, latest=False, idx=None):
        """
        Loads the settings of the experiment from a pickle file.
        """
        if load_path is not None:
            print("Attempting to load experiment from: ", load_path)
            with open(load_path, 'rb') as f:
                print("Loading experiment from: ", load_path)
                return pickle.load(f)
            # print("Loaded object type:", type(loaded_obj))

            # return deepcopy(loaded_obj)


        # If no explicit path is provided
        search_dir = self.settings.logs_dir
        print("Searching for experiment files in: ", search_dir)

        pkl_files = glob.glob(os.path.join(search_dir, "*.pkl"))
        print("Found files: ", pkl_files)

        if not pkl_files:
            print("No experiment files found.")
            raise FileNotFoundError
        pkl_files = sorted(pkl_files, key=os.path.getctime)
        if latest is True:
            print("Loading latest experiment.")
            file = pkl_files[-1]
        elif idx is not None:
            print(f"Loading {idx} experiment.")
            file = pkl_files[idx]
        else:
            print("Loading first experiment.")
            file = pkl_files[0]

        
        load_path = file    
        print("Loading experiment from: ", load_path)
        with open(load_path, 'rb') as f:
            return pickle.load(f)
