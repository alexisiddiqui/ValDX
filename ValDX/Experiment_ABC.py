### C

from abc import ABC, abstractmethod
from ValDX.VDX_Settings import Settings
from ValDX.helpful_funcs import segs_to_df, dfracs_to_df, segs_to_file
import pandas as pd
import os
import MDAnalysis as mda

class Experiment(ABC):
    def __init__(self,settings: Settings, name=None):
        super().__init__(settings, name)
        self.settings = settings
        self.HDXer_path = self.settings.HDXer_path
        self.HDXer_env = self.settings.HDXer_env
        self.times = self.settings.times

        if name is not None:
            self.name = name
        else:
            self.name = "ABC"
        self.calc_names = []
        self.paths = pd.DataFrame()
        self.
        self.structures = pd.DataFrame()
        self.train_segs = pd.DataFrame()
        self.val_segs = pd.DataFrame()
        self.HDX_data = pd.DataFrame()

    def prepare_HDX_data(self, calc_name: str=None):
        """
        Prepares dataframes of HDX data from paths
        """
        try:
            new_HDX_data = dfracs_to_df(self.paths.loc[self.paths['calc_name'] == calc_name, 'HDX'], 
                                        names=self.times)
            ### do we need this line??
            new_HDX_data['calc_name'] = calc_name
            ### 
            self.HDX_data = pd.concat([self.HDX_data, new_HDX_data], ignore_index=True)
        except:
            new_HDX_data = None
            raise Warning(f"No HDX data found for {calc_name}.")
        
        try:
            new_segs_data = segs_to_df(self.paths.loc[self.paths['calc_name'] == calc_name, 'SEG'])
            ### do we need this line??
            new_segs_data['calc_name'] = calc_name
            ###
            self.train_segs = pd.concat([self.train_segs, new_segs_data], ignore_index=True)
            self.val_segs = pd.concat([self.val_segs, new_segs_data], ignore_index=True)
        except:
            new_segs_data = None
            raise Warning(f"No residue segments found for {calc_name}.")
        
        return new_HDX_data, new_segs_data

    def split_segments(self, calc_name: str=None, mode: str='r', random_seed: int=None, train_frac: float=None, rep: int=None):
        """
        splits segments into train and validation sets
        various modes:
        r - random split of sequences (default)
        ### Not implemented yet
        s - split by N-terminal and C-terminal
        x - spatial split across Xture
        R - Redundancy aware split
        ###
        """
        if random_seed is None:
            random_seed = self.settings.random_seed
        if train_frac is None:
            train_frac = self.settings.train_frac

        if mode == 'r':
            train_segs = self.train_segs.sample(frac=train_frac, random_state=random_seed, axis=0)
            val_segs = self.train_segs.drop(train_segs.index)
        else:
            raise ValueError(f"Mode {mode} not implemented yet.")

        calc_name_ext = "_".join([calc_name, str(rep)])
        calc_name = "_".join([calc_name, calc_name_ext])
        
        train_segs["calc_name"] = calc_name
        val_segs["calc_name"] = calc_name

        # save to file
        train_segs_name = "_".join(["train",self.settings.segs_name[0], calc_name_ext, self.settings.segs_name[1]])
        val_segs_name = "_".join(["val",self.settings.segs_name[0], calc_name_ext, self.settings.segs_name[1]])

        train_segs_path = os.path.join(self.settings.data_dir, self.name, calc_name, train_segs_name)
        val_segs_path = os.path.join(self.settings.data_dir, self.name, calc_name, val_segs_name)

        train_segs["path"] = train_segs_path
        val_segs["path"] = val_segs_path
        
        segs_to_file(train_segs_path, train_segs)
        print(f"Saved train {calc_name} segments to {train_segs_path}")
        segs_to_file(val_segs_path, val_segs)
        print(f"Saved val {calc_name} segments to {val_segs_path}")

        self.train_segs = pd.concat([self.train_segs, train_segs], ignore_index=True)
        self.val_segs = pd.concat([self.val_segs, val_segs], ignore_index=True)

        return calc_name

    def prepare_structures(self, calc_name: str=None):
        """
        Prepares MDA Universe object from topology and trajectory files.
        """
        # do we need iloc[0]??? only one top per calc name
        top = mda.Universe(self.structures.loc[self.structures['calc_name'] == calc_name, 'top'])
        # ensure that the traj unpacks all paths in the list
        traj = mda.Universe(topology=self.structures.loc[self.structures['calc_name'] == calc_name, 'top'],
                            trajectory=self.structures.loc[self.structures['calc_name'] == calc_name, 'traj']) 
                            

        structures_to_add = pd.DataFrame({"top": [top], "traj": [traj], "calc_name": [calc_name]})

        self.structures = pd.concat([self.structures, structures_to_add], ignore_index=True)

        print(f"Structures loaded {calc_name}: ")
        print(f"Topology: {top}")
        print(f"Trajectory: {traj}")

        return top, traj


    def generate_directory_structure(self, calc_name: str=None, overwrite=False):
        """
        Generates directory structure for the experiment. 
        This is for the outputs of the experiment.
        """
        if calc_name is None:
            name = self.name
            exp_dir = os.path.join(self.settings.data_dir, name)

            exists = os.path.isdir(exp_dir)
            if overwrite:
                os.removedirs(exp_dir)
                os.makedirs(exp_dir)
    
            count = 0
            while exists:
                name = self.name + str(count)
                print(f"Experiment name {self.name} already exists. Attempting to change name to {name}")

            # when doesnt exist - set self.name to name
            self.name = name
            exp_dir = os.path.join(self.settings.data_dir, self.name)
            os.makedirs(exp_dir)

            return self.name, exp_dir

        elif calc_name is not None:
            name = calc_name
            calc_dir = os.path.join(self.settings.data_dir, self.name, name)

            exists = os.path.isdir(calc_dir)
            if overwrite:
                os.removedirs(calc_dir)
                os.makedirs(calc_dir)
    
            elif exists:
                raise ValueError(f"Calculation {calc_name} already exists. Please choose a different name.")
            
            calc_dir = os.path.join(self.settings.data_dir, self.name, calc_name)
            os.makedirs(calc_dir)

            return calc_name, calc_dir
            

    @abstractmethod
    def prepare_config(self):
        """
        Prepares the configuration...for the environment setup.
        Includes HDXER env as well as the HDXER executable.
        """
        pass



    @abstractmethod
    def save_experiment(self):
        pass

    @abstractmethod
    def load_experiment(self):
        pass