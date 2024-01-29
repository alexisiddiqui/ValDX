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
        self.weights = pd.DataFrame()
        self.BV_constants = pd.DataFrame()
        self.test_HDX_data = pd.DataFrame()

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
                       mode: str=None, 
                       random_seed: int=None, 
                       train_frac: float=None, 
                       rep: int=None):
        """
        splits segments into train and validation sets
        various modes:
        r - random split of sequences (default)
        ### Not implemented yet
        s - split by N-terminal and C-terminal
        x - spatial split across Xture (alpha vs beta)
        X - spatial split across Xture (loops vs structured)
        R - Redundancy aware split
        ###
        seg_name is the name of the segments dir to split loaded by load_HDX -> prepare_HDX_data
        calc_name is the name of the calculation that the segments are being used for
        """
        if random_seed is None:
            random_seed = self.settings.random_seed
        if train_frac is None:
            train_frac = self.settings.train_frac
        # if seg_name is None:
        #     seg_name = calc_name

        rep_name = "_".join([calc_name, str(rep)])
        train_rep_name = "_".join(["train", rep_name])
        val_rep_name = "_".join(["val", rep_name])
        self.segs = self.segs.loc[self.segs['calc_name'] == seg_name].sort_values(by=['ResStr', 'ResEnd'])

        if mode is None:
            mode = self.settings.split_mode

        if mode is 'r':
            print(f"Randomly splitting segments for {calc_name} with random seed {random_seed} and train fraction {train_frac}")
            train_segs = self.segs.loc[self.segs['calc_name'] == seg_name].sample(frac=train_frac, random_state=random_seed)
            val_segs = self.segs.loc[self.segs['calc_name'] == seg_name].drop(train_segs.index)
        elif mode is 's':
            print(f"Splitting segments for {calc_name} by N-terminal and C-terminal")
            #
            no_segs = len(self.segs.loc[self.segs['calc_name'] == seg_name])
            no_segs = int(no_segs/2)
            # select first or second half basedd on random seed odd vs even
            if random_seed % 2 == 0:
                no_segs *= -1
            # make srue to order by ResStr and then ResEnd
            train_segs = self.segs.loc[self.segs['calc_name'] == seg_name].iloc[:no_segs]
            val_segs = self.segs.loc[self.segs['calc_name'] == seg_name].iloc[no_segs:]
        elif mode == 'x':
            raise NotImplementedError
        elif mode == 'X':
            raise NotImplementedError
        elif mode == 'R':
            segs = self.segs.copy()
            print(f"Splitting segments for {calc_name} by redundancy")
            segs = segs.loc[segs['calc_name'] == seg_name].copy()
            no_segs = len(segs)
            segs['ResNums'] = segs.apply(lambda row: np.arange(row['ResStr'], row['ResEnd']+1), axis=1)
            segs = segs.explode('ResNums')
            segs = segs.groupby('ResNums').value_counts().reset_index(name='counts')
            # sort by counts
            segs = segs.sort_values(by=['counts','ResNums'], ascending=[False, True])
            # get list of all counts
            counts = segs['counts'].unique()
            # sort descending
            counts = np.sort(counts)[::-1]

            train_peptides = np.array([])
            val_peptides = np.array([])

            # single_peptide_no = segs.loc[segs['counts'] == 1, 'peptide'].sample(1).values

            # # append single_peptide_no to val_peptides
            # val_peptides = np.append(val_peptides, single_peptide_no[0]).flatten()
            # # drop single_peptide_no from segs
            #   # Remove single_peptide_no from segs
            # print(segs)
            # peptide_to_remove = single_peptide_no[0]
            # segs = segs.loc[segs['peptide'] != peptide_to_remove]
            # print(segs)

            # for c in counts:
            #     if len(counts) == 1:
            #     # if redundant peptides have been selected split them so that train and val have the correct ratio in the end
            #         remaining = len(segs.loc[segs['counts'] == c])

            #         train_no = (remaining * train_frac) - len(train_peptides)

            #         train_peps = segs.loc[segs['counts'] == c, 'peptide'].sample(int(train_no)).values
            #         # make sure peptides are unique
            #         train_peps = np.unique(train_peps)
            #         train_peptides = np.append(train_peptides, train_peps).flatten()

            #         # val peptides are the rest
            #         val_peps = segs.loc[segs['counts'] == c, 'peptide'].drop(train_peps).values
            #         train_peptides = np.append(train_peptides, val_peps).flatten()

            #     if len(train_peptides) < len(val_peptides):
            #         peptides_to_add = train_peptides
            #     else:
            #         peptides_to_add = val_peptides

            #     peps = segs.loc[segs['counts'] == c, 'peptide'].to_numpy()
            #     peps = np.unique(peps)
            #     peptides_to_add = np.append(peptides_to_add, peps).flatten()
            #     # drop from segs
            #     segs = segs.loc[~segs['peptide'].isin(peps)]
            #     # remove from counts
            #     counts = counts[counts != c]



            # # assert that peptides that train and val peptides do not overlap
            # overlap = np.intersect1d(train_peptides, val_peptides)
            # assert len(overlap) == 0, f"Train and Val peptides overlap: {overlap}"


            # print("Train frac: ", train_frac)
            # print("No Train peptides: ", len(train_peptides))
            # print("No Val peptides: ", len(val_peptides))
            # print("Final Train Frac: ", len(train_peptides)/(len(train_peptides)+len(val_peptides)))


            # self.train_segs = self.segs.loc[segs['peptide'].isin(train_peptides)]
            # self.val_segs = self.segs.loc[segs['peptide'].isin(val_peptides)]


            single_peptide_no = segs.loc[segs['counts'] == 1, 'peptide'].sample(1).values[0]
            # Set to be used for val peptides
            val_peptides = {single_peptide_no}
            # Drop single_peptide_no from segs
            segs = segs.loc[segs['peptide'] != single_peptide_no]

            # Iterate over unique counts, starting with the highest
            for count in segs['counts'].unique()[::-1]:
                peptides_with_count = segs[segs['counts'] == count]['peptide'].unique()
                
                # If it's the last count level, handle the remaining peptides
                if count == 1:
                    remaining_train_count = int((no_segs-1)*train_frac) - len(train_peptides)
                    train_peptides_with_count = np.random.choice(peptides_with_count, remaining_train_count, replace=False)
                else:
                    train_peptides_with_count = np.random.choice(peptides_with_count, int(len(peptides_with_count) * train_frac), replace=False)
                
                # Update the sets
                val_peptides.update(set(peptides_with_count) - set(train_peptides_with_count))

            # Convert sets to lists
            train_peptides = list(set(segs['peptide']) - val_peptides)
            val_peptides = list(val_peptides)
            
            # Assert no overlap between train and validation peptides
            assert not set(train_peptides) & set(val_peptides), f"Train and Val peptides overlap."

            # Select the segments belonging to train and validation sets
            train_segs = self.segs[self.segs['peptide'].isin(train_peptides)]
            val_segs = self.segs[self.segs['peptide'].isin(val_peptides)]

            print("Train frac: ", train_frac)
            print("No Train peptides: ", len(train_peptides))
            print("No Val peptides: ", len(val_peptides))
            print("Final Train Frac: ", len(train_peptides) / (len(train_peptides) + len(val_peptides)))
        
        else:
            raise ValueError(f"Mode {mode} not implemented yet.")

        # calc_name_ext = "_".join([calc_name, str(rep)])
        # calc_name = "_".join([calc_name, calc_name_ext])
        
        train_segs["calc_name"] = train_rep_name
        val_segs["calc_name"] = val_rep_name

        print("train_segs")
        print(train_segs.head())

        # save to file
        train_segs_name = "_".join(["train",self.settings.segs_name[0], calc_name, self.settings.segs_name[1]])
        val_segs_name = "_".join(["val",self.settings.segs_name[0], calc_name, self.settings.segs_name[1]])
        _, train_segs_dir = self.generate_directory_structure(calc_name=train_rep_name, overwrite=True)
        _, val_segs_dir = self.generate_directory_structure(calc_name=val_rep_name, overwrite=True)

        train_segs_path = os.path.join(train_segs_dir, train_segs_name)
        val_segs_path = os.path.join(val_segs_dir, val_segs_name)

        train_segs["path"] = [train_segs_path]*len(train_segs)
        val_segs["path"] = [val_segs_path]*len(val_segs)
        
        # print("train_segs")
        # print(train_segs.head())

        segs_to_file(train_segs_path, train_segs)
        print(f"Saved train {rep_name} segments to {train_segs_path}")
        print(f"Train Peptide numbers: {np.sort(train_segs['peptide'].values)}")
        segs_to_file(val_segs_path, val_segs)
        print(f"Saved val {rep_name} segments to {val_segs_path}")
        print(f"Val Peptide numbers: {np.sort(val_segs['peptide'].values)}")

        self.train_segs = pd.concat([self.train_segs, train_segs], ignore_index=True)
        self.val_segs = pd.concat([self.val_segs, val_segs], ignore_index=True)

        
        ### split HDX data based on segments
        train_HDX_data = self.HDX_data.loc[self.HDX_data['calc_name'] == seg_name].iloc[train_segs.index].copy()
        val_HDX_data = self.HDX_data.loc[self.HDX_data['calc_name'] == seg_name].iloc[val_segs.index].copy()

        try:
            train_HDX_data = train_HDX_data.drop(columns=['ResStr','ResEnd'])
            val_HDX_data = val_HDX_data.drop(columns=['ResStr','ResEnd'])
        except:
            pass



        train_HDX_data["calc_name"] = [train_rep_name]*len(train_HDX_data)
        val_HDX_data["calc_name"] = [val_rep_name]*len(val_HDX_data)

        train_HDX_name = "_".join([train_rep_name, "expt_dfracs.dat"])
        val_HDX_name = "_".join([val_rep_name, "expt_dfracs.dat"])

        train_HDX_path = os.path.join(train_segs_dir, train_HDX_name)
        val_HDX_path = os.path.join(val_segs_dir, val_HDX_name)

        train_HDX_data["path"] = train_HDX_path
        val_HDX_data["path"] = val_HDX_path


        train_segs = train_segs.drop(columns=["calc_name", "path"]).copy()
        val_segs = val_segs.drop(columns=["calc_name", "path"]).copy()

        # merge segs and HDX on peptide number
        train_HDX_data = pd.merge(train_HDX_data, train_segs, on=['peptide'])
        val_HDX_data = pd.merge(val_HDX_data, val_segs, on=['peptide'])
        print("train_HDX_data")
        print(train_HDX_data)
        
        # return None


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
