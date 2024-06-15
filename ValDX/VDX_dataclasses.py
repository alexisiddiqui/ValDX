import pandas as pd
import time
import os
from icecream import ic
from dataclasses import dataclass
from typing import List, Tuple, Union
from ValDX.helpful_funcs import segs_to_df, PDB_to_DSSP
import numpy as np
import MDAnalysis as mda

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@dataclass
class AnalysisInfo:
    settings_name: str
    analysis_name: Union[str, List[str]]
    n_reps: int
    split_mode: str
    times: list
    calc_name: str = None
    expt_name: str = None
    system_name: str = None
    # benchmark: bool = False # not implemented yet

    def __post_init__(self):
        if self.system_name is None:
            self.system_name = self.settings_name

        if isinstance(self.analysis_name, list):
            self.analysis_name = "_".join(self.analysis_name)

        

@dataclass
class AnalysisData:
    train_dfs: List[pd.DataFrame]
    val_dfs: List[pd.DataFrame]
    expt_df: pd.DataFrame
    merge_df: pd.DataFrame
    expt_segs: pd.DataFrame
    train_segs: pd.DataFrame
    val_segs: pd.DataFrame
    train_rep_names: list
    val_rep_names: list
    HDX_data: pd.DataFrame
    train_gammas: List[float]
    val_gammas: List[float]
    weights: pd.DataFrame
    BV_constants: pd.DataFrame
    LogPfs: pd.DataFrame    
    analysis_df: pd.DataFrame = None
    info: AnalysisInfo = None


    dataframe_names = ["train_dfs", 
                       "val_dfs", 
                       "expt_df", 
                       "merge_df", 
                       "expt_segs", 
                       "train_segs", 
                       "val_segs", 
                       "HDX_data", 
                       "weights", 
                       "BV_constants", 
                       "LogPfs", 
                       "analysis_df"]


    def verify(self, info: AnalysisInfo=None):
        if info is not None:
            self.info = info
        assert isinstance(self.info, AnalysisInfo)
        self.name = self.info.system_name

        self.dataframes = {
            "train_dfs": self.train_dfs,
            "val_dfs": self.val_dfs,
            "expt_df": self.expt_df,
            "merge_df": self.merge_df,
            "expt_segs": self.expt_segs,
            "train_segs": self.train_segs,
            "val_segs": self.val_segs,
            "HDX_data": self.HDX_data,
            "weights": self.weights,
            "BV_constants": self.BV_constants,
            "LogPfs": self.LogPfs,
            "analysis_df": self.analysis_df
        }


        for key, df in self.dataframes.items():
            if isinstance(df, list):
                for idx, sub_df in enumerate(df):
                    print(self.info)
                    print(f"{key}, {idx} verifying...")
                    try:
                        self.dataframes[key][idx] = self.verify_dataframe(sub_df)
                    except:
                        print(f"{key}, {idx} failed")
                        print(sub_df)
                        raise ValueError(f"{key,idx} failed")

            else:
                print(self.info)
                print(f"{key} verifying...")
                try:
                    self.dataframes[key] = self.verify_dataframe(df)
                except:
                    print(f"{key} failed")
                    print(df)
                    raise ValueError(f"{key} failed")


    def verify_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # if "name" not in df.columns:
        df["name"] = [self.info.settings_name] * len(df)
        df["system"] = [self.info.system_name] * len(df)

        df["analysis"] = [self.info.analysis_name] * len(df)
        df["split_type"] = [self.info.split_mode] * len(df)

        df["protein"] = [i.split("_")[3] if len(i.split("_")) > 3 else "Experiment" for i in df["name"]]

        df["name_name"] = df["name"]+"_"+df["calc_name"]
        # if "calc_name" empty 
        if "calc_name" not in df.columns:
            df["calc_name"] = [self.info.calc_name] * len(df)

        df["dataset"] = df["calc_name"].apply(lambda x: x.split("_")[0])
        df["class"] =  df["dataset"] + "_" + df["split_type"]

        return df


def merge_AnalysisData_classes(data_list: List[AnalysisData])->dict:
    """
    Merge the dataframes in the AnalysisData classes into individual dataframes 
    one for each key in the AnalysisData class
    """
    print("Merging AnalysisData classes")
    keys = AnalysisData.dataframe_names
    print(keys)
    merged_dfs = {}
    for key in keys:
        _df = data_list[0].dataframes[key]
        if isinstance(_df, pd.DataFrame):
            df_list = [data.dataframes[key] for data in data_list]
            merged_dfs[key] = pd.concat(df_list, axis=0, ignore_index=True)
        elif isinstance(_df, list):
            #flatten all lists
            df_list = [df for data in data_list for df in data.dataframes[key]]
            merged_dfs[key] = pd.concat(df_list, axis=0,  ignore_index=True)

    return merged_dfs





class Segments():
    """
    takes in a dataframe of experimental hdx segments 
    and contains the values of the peptides
    """
    def __init__(self, segs_df: pd.DataFrame=None, segs_path:str=None, keys=['ResStr', 'ResEnd']):
        if segs_df is not None:
            self.segs_df = segs_df.copy()
        if segs_path is not None:
            self.segs_df = segs_to_df(segs_path)
        self.keys = keys
        if "peptide" not in self.segs_df.columns:
            self.segs_df["peptide"] = np.arange(len(self.segs_df))
        self.pep_nums = self.get_pep_nums(self.segs_df)
        assert len(self.segs_df["peptide"].unique()) == len(self.pep_nums), "Peptides are not unique"
        print(f"Segments class created with {len(self.pep_nums)} peptides")
        self.res_nums = self.get_resnums(self.segs_df)
        self.residues = self.get_residues(self.segs_df)
        self.peptides = self.get_peptides(self.segs_df)
        self.residue_centrality = self.get_residue_centrality(self.segs_df)
        self.peptide_centrality = self.get_peptide_centrality(self.segs_df)

    def get_resnums(self, df: pd.DataFrame, keys=None):
        """
        Get the residue numbers of the segments in the dataframe
        """
        if df is None:
            df = self.segs_df
        
        if keys is None:
            keys = self.keys
        res_nums = df.apply(lambda x: np.arange(x[keys[0]]+1, x[keys[1]]+1), 
                            axis=1).to_numpy()
        print(f"Resnumbers calculated for {len(res_nums)} segments")
        return res_nums
    
    def get_peptides(self, df: pd.DataFrame=None):
        """
        Get the peptides from the dataframe,
        creates a dictionary with the peptide number as the key and the residues as the values
        """
        if df is None:
            df = self.segs_df
        peptides = {pep: res for pep, res in zip(df["peptide"], 
                                                 self.get_resnums(df))}
        print(f"Peptides calculated for {len(peptides)} segments")
        return peptides
    
    def get_residues(self, df: pd.DataFrame=None):
        """
        Get the residues from the dataframe
        """
        if df is None:
            df = self.segs_df
        residues = np.unique(np.concatenate(self.get_resnums(df)))
        print(f"Residues calculated for {len(residues)} segments")
        return residues
    
    def get_pep_nums(self, df: pd.DataFrame=None):
        """
        Get the peptide numbers from the dataframe
        """
        if df is None:
            df = self.segs_df
        return df["peptide"].to_numpy()
        
    def get_residue_centrality(self, df: pd.DataFrame):
        """
        calculate the number of peptides that contain each residue
        """
        print("Calculating residue centrality")
        if df is None:
            df = self.segs_df

        res_nums = self.get_resnums(df)

        res_cent = {res: 0 for res in self.residues}

        for res in self.residues:
            for res_num in res_nums:
                if res in res_num:
                    res_cent[res] += 1

        return res_cent

    def get_peptide_centrality(self, df: pd.DataFrame):
        """
        calculate the mean residue centrality for each peptide
        """

        print("Calculating peptide centrality")
        if df is None:
            df = self.segs_df
        peptides = self.get_peptides(df)
        res_cent = self.get_residue_centrality(df)

        pep_cent = {pep: np.mean([res_cent[res] for res in res_nums]) 
                    for pep, res_nums in peptides.items()}
        
        print(f"Peptide centrality calculated for {len(pep_cent)} peptides")
        return pep_cent
    
    def update(self, df: pd.DataFrame=None):
        """
        Update the values of the class
        """
        if df is not None:
            self.segs_df = df.copy()
        self.residues = self.get_residues(self.segs_df)
        self.res_nums = self.get_resnums(self.segs_df)
        self.peptides = self.get_peptides(self.segs_df)
        self.pep_nums = self.get_pep_nums(self.segs_df)

        self.residue_centrality = self.get_residue_centrality(self.segs_df)
        self.peptide_centrality = self.get_peptide_centrality(self.segs_df)


    def df_from_peptides(self, peptides: dict):
        """
        Update the dataframe with the peptides and residues
        Peptide is the index of the dataframe
        """
        keys = self.keys
        print(f"Creating dataframe from peptides {peptides}")
        df = pd.DataFrame(columns=[keys[0], keys[1], "peptide"])
    
        for pep, res in peptides.items():
            df = pd.concat([df, pd.DataFrame({keys[0]: res[0], 
                                              keys[1]: res[-1], 
                                              "peptide": pep}, 
                                              index=[pep])])
        print("Dataframe created")
        print(df.head())
        return df
    
    def df_select_peptides(self, pep_nums: list):
        """
        Select the peptides from the dataframe
        """
        print(f"Selecting {pep_nums} peptides from the dataframe")
        return self.segs_df.loc[pep_nums].copy()
    
    def df_remove_peptides(self, pep_nums: list):
        """
        Remove the peptides from the dataframe
        """
        print(f"Removing {pep_nums} peptides from the dataframe")
        return self.segs_df.loc[~self.segs_df["peptide"].isin(pep_nums)].copy()
    
    def df_select_residues(self, residues: list):
        """
        Select peptides from the dataframe based on the residues
        """
        print(f"Selecting peptides with residues {residues}")
        unique_res = np.unique(residues)

        selected_peptides = []
        for pep, res in self.peptides.items():
            if len(np.intersect1d(res, unique_res)) > 0:
                selected_peptides.append(pep)
        return self.df_select_peptides(selected_peptides).copy()

    def df_remove_residues(self, residues: list):
        """
        Remove peptides from the dataframe based on the residues
        """
        print(f"Removing peptides with residues {residues}")
        unique_res = np.unique(residues)

        selected_peptides = []
        for pep, res in self.peptides.items():
            if len(np.intersect1d(res, unique_res)) == 0:
                selected_peptides.append(pep)
        return self.df_select_peptides(selected_peptides).copy()

    
    def select_segments(self, 
                        peptides: list=None, 
                        residues: list=None, 
                        new_segs_df: pd.DataFrame=None,
                        remove: bool=False):
        """
        Update the segments class with new peptides or residues
        Either remove or select the peptides or residues
        First updates based on peptides, then updates based on residues
        """
        if new_segs_df is None:
            new_segs_df = self.segs_df.copy()

        
        if remove:
            peptide_update_function = self.df_remove_peptides
            residue_update_function = self.df_remove_residues
        else:
            peptide_update_function = self.df_select_peptides
            residue_update_function = self.df_select_residues

        if peptides is not None:    
            new_segs_df = peptide_update_function(peptides)

        if residues is not None:
            new_segs_df =  residue_update_function(residues)

        print("Updating Segments")
        self.update(new_segs_df)


    def create_train_val_segs(self,
                               train_peps: np.ndarray,
                                 val_peps: np.ndarray):
        
        train_segs = self.df_select_peptides(list(train_peps))
        val_segs = self.df_select_peptides(list(val_peps))

        return train_segs, val_segs


class PeptideSplitter():
    """
    Split the peptides into different splits
    """
    def __init__(self, 
                 expt_segs_path: str=None, 
                 expt_segs_df: pd.DataFrame=None,
                 train_frac: float=0.5,
                 random_seed: int=None,
                 keys=['ResStr', 'ResEnd']):
        if expt_segs_df is None and expt_segs_path is not None:
            expt_segs_df = segs_to_df(expt_segs_path)
        self.expt_segs_df = expt_segs_df
        assert isinstance(self.expt_segs_df, pd.DataFrame), "expt_segs_df must be a pandas DataFrame"
        self.keys = keys
        self.train_frac = train_frac
        self.expt_segments = Segments(self.expt_segs_df, keys=self.keys)
        if random_seed is None:
            random_seed = int(time.time())
            print(f"Random seed not set, using {random_seed}")
        self.random_seed = random_seed
        np.random.seed(random_seed)
    

    def peptide_centrality_split(self, 
                                 split_fraction: float=None,
                                 handle_intersection: bool=False):
        """
        Splits the peptides based on the peptide centrality,
        random selection, weighted by the peptide centrality
        To implement: weighted k means clustering using peptide centrality
        """
        if split_fraction is None:
            split_fraction = self.train_frac
        print(f"Peptide centrality split with split_fraction {split_fraction}")
        print(f"Handle intersection {handle_intersection}")
        pep_cent = self.expt_segments.peptide_centrality
        expt_segs = self.expt_segments
        # Calculate the weights based on peptide centrality
        weights = np.array(list(pep_cent.values())).astype(float)
        weights /= np.sum(weights)
        
        # Randomly select peptides based on the weights
        val_peps = np.random.choice(list(pep_cent.keys()), 
                                      size=int(len(pep_cent) * (1-split_fraction)), 
                                      replace=False, 
                                      p=weights)
        
        # Get the remaining peptides as train peptides
        train_peps = np.setdiff1d(list(pep_cent.keys()), val_peps)

        if handle_intersection:
            raise NotImplementedError("Intersection handling not implemented yet")
        # TODO use kmeans of the peptide centrality to split the peptides
            train_peps, val_peps = drop_intersection(expt_segs, train_peps, val_peps)


        return train_peps, val_peps
        


    def drop_centrality(self, split_fraction: float=0.9, drop: bool=True):
        if drop:
            _, val_peps = self.peptide_centrality_split(split_fraction=split_fraction,
                                                        handle_intersection=False)
            print(f"Dropping {len(val_peps)} peptides based on centrality")
            expt_segments = self.expt_segments.df_remove_peptides(list(val_peps))
        else:
            expt_segments = self.expt_segments.segs_df.copy()

        return expt_segments
    
    def validate_split(self, train_peps: np.ndarray, val_peps: np.ndarray, lower_min_pep_threshold: float=0.1):
        """Check that the training and validation peptides are valid
        valid criteria:
        - train and val peptides contain above the minimum threshold of peptides
        - train and val peptides are unique

        """

        if any([
            len(train_peps) < lower_min_pep_threshold * len(self.expt_segments.pep_nums),
            len(val_peps) < lower_min_pep_threshold * len(self.expt_segments.pep_nums),
            len(np.intersect1d(train_peps, val_peps)) > 0
        ]):
            # change the random seed
            print("Split not valid")

            randomseed = self.random_seed + 1
            print(f"New random seed: {randomseed}")
            np.random.seed(randomseed)
            return False

        return True

    def random_split(self, 
                     train_frac: float=None,
                     drop_centrality: bool=True,
                     drop: bool=False,
                     hard_intersection: bool=False):
        """
        Splits the peptides randomly based on the train_frac
        Optionally drops the highest centrality peptides
        """
        if train_frac is None:
            train_frac = self.train_frac
        print(f"Random split with train_frac {train_frac} and drop_centrality {drop_centrality}")
        print(f"Drop {drop} and hard_intersection {hard_intersection}")
        expt_segments = self.drop_centrality(drop=drop_centrality)
        new_expt_segs = Segments(expt_segments, keys=self.keys)

        train_peps = np.random.choice(expt_segments["peptide"].to_numpy(), 
                                      size=int(len(expt_segments) * train_frac), 
                                      replace=False)
        val_peps = np.setdiff1d(expt_segments["peptide"].to_numpy(), train_peps)

        if drop:
            return drop_intersection(new_expt_segs, train_peps, val_peps, hard=hard_intersection)
        else:
            print(f"Train peptides: {train_peps}")
            print(f"Val peptides: {val_peps}")
            print(f"Train %: {len(train_peps)/(len(train_peps)+len(val_peps)):.2f}")
            return train_peps, val_peps


    def sequence_split(self, 
                     train_frac: float=None,
                     drop_centrality: bool=True,
                     drop: bool=False,
                     hard_intersection: bool=True):
        """
        Splits the peptides based on the sequence
        """
        if train_frac is None:
            train_frac = self.train_frac
        print(f"Sequence split with train_frac {train_frac} and drop_centrality {drop_centrality}")
        print(f"Drop {drop} and hard_intersection {hard_intersection}")
        expt_segments_df = self.drop_centrality(drop=drop_centrality) # should also increase this as well to improve randomness of split
        new_expt_segs = Segments(expt_segments_df, keys=self.keys)

        peptide_numbers = new_expt_segs.segs_df["peptide"].to_numpy()

        # _mod = 1
        # if self.random_seed%2 == 0:
        #     _mod = -1

        sequence_pep = int(len(peptide_numbers) * train_frac)

        train_peps = peptide_numbers[:sequence_pep]
        val_peps = peptide_numbers[sequence_pep:]

        peps = [train_peps, val_peps]
        np.random.shuffle(peps)

        train_peps, val_peps = peps

        if drop:
            train_peps, val_peps = drop_intersection(new_expt_segs,train_peps, val_peps, hard=hard_intersection)
            if self.validate_split(train_peps, val_peps):
                return train_peps, val_peps
            else:
                print("Split not valid, recalculating")
                return self.sequence_split(train_frac=train_frac,
                                           drop_centrality=drop_centrality,
                                           drop=drop,
                                           hard_intersection=hard_intersection)
        else:
            print(f"Train peptides: {train_peps}")
            print(f"Val peptides: {val_peps}")
            print(f"Train %: {len(train_peps)/(len(train_peps)+len(val_peps)):.2f}")
            return train_peps, val_peps
    
        
    
    def redundant_sequence_split(self,
                        train_frac: float=None,
                        drop_centrality: bool=True,
                        hard_intersection: bool=False):
        """
        Redundant split using KMeans clustering of the start and end residues for each peptide
        """
        if train_frac is None:
            train_frac = self.train_frac
        print(f"Redundant kmeans sequence split with train_frac {train_frac} and drop_centrality {drop_centrality}")
        print(f"and hard_intersection {hard_intersection}")

        expt_segments_df = self.drop_centrality(drop=drop_centrality)
        new_expt_segs = Segments(expt_segments_df, keys=self.keys)

        features = expt_segments_df[self.keys].to_numpy()
        # print(f"Features shape: {features.shape}")
        # print(f"features: {features}")
        k_splits = len(expt_segments_df)//10
        kmeans = KMeans(n_clusters=k_splits, random_state=self.random_seed).fit(features)
        labels = kmeans.labels_
        unique_labels = np.unique(labels)

        train_labels = np.random.choice(unique_labels, 
                                        size=int(k_splits * train_frac),
                                        replace=False)        
        train_indexes = np.where(np.isin(labels, train_labels))[0]

        train_peps = new_expt_segs.pep_nums[train_indexes]
        val_peps = new_expt_segs.pep_nums[~np.isin(new_expt_segs.pep_nums, train_peps)]

        train_peps, val_peps =  drop_intersection(new_expt_segs,train_peps, val_peps, hard=hard_intersection)
        if self.validate_split(train_peps, val_peps):
            return train_peps, val_peps
        else:
            print("Split not valid, recalculating")
            return self.redundant_sequence_split(train_frac=train_frac,
                                                 drop_centrality=drop_centrality,
                                                 hard_intersection=hard_intersection)

    def structural_split(self,
                         top_path:str,
                         train_frac: float=None,
                         loops: bool=True,
                         compare: bool=True,
                         drop_centrality: bool=True,
                         hard_intersection: bool=False):
        print(f"Structural split with train_frac {train_frac} and drop_centrality {drop_centrality}")
        print(f"and hard_intersection {hard_intersection}")
        print(f"Loops {loops} and compare (alpha vs beta) {compare}")
        raise ValueError("Not implemented yet: DSSP not working at the moment")
        if train_frac is None:
            train_frac = self.train_frac
        expt_segments_df = self.drop_centrality(drop=drop_centrality)
        new_expt_segs = Segments(expt_segments_df, keys=self.keys)

        secondary_structure = PDB_to_DSSP(top_path)
        loop_residues = np.where(secondary_structure == "L")[0]
        helix_residues = np.where(secondary_structure == "H")[0]
        sheet_residues = np.where(secondary_structure == "S")[0]

        loop_peptides = new_expt_segs.df_select_residues(list(loop_residues))["peptide"].to_numpy()

        if not loops:
            # remove the loop peptides
            new_expt_segs.select_segments(peptides=list(loop_peptides), remove=True)

        helix_peptides = new_expt_segs.df_select_residues(list(helix_residues))["peptide"].to_numpy()
        sheet_peptides = new_expt_segs.df_select_residues(list(sheet_residues))["peptide"].to_numpy()
    
        if compare:
            # compare the helix and sheet peptides

            # randomly swap the helix and sheet peptides
            peptide_sets = [helix_peptides, sheet_peptides]
            np.random.shuffle(peptide_sets)
            train_peps, val_peps = peptide_sets

            return drop_intersection(new_expt_segs, train_peps, 
                                     val_peps, 
                                     hard=hard_intersection)
            
        else:
            # K means split of the remaining peptides
            _peptide_splitter = PeptideSplitter(expt_segs_df=new_expt_segs.segs_df, 
                                                train_frac=train_frac,
                                                random_seed=self.random_seed)
            return _peptide_splitter.redundant_sequence_split(train_frac=train_frac, 
                                                    drop_centrality=False,
                                                    hard_intersection=hard_intersection)


    def neighbours_split(self,
                        top_path: str,
                        train_frac: float=None,
                        drop_centrality: bool=True,
                        hard_intersection: bool=False):
        """
        Split the peptides based on neighbouring residues to a random residue
        """
        if train_frac is None:
            train_frac = self.train_frac
        print(f"Neighbours split with train_frac {train_frac} and drop_centrality {drop_centrality}")
        print(f"and hard_intersection {hard_intersection}")
        expt_segments_df = self.drop_centrality(drop=drop_centrality)
        new_expt_segs = Segments(expt_segments_df, keys=self.keys)

        top = mda.Universe(top_path)
        residues = top.select_atoms("protein").residues
        
        HDX_residues = new_expt_segs.residues    

        random_residue = np.random.choice(residues,1)[0]
        random_CA = random_residue.atoms.select_atoms("name CA")
        random_CA_coords = random_CA.positions

        residue_selection_string = " or ".join([f"(resnum {residue} and name CA)" for residue in HDX_residues])
        HDX_CA_atoms = top.select_atoms(residue_selection_string)
        HDX_CA_coords = HDX_CA_atoms.positions

        distances = np.linalg.norm(HDX_CA_coords - random_CA_coords, axis=1)
        distance_indexes = np.argsort(distances)

        train_indexes = distance_indexes[:int(len(HDX_CA_coords) * train_frac)]

        train_residues = HDX_CA_atoms[train_indexes].residues.resids

        train_peps = new_expt_segs.df_select_residues(train_residues)["peptide"].to_numpy()
        val_peps = new_expt_segs.df_remove_residues(train_residues)["peptide"].to_numpy()


        train_peps, val_peps = drop_intersection(new_expt_segs, train_peps, val_peps, hard=hard_intersection)
        if self.validate_split(train_peps, val_peps):
            return train_peps, val_peps
        else:
            print("Split not valid, recalculating")
            return self.neighbours_split(top_path=top_path,
                                        train_frac=train_frac,
                                        drop_centrality=drop_centrality,
                                        hard_intersection=hard_intersection)

    def spatial_split(self,
                    top_path: str,
                    kmeans_cluster: bool,
                    train_frac: float=None,
                    PCA_dims: int=1, #TODO test multiple PCA dims
                    hard_intersection: bool=True):
        """
        Split the peptides based on the spatial location of the residues
        Spatial distribution is determined by PCA
        """
        if train_frac is None:
            train_frac = self.train_frac
        if kmeans_cluster:
            drop_centrality = False
        else:
            drop_centrality = True
        print(f"Spatial split with train_frac {train_frac} and drop_centrality {drop_centrality}")
        print(f"and hard_intersection {hard_intersection}")
        print(f"PCA_dims {PCA_dims} and kmeans_cluster {kmeans_cluster}")

        expt_segments_df = self.drop_centrality(drop=drop_centrality) # perhaps we can increase the drop_centrality %
        new_expt_segs = Segments(expt_segments_df, keys=self.keys)
        HDX_residues = new_expt_segs.residues    

        top = mda.Universe(top_path)
        residue_selection_string = " or ".join([f"(resnum {residue} and name CA)" for residue in HDX_residues])
        HDX_CA_atoms = top.select_atoms(residue_selection_string)
        HDX_CA_coords = HDX_CA_atoms.positions

        if not kmeans_cluster:
            PCA_dims = 1
        pca = PCA(n_components=PCA_dims)

        pca.fit(HDX_CA_coords)
        pca_coords = pca.transform(HDX_CA_coords)


        if kmeans_cluster:
            ksplits = 10
            kmeans = KMeans(n_clusters=ksplits, 
                        random_state=self.random_seed).fit(pca_coords) #weight kmeans by peptide centrality

            labels = kmeans.labels_
            unique_labels = np.unique(labels)
            train_labels = np.random.choice(unique_labels, 
                                            int(ksplits * train_frac), 
                                            replace=False)
            train_indexes = np.where(np.isin(labels, train_labels))[0]

            train_residues = HDX_CA_atoms.residues[train_indexes].resids

            train_peptides = new_expt_segs.df_select_residues(train_residues)["peptide"].to_numpy()
            val_peptides = new_expt_segs.df_remove_peptides(train_residues)["peptide"].to_numpy()
        else:

            flat_pca = pca_coords.flatten()
            pca_indexes = np.argsort(flat_pca)

            sequence_mod = int(len(pca_indexes) * train_frac)

            train_residue_indexes = pca_indexes[:sequence_mod]
            val_residue_indexes = pca_indexes[sequence_mod:]

            indexes = [train_residue_indexes, val_residue_indexes]
            np.random.shuffle(indexes)

            train_residue_indexes, val_residue_indexes = indexes

            train_residues = HDX_CA_atoms.residues[train_residue_indexes].resids
            
            train_peptides = new_expt_segs.df_select_residues(train_residues)["peptide"].to_numpy()
            val_peptides = new_expt_segs.df_remove_residues(train_residues)["peptide"].to_numpy()


        train_peps, val_peps = drop_intersection(new_expt_segs,train_peptides, val_peptides, hard=hard_intersection)
        if self.validate_split(train_peps, val_peps):
            return train_peps, val_peps
        else:
            print("Split not valid, recalculating")
            return self.spatial_split(top_path=top_path,
                                    kmeans_cluster=kmeans_cluster,
                                    train_frac=train_frac,
                                    PCA_dims=PCA_dims,
                                    hard_intersection=hard_intersection)
        

def drop_intersection(expt_segs: Segments,
                        train_peps: np.ndarray, 
                        val_peps: np.ndarray,
                        hard: bool=False):
    """
    Handle the intersection of the train and val peptides
    If hard is False then remove the intersection
    If hard is True then add the intersection to the val peptides
    """
    # raise ValueError("Not implemented yet")

    train_residues = expt_segs.get_residues(expt_segs.df_select_peptides(list(train_peps)))
    val_residues = expt_segs.get_residues(expt_segs.df_select_peptides(list(val_peps)))

    train_val_intersection = np.intersect1d(train_residues, val_residues)
    print(f"Found intersecting residues: {train_val_intersection}")
    intersection_peptides = expt_segs.df_select_residues(list(train_val_intersection))["peptide"].to_numpy()
    print(f"Found intersecting peptides: {intersection_peptides}")
    # train_peps = np.setdiff1d(train_peps, intersection_peptides)
    # val_peps = np.setdiff1d(val_peps, intersection_peptides)

    if hard:
        val_peps = np.concatenate([val_peps, intersection_peptides])
        val_peps = np.unique(val_peps)
        train_peps = np.setdiff1d(train_peps, intersection_peptides)

    else:
        val_peps = np.setdiff1d(val_peps, intersection_peptides)
        train_peps = np.setdiff1d(train_peps, intersection_peptides)

    print(f"Train peptides: {train_peps}")
    print(f"Val peptides: {val_peps}")
    print(f"Intersection %: {len(train_val_intersection)/(len(expt_segs.pep_nums)):.2f}")
    print(f"Train %: {len(train_peps)/(len(train_peps)+len(val_peps)):.2f}")

    return train_peps, val_peps



def create_train_val_dfs(expt_segs_df: pd.DataFrame,
                        train_peps: np.ndarray,
                        val_peps: np.ndarray,
                        keys=['ResStr', 'ResEnd']):
    """
    Create the train and val dataframes from the experimental segments dataframe
    """
    print("Creating train and val dataframes")
    train_df = expt_segs_df.loc[expt_segs_df["peptide"].isin(train_peps)].copy()
    val_df = expt_segs_df.loc[expt_segs_df["peptide"].isin(val_peps)].copy()

    print(f"Train peptides: {train_peps}")
    print(f"Val peptides: {val_peps}")
    print(f"Train %: {len(train_peps)/(len(train_peps)+len(val_peps)):.2f}")

    return train_df, val_df