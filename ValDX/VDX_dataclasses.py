import pandas as pd
import time
import os
from icecream import ic
from dataclasses import dataclass
from typing import List, Tuple, Union


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


        for key in self.dataframes.keys():
            df = self.dataframes[key]
            if isinstance(df, list):
                for sub_df in df:
                    print(self.info)
                    print(f"{key} verifying...")
                    try:
                        self.verify_dataframe(sub_df)
                    except:
                        print(f"{key} failed")
                        print(sub_df)
                        raise ValueError(f"{key} failed")

            else:
                print(self.info)
                print(f"{key} verifying...")
                try:
                    self.verify_dataframe(df)
                except:
                    print(f"{key} failed")
                    print(df)
                    raise ValueError(f"{key} failed")


    def verify_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        if "name" not in df.columns:
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



def merge_AnalysisData_classes(data_list: List[AnalysisData])->dict:
    """
    Merge the dataframes in the AnalysisData classes into individual dataframes - one for each key in the AnalysisData class
    """
    keys = data_list[0].dataframes.keys()

    merged_dfs = {}
    for key in keys:
        _df = data_list[0].dataframes[key]
        if isinstance(_df, pd.DataFrame):
            df_list = [data.dataframes[key] for data in data_list]
            merged_dfs[key] = pd.concat(df_list, ignore_index=True)
        elif isinstance(_df, list):
            #flatten all lists
            df_list = [df for data in data_list for df in data.dataframes[key]]
            merged_dfs[key] = pd.concat(df_list, ignore_index=True)

    return merged_dfs