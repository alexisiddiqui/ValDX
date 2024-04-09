import pandas as pd
import time
import os
from icecream import ic
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AnalysisInfo:
    settings_name: str
    analysis_name: str
    n_reps: int
    split_mode: str
    calc_name: str
    expt_name: str
    times: list
    system_name: str = None
    benchmark: bool = False

    def __post_init__(self):
        if self.system_name is None:
            self.system_name = self.settings_name

@dataclass
class AnalysisData:
    name: str
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
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            print(f"{key} verifying...")
            try:
                df["name"] = [self.info.settings_name] * len(df)
                df["system"] = [self.info.system_name] * len(df)

                df["analysis"] = [self.info.analysis_name] * len(df)
                df["split_type"] = [self.info.split_mode] * len(df)

                df["protein"] = df["system"]
                df["dataset"] = [self.info.calc_name] * len(df)
                df["class"] =  df["dataset"] + "_" + df["split_type"]
            except:
                print(f"{key} failed")
                print(df)
                raise ValueError(f"{key} failed")

