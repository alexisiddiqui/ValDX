#%%
import os
import pandas as pd

import sys
sys.path.append("/home/alexi/Documents/ValDX/")

from ValDX.helpful_funcs import dfracs_to_df, HDX_to_file


#%%

BRD4_HDX_path = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO.dat"
BRD4_times = [0.0, 15.0, 60.0, 600.0, 3600.0, 14400.0]

MBP_HDX_path = "/home/alexi/Documents/ValDX/raw_data/MBP/MaltoseBindingProtein/MBP_wt1.dat"
MBP_times = [30, 240, 1800, 14400]


HOIP_HDX_path = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo.dat"
HOIP_times = [0, 0.5, 5.0]




# read in the HDX data

BRD4_HDX = dfracs_to_df(BRD4_HDX_path, BRD4_times)

MBP_HDX = dfracs_to_df(MBP_HDX_path, MBP_times)

HOIP_HDX = dfracs_to_df(HOIP_HDX_path, HOIP_times)


# divide times in BRD4 and MBP by 60 to get minutes use times to get appropriate column names
for time in BRD4_times:
    BRD4_HDX.rename(columns={time: time/60}, inplace=True)

for time in MBP_times:
    MBP_HDX.rename(columns={time: time/60}, inplace=True)









HDX_data = {"BRD4": BRD4_HDX, "MBP": MBP_HDX, "HOIP": HOIP_HDX}

# %%

#remove really long or 0 timepoints
fake_timepoints = {0}


for key, df in HDX_data.items():
    for time in fake_timepoints:
        try:
            df = df.drop(time, axis=1)
        except:
            pass
    print(df)

    HDX_data[key] = df




new_BRD4_HDX_path = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_clean.dat"

new_MBP_HDX_path = "/home/alexi/Documents/ValDX/raw_data/MBP/MaltoseBindingProtein/MBP_wt1_clean.dat"

new_HOIP_HDX_path = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo_clean.dat"


HDX_to_file(new_BRD4_HDX_path, HDX_data["BRD4"])

HDX_to_file(new_MBP_HDX_path, HDX_data["MBP"])

HDX_to_file(new_HOIP_HDX_path, HDX_data["HOIP"])



# %%
