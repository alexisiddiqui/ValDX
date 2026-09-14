import os
import pandas as pd
import copy
import sys
import numpy as np

sys.path.append("/home/alexi/Documents/ValDX/")

from ValDX.VDX_dataclasses import Segments
from ValDX.helpful_funcs import segs_to_df, segs_to_file, dfracs_to_df, HDX_to_file

# BPTI data
BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO"
# BPTI_dir = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI"
os.listdir(BPTI_dir)

segs_name = "BRD4_APO_segs.txt"
segs_path = os.path.join(BPTI_dir, segs_name)
hdx_name = "BRD4_APO_clean.dat"
hdx_path = os.path.join(BPTI_dir, hdx_name)

print(hdx_path)

all_segs = Segments(segs_path=segs_path)
all_segs_df = all_segs.segs_df

print(all_segs_df)
print(all_segs.residues)

res_range = list(range(min(all_segs.residues), max(all_segs.residues) + 1))

# find the middle of the residues
mid_res = (min(res_range) + max(res_range)) / 2
print("Middle of Res Range: ", mid_res)

mean_res = all_segs.residues.mean()
print("Mean of Resisudes: ", mean_res)

threshold = round(mean_res)
times = [0.25, 1.0, 10.0, 60.0]

# now read in dfracs
all_hdx_df = dfracs_to_df(path=hdx_path, names=times)

print(all_hdx_df.peptide)

hdx_df_stats = all_hdx_df.copy()

# Calculate the coefficient of variation for each peptide
for idx, row in hdx_df_stats.iterrows():
    peptide_values = row[times].values
    mean_value = np.mean(peptide_values)
    std_value = np.std(peptide_values)
    if mean_value != 0:
        cv = std_value / mean_value * 100
        hdx_df_stats.loc[idx, 'cv'] = cv
    else:
        hdx_df_stats.loc[idx, 'cv'] = np.nan

print(hdx_df_stats)

# Plot the coefficient of variation of each peptide
import matplotlib.pyplot as plt

plt.bar(hdx_df_stats.peptide, hdx_df_stats['cv'])
plt.axhline(y=5, color='r', linestyle='-')  # Add a reference line at CV=30%
plt.xlabel('Peptide')
plt.ylabel('Coefficient of Variation (%)')
plt.title('Variability of Peptides Across Time Points')
# plt.yscale('log')
plt.show()

# Print peptides with CV < 5%
for idx, peptide in enumerate(hdx_df_stats.peptide):
    if hdx_df_stats.loc[idx, 'cv'] < 4:
        print(hdx_df_stats.loc[idx, 'cv'])
        print(all_segs_df[all_segs_df['peptide'] == peptide])
        print(idx, peptide)


pep_nums_to_remove = [0,1,2,3,4,99,100,101,102]





BRD4a_residues = [res for res in all_segs.residues if res < threshold]

BRD4b_residues = [res for res in all_segs.residues if res > threshold]
# create copy of all_segs
BRD4a_segs = copy.deepcopy(all_segs)
BRD4b_segs = copy.deepcopy(all_segs)

BRD4a_segs.select_segments(residues=BRD4a_residues)
BRD4a_segs.select_segments(peptides=pep_nums_to_remove, remove=True)
BRD4a_segs_df = BRD4a_segs.segs_df
BRD4a_peptides = BRD4a_segs.pep_nums

BRD4b_segs.select_segments(residues=BRD4b_residues)
BRD4b_segs.select_segments(peptides=pep_nums_to_remove, remove=True)
BRD4b_segs_df = BRD4b_segs.segs_df
BRD4b_peptides = BRD4b_segs.pep_nums

BRD4a_segs_path = segs_path.replace("BRD4_APO/BRD4_", "BRD4_APO/BRD4a_")

segs_to_file(path=BRD4a_segs_path, df=BRD4a_segs_df)


BRD4b_segs_path = segs_path.replace("BRD4_APO/BRD4_", "BRD4_APO/BRD4b_")

segs_to_file(path=BRD4b_segs_path, df=BRD4b_segs_df)

print(BRD4a_peptides)
print(BRD4b_peptides)














BRD4a_hdx_df = all_hdx_df[all_hdx_df['peptide'].isin(BRD4a_peptides)]
BRD4b_hdx_df = all_hdx_df[all_hdx_df['peptide'].isin(BRD4b_peptides)]

print(BRD4a_hdx_df.peptide)
print(BRD4b_hdx_df.peptide)

BRD4a_hdx_path = hdx_path.replace("BRD4_APO/BRD4_", "BRD4_APO/BRD4a_")
BRD4b_hdx_path = hdx_path.replace("BRD4_APO/BRD4_", "BRD4_APO/BRD4b_")

HDX_to_file(path=BRD4a_hdx_path, df=BRD4a_hdx_df)
HDX_to_file(path=BRD4b_hdx_path, df=BRD4b_hdx_df)





# plot the residues in the segments as a bar chart
import matplotlib.pyplot as plt
import numpy as np

# use tkagg backend
# plt.switch_backend('tkagg')

# 1 if residue in res_range, 0 if not
res_ys = [1 if res in res_range else 0 for res in all_segs.residues]

plt.bar(all_segs.residues, res_ys)

plt.show()