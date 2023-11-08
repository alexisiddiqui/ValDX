# %%
### ValDXer testing
import os
from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings
import concurrent.futures
settings = Settings(name='test')

VDX = ValDXer(settings)

# %% [markdown]
# 

# %%


# %%
# BPTI data
BPTI_dir = "/Users/alexi/Library/CloudStorage/OneDrive-Nexus365/Rotation_Projects/Rotation_3/Project/ValDX/raw_data/HDXer_tutorial/BPTI"

expt_name = 'BPTI_expt'


# %%
expt_dir = os.path.join(BPTI_dir, "BPTI_expt_data")

os.listdir(expt_dir)

segs_name = "BPTI_residue_segs.txt"
segs_path = os.path.join(expt_dir, segs_name)

hdx_name = "BPTI_expt_dfracs.dat"
hdx_path = os.path.join(expt_dir, hdx_name)
print(hdx_path)

rates_name = "BPTI_Intrinsic_rates.dat"
rates_path = os.path.join(expt_dir, rates_name)


# %%

VDX.load_HDX_data(HDX_path=hdx_path, SEG_path=segs_path, calc_name=expt_name)
VDX.load_intrinsic_rates(path=rates_path, calc_name=expt_name)


# %%
print(VDX.paths)


# %%


# %%
sim_name = 'BPTI_MD'

sim_dir = os.path.join(BPTI_dir, "BPTI_simulations")

os.listdir(sim_dir)

md_reps = 1

rep_dirs = ["Run_"+str(i+1) for i in range(md_reps)]

top_name = "bpti_5pti_eq6_protonly.gro"

top_path = os.path.join(sim_dir, rep_dirs[0], top_name)

traj_name = "bpti_5pti_reimg_protonly.xtc"

traj_paths = [os.path.join(sim_dir, rep_dir, traj_name) for rep_dir in rep_dirs]

print(top_path)
print(traj_paths)


VDX.load_structures(top_path=top_path, traj_paths=traj_paths, calc_name=sim_name)

# %%
test_name = "BPTI_RW_test"
VDX.load_structures(top_path=top_path, traj_paths=traj_paths, calc_name=test_name)
VDX.split_segments(seg_name=expt_name, calc_name=test_name, rep=1)


# %%
df, rep_name = VDX.predict_HDX(calc_name=test_name, rep=1, train=True)

# %%
train_gamma, rw_df = VDX.reweight_HDX(expt_name=expt_name, calc_name=test_name, rep=1, train=True)

# %%
VDX.paths

# %%
df, rep_name = VDX.predict_HDX(calc_name=test_name, rep=1, train=False)

# %%
opt_gamma, rw_df = VDX.reweight_HDX(expt_name=expt_name, calc_name=test_name, rep=1, train=False, train_gamma=train_gamma)

# %%
VDX.run_VDX(expt_name=expt_name, calc_name=test_name)


