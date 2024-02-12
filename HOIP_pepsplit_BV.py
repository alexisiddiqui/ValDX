


### ValDXer testing
import os
os.environ['HDXER_PATH'] = '/home/alexi/Documents/HDXer'
import subprocess


# from autonotebook import tqdm as notebook_tqdm

from ValDX.ValidationDX import ValDXer
from ValDX.VDX_Settings import Settings
import pandas as pd
import MDAnalysis as mda


settings = Settings()
settings.replicates = 2
settings.gamma_range = (1,8)
settings.train_frac = 0.5
settings.times = [0.0, 0.5, 5.0]
settings.RW_exponent = [0]
settings.HDXer_stride = 1 # this must be smaller than the size of trajectory
settings.RW_do_reweighting = False
settings.RW_do_params = True
settings.save_figs = True
settings.stride = 100
reps = 5

expt_dir = "/home/alexi/Documents/ValDX/raw_data/HOIP/dab3"

HOIP_dir = "/home/alexi/Documents/xMD-HOIP/data/MD/6SC6/APO_dab3"




reps_list = [os.path.join(HOIP_dir, f"R_{i}") for i in range(1, reps+1)]

print(reps_list)


traj_names = ["APO_dab3_6SC6_1-nojump.xtc"]

reordered_traj_names = ["APO_dab3_6SC6_1-nojump_reordered.xtc"]

sim_name = "HOIP_dab3_APO"
top_name = "APO_dab3_6SC6_1-nojump.pdb"
reordered_top_name = "APO_dab3_6SC6_1-nojump_reordered.pdb"

top_path = os.path.join(HOIP_dir, top_name)





import pickle

expt_name = 'Experimental'
test_name = "HOIPdab3"




import os
print(os.getenv('HDXER_PATH'))



print(os.environ["HDXER_PATH"])
print(__name__)


def preprocess_main_HDX():

    raw_csv = os.path.join("raw_data", "HOIP","dab3","dab3_3_excel.csv")

    raw_df = pd.read_csv(raw_csv)
    # remove multilevel columns
    # print(raw_df.head())

    # select state dab3
    dab3_df = raw_df[raw_df['State'] == 'dAb3_3']


    # add 697 to Start and End  
    dab3_df['Start'] = dab3_df['Start'] 
    dab3_df['End'] = dab3_df['End']

    # add UptakeFraction column
    dab3_df['UptakeFraction'] = dab3_df['Uptake'] / dab3_df['MaxUptake']
    dab3_df.tail()

    columns_to_drop = ["Protein", "Sequence", "Fragment", "Modification", "State", "MaxUptake", "Uptake", "MHP", "Center", "Center SD", "Uptake", "Uptake SD", "RT", "RT SD"]
    dab3_df = dab3_df.drop(columns_to_drop, axis=1)
    dab3_df.head()
    dab3_df= dab3_df.drop(columns=['Unnamed: 1','Unnamed: 2'])
    dab3_df.head()


    # pivot exposure and uptake fraction
    dab3_df = dab3_df.groupby(['Start', 'End', 'Exposure'])['UptakeFraction'].mean().reset_index()

    dab3_df.head()


    # remove End > 1072
    dab3_df = dab3_df.loc[dab3_df['End'] <= 1072-697]
    dab3_df.head()

    dab3_df = dab3_df.pivot(index=['Start','End'], columns='Exposure', values='UptakeFraction')



    # print entire dataframe
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(dab3_df)

    # fill in missing

    dab3_df.reset_index(inplace=True)

    # Forward fill the 'Start' column
    dab3_df['Start'] = dab3_df['Start'].ffill()

    # Set 'Start' and 'End' back as the index if needed
    dab3_df.set_index(['Start', 'End'], inplace=True)

    # save with space delimiter
    # round to 5 dp

    dab3_df = dab3_df.round(5)
    dab3_df.to_csv(os.path.join("raw_data", "HOIP", 'HOIP_dab3_dfs.csv'), sep=' ')





    dab3_df.to_csv(os.path.join("raw_data", "HOIP", "dab3.csv"), index=False)

    # select only Start and End
    dab3_df.reset_index(inplace=True)

    dab3_segs = dab3_df[['Start', 'End']]
    # dab3_segs.drop(columns=["Exposure"])

    dab3_segs.head()


    os.listdir(expt_dir)

    segs_name = "HOIP_dab3_segs.txt"
    segs_path = os.path.join(expt_dir, segs_name)

    hdx_name = "HOIP_dab3_dfs.dat"
    hdx_path = os.path.join(expt_dir, hdx_name)
    print(hdx_path)

    rates_name = "out__train_HOIPdab3_1Intrinsic_rates.dat" #need to correct this
    rates_path =os.path.join(expt_dir, rates_name)


    segs = [(1, 7),
    (6, 18),
    (6, 21),
    (6, 25),
    (7, 18),
    (7, 21),
    (7, 25),
    (8, 18),
    (8, 20),
    (8, 21),
    (8, 24),
    (11, 18),
    (11, 20),
    (11, 21),
    (11, 23),
    (11, 25),
    (19, 25),
    (26, 32),
    (26, 33),
    (27, 33),
    (33, 40),
    (34, 40),
    (34, 41),
    (41, 48),
    (41, 49),
    (41, 51),
    (42, 49),
    (42, 51),
    (42, 65),
    (49, 61),
    (49, 65),
    (50, 65),
    (52, 61),
    (52, 65),
    (74, 84),
    (76, 84),
    (77, 84),
    (77, 85),
    (77, 87),
    (77, 88),
    (77, 97),
    (85, 97),
    (86, 97),
    (88, 97),
    (89, 97),
    (98, 104),
    (98, 106),
    (98, 108),
    (113, 120),
    (114, 120),
    (117, 131),
    (121, 131),
    (121, 132),
    (132, 149),
    (132, 150),
    (132, 153),
    (150, 169),
    (151, 162),
    (151, 167),
    (151, 169),
    (151, 170),
    (154, 169),
    (155, 169),
    (169, 183),
    (170, 183),
    (170, 185),
    (170, 186),
    (171, 183),
    (171, 186),
    (184, 195),
    (184, 208),
    (187, 195),
    (187, 208),
    (187, 209),
    (188, 208),
    (194, 208),
    (196, 208),
    (213, 223),
    (224, 238),
    (224, 239),
    (247, 257),
    (247, 259),
    (247, 260),
    (248, 257),
    (248, 259),
    (248, 260),
    (250, 257),
    (250, 260),
    (253, 260),
    (261, 275),
    (261, 276),
    (261, 278),
    (261, 281),
    (305, 313),
    (305, 315),
    (308, 315),
    (314, 327),
    (316, 327),
    (317, 327),
    (333, 356),
    (334, 356),
    (334, 357),
    (336, 356),
    (336, 357),
    (336, 360),
    (338, 360),
    (339, 360),
    (342, 360)]


    # save as tabbed txt file between each column
    with open(segs_path, 'w') as f:
        for seg in segs:
            f.write("%s\t%s\n" % seg)

    return segs_path, hdx_path, rates_path













def preprocess_main_MD():

    # test reordering 
    # HOIP Chain B resi 697-1072
    # dab3 Chain A C resi 1-120

    top_test = mda.Universe(top_path)
    print(top_test.atoms)

    HOIP_selection = "protein and segid B and resid 697:1072"
    dab3_selection1 = "protein and segid A and resid 1:120"
    dab3_selection2 = "protein and segid C and resid 1:120"

    HOIP = top_test.select_atoms(HOIP_selection)
    # set to Chain A
    # for atom in HOIP:
    #     atom.segment.segid = "A"
        # set chain to A
    dab3_1 = top_test.select_atoms(dab3_selection1)
    # for atom in dab3_1:
    #     atom.segment.segid = "B"
    dab3_2 = top_test.select_atoms(dab3_selection2)
    # for atom in dab3_2:
    #     atom.segment.segid = "C"

    new_order = HOIP + dab3_1 + dab3_2

    print(new_order)

    new_order_universe = mda.Merge(new_order)
    # # renumber residues
    # for idx, res in enumerate(new_order_universe.residues):
    #     res.resid = idx + 1

    print(new_order_universe.atoms)



    # save pdb
    reordered_top_path = os.path.join(HOIP_dir, reordered_top_name)
    new_order_universe.atoms.write(reordered_top_path)


    # re number residues with pdb-tools
    renumbered_path = os.path.join(HOIP_dir, "renumbered.pdb")
    f"pdb_reres -1 {reordered_top_path} > {renumbered_path}"
    subprocess.run(f"pdb_reres -1 {reordered_top_path} > {renumbered_path}", shell=True)

    # read in reordered pdb as text
    with open(reordered_top_path, 'r') as f:
        reordered_pdb = f.readlines()

        new_lines = []
        for line in reordered_pdb:
            # print(line)
            split = line.split()
            if "ATOM" == split[0]:
                print(line)
                print(line[21])
                print(line[72])
                # replace index 21 with 72
                line = line[:21] + line[72] + line[22:]
                # break

            new_lines.append(line)

    # with open(reordered_top_path, 'w') as f:
    #     f.writelines(new_lines)
            
        
    #traj_paths is a list of every combination of rep_paths and traj_names

    traj_paths = []
    for rep_path in reps_list:
        for traj_name in traj_names:
            path = os.path.join(rep_path, traj_name)
            print(path)
            traj_paths.append(path)

    # print(top_path)

    # test reordering 
    # HOIP Chain B resi 697-1072
    # dab3 Chain A C resi 1-120

    top_test = mda.Universe(top_path, traj_paths)
    print(len(top_test.trajectory))
    print(top_test.atoms)

    HOIP_selection = "protein and segid B and resid 697:1072"
    dab3_selection1 = "protein and segid A and resid 1:120"
    dab3_selection2 = "protein and segid C and resid 1:120"

    HOIP = top_test.select_atoms(HOIP_selection)
    # set to Chain A
    # for atom in HOIP:
    #     atom.segment.segid = "A"
        # set chain to A
    dab3_1 = top_test.select_atoms(dab3_selection1)
    # for atom in dab3_1:
    #     atom.segment.segid = "B"
    dab3_2 = top_test.select_atoms(dab3_selection2)
    # for atom in dab3_2:
    #     atom.segment.segid = "C"

    new_order = HOIP + dab3_1 + dab3_2

    print(new_order)

    new_order_universe = mda.Merge(new_order)
    print(len(new_order_universe.trajectory))

    # Renumber residues if needed
    # new_resid = 1
    # for residue in new_order_universe.residues:
    #     residue.resid = new_resid
    #     new_resid += 1

    print(new_order_universe.atoms)
    # new_order_universe.atoms.write("test.pdb")

    # Prepare to write the new trajectory
    new_traj_path = os.path.join(HOIP_dir, reordered_traj_names[0])
    with mda.Writer(new_traj_path, new_order_universe.atoms.n_atoms) as W:
        for ts in top_test.trajectory[::settings.stride]:
            # Update positions of the new universe to match the current timestep
            new_order_universe.atoms.positions = top_test.atoms.positions
            # Write the timestep to the new trajectory
            W.write(new_order_universe)
            W.write(new_order_universe)
            break



    return new_traj_path, renumbered_path










def run_split_test(split_mode, name, system):

    settings.split_mode = split_mode
    settings.name = "_".join([name, system])

    VDX = ValDXer(settings)


    VDX.load_HDX_data(HDX_path=hdx_path, SEG_path=segs_path, calc_name=expt_name)
    VDX.load_intrinsic_rates(path=rates_path, calc_name=expt_name)

    VDX.load_structures(top_path=renumbered_path, traj_paths=[new_traj_path], calc_name=test_name)
    # VDX.load_structures(top_path=top_path, traj_paths=traj_paths, calc_name=test_name)

    run_outputs = VDX.run_VDX(calc_name=test_name, expt_name=expt_name)
    analysis_dump, df, name = VDX.dump_analysis()
    save_path = VDX.save_experiment()

    return run_outputs, analysis_dump, df, name, save_path




if __name__ == "__main__":
    segs_path, hdx_path, rates_path = preprocess_main_HDX()
    new_traj_path, renumbered_path = preprocess_main_MD()

    splits = ['r', 's', 'R']
    split_names = ['random', 'sequential', 'redundancy']
    system = 'HOIPdab3_test'

    raw_run_outputs = {}
    analysis_dumps = {}
    analysis_df = pd.DataFrame()
    names = []
    save_paths = []


    for split, split_name in zip(splits, split_names):
        run_outputs, analysis_dump, df, name, save_path = run_split_test(split, split_name, system)
        raw_run_outputs[name] = run_outputs
        analysis_dumps[name] = analysis_dump[name]
        analysis_df = pd.concat([analysis_df, df])
        names.append(name)
        save_paths.append(save_path)
















    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming your DataFrame is named df
    # Replace 'your_dataframe' with your actual DataFrame variable
    df = analysis_df

    # Create a FacetGrid, using 'name' for each subplot
    g = sns.FacetGrid(df, col="name", col_wrap=3, height=4, aspect=1.5)
    g.fig.suptitle('MSE over Time by Type for each Named Split Mode')

    # Create boxplots
    g = g.map(sns.boxplot, "time", "mse", "Type", palette="Set3")

    # Adding some additional options for better visualization
    g.add_legend(title='Type')
    g.set_axis_labels("Time", "MSE")
    g.set_titles("{col_name}")

    # Adjust the arrangement of the plots
    plt.subplots_adjust(top=0.9)

    # Show plot
    plt.show()



    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming your DataFrame is named df
    # Replace 'your_dataframe' with your actual DataFrame variable
    df = analysis_df

    # Create a FacetGrid, using 'name' for each subplot
    g = sns.FacetGrid(df, col="name", col_wrap=3, height=4, aspect=1.5)
    g.fig.suptitle('R over Time by Type for each Named Split Mode')

    # Create boxplots
    g = g.map(sns.boxplot, "time", "R", "Type", palette="Set3")

    # Adding some additional options for better visualization
    g.add_legend(title='Type')
    g.set_axis_labels("Time", "R")
    g.set_titles("{col_name}")

    # Adjust the arrangement of the plots
    plt.subplots_adjust(top=0.9)

    # Show plot
    plt.show()






