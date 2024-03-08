
# run serial analysis 
import os 
import subprocess
BPTI_script = "/home/alexi/Documents/ValDX/GSD_AFS_RW_ValDXer_testing_BPTI_0.5split.py"

BRD484_script = "/home/alexi/Documents/ValDX/GSD_AFS_RW_ValDXer_testing_BRD4_apo1_0.5split.py"

HOIP_script = "/home/alexi/Documents/ValDX/GSD_AFS_RW_ValDXer_testing_HOIP_apo_0.5split.py"

LXRa200_script = "/home/alexi/Documents/ValDX/GSD_AFS_RW_ValDXer_testing_LXRa200_0.5split.py"

MBP_script = "/home/alexi/Documents/ValDX/GSD_AFS_RW_ValDXer_testing_MBP_wt1_0.5split.py"

if __name__ == "__main__":

    data_dir = "/home/alexi/Documents/ValDX/data" # do NOT remove raw_data
    os.removedirs(data_dir)
    os.makedirs(data_dir)

    plots_dir = "/home/alexi/Documents/ValDX/plots"
    os.removedirs(plots_dir)
    os.makedirs(plots_dir)

    results_dir = "/home/alexi/Documents/ValDX/results"
    os.removedirs(results_dir)
    os.makedirs(results_dir)

    import subprocess
    try:
        subprocess.run(["python", BPTI_script], check=True)
    except:
        print("BPTI failed")
    
    try:
        subprocess.run(["python", BRD484_script], check=True)
    except:
        print("BRD484 failed")

    try:
        subprocess.run(["python", HOIP_script], check=True)
    except:
        print("HOIP failed")

    try:    
        subprocess.run(["python", LXRa200_script], check=True)
    except:
        print("LXRa200 failed")

    try:
        subprocess.run(["python", MBP_script], check=True)
    except:
        print("MBP failed")

    print("All scripts ran successfully")