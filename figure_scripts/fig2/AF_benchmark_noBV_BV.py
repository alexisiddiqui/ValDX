
# run serial analysis 
import os 
import subprocess
import shutil

BPTI_script = "/home/alexi/Documents/ValDX/figure_scripts/fig2/BPTI_AF_benchmark_noBV_BV.py"

BRD484_script =  "/home/alexi/Documents/ValDX/figure_scripts/fig2/BRD4_AF_benchmark_noBV_BV.py"

HOIP_script =  "/home/alexi/Documents/ValDX/figure_scripts/fig2/HOIP_AF_benchmark_noBV_BV.py"

LXRa200_script =  "/home/alexi/Documents/ValDX/figure_scripts/fig2/LXR_AF_benchmark_noBV_BV.py"

MBP_script = "/home/alexi/Documents/ValDX/figure_scripts/fig2/MBP_AF_benchmark_noBV_BV.py"


if __name__ == "__main__":

    data_dir = "/home/alexi/Documents/ValDX/data" # do NOT remove raw_data
    shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    plots_dir = "/home/alexi/Documents/ValDX/plots"
    shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)

    results_dir = "/home/alexi/Documents/ValDX/results"
    shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    logs_dir = "/home/alexi/Documents/ValDX/logs"
    shutil.rmtree(logs_dir)
    os.makedirs(logs_dir)

    # # import subprocess
    # # try:
    # subprocess.run(["python", BPTI_script], check=True)
    # # except:
    # #     print("BPTI failed")
    
    # # try:
    # subprocess.run(["python", BRD484_script], check=True)
    # # except:
    # #     print("BRD484 failed")

    # # try:
    # subprocess.run(["python", HOIP_script], check=True)
    # # except:
    # #     print("HOIP failed")

    # # try:    
    # subprocess.run(["python", LXRa200_script], check=True)
    # # except:
    # #     print("LXRa200 failed")

    # # try:
    # subprocess.run(["python", MBP_script], check=True)
    # # except:
    #     print("MBP failed")

    # run all scripts in parallel

    subprocess.run(["python", BPTI_script])
    subprocess.run(["python", BRD484_script])
    subprocess.run(["python", HOIP_script])
    subprocess.run(["python", LXRa200_script])
    subprocess.run(["python", MBP_script])


    print("All scripts ran successfully")