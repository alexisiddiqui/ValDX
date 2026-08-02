# Usage instructions for Clustering
---

calculate_RMSDcluster_membership.py and extract_pops_and_frames.py are Python 3.7 scripts that perform clustering analysis of the TeaA test ensemble before and after reweighting, and extract ensemble structures into sub-ensembles based on their cluster membership. In addition to the standard Python libraries, they have been used with the following dependencies:

Numpy 1.16.4
MDtraj 1.9.3
Scikit-learn 0.21.2
Matplotlib 3.1.0

To perform an example clustering using weights from the reweighting with residue-resolved artificial HDX data, run:

`./calculate_RMSDcluster_membership.py`
`./extract_pops_and_frames.py`

Example outputs (excluding the DCD trajectory files of sub-ensembles, which merely replicate the structures present in ../../data/trajectories/TeaA_initial_reimaged.xtc) are available in the ./data subdirectory


