# Code supporting "Structural Interpretation of Hydrogen-Deuterium Exchange with Maximum-entropy Simulation Reweighting"
---

Contents:

# calc_hdx
The calc_hdx folder contains code used to analyze molecular dynamics trajectories and calculate predicted HDX-MS protection factors and deuterated fractions. The predictions are optionally compared with experimental data, and results plotted in a series of output pdf files. The folder contains the following files, which correspond to release v0.1 of calc_hdx available at https://github.com/rtb1c13/calc_hdx:
 
- calc_hdx.py : Main executable, usable from the command line with provided arguments, or by importing the calc_hdx package

- Functions.py : Transferable functions for trajectory reading/manipulating topologies and trajectory analysis

- DfPred.py : Parent class for analysis methods. Functions for calculating intrinsic rates from a topology, and for calculating deuterated fractions

- Methods.py : Analysis method classes for calculating protection factors with various models. Currently contains 'Radou' model (identical to Best & Vendruscolo, Structure, 2006, 14 (1), 97-106), and 'Persson-Halle' model (Persson & Halle, PNAS, 2015, 112 (33), 10383-10388).

- Analysis.py : Classes for calculating deuterated fractions and output results in tabular and graphical form

For usage instructions and to regenerate data, see the README.md file inside the calc_hdx folder

# HDXer
The HDXer folder contains code used to reweight a structural ensemble to best fit a target set of deuterated fractions, using a Best & Vendruscolo-style model for calculation of protection factors and a maximum entropy criterion to enforce agreement of the reference and target ensembles within a given level of error. The code uses the output of analysis with calc_hdx.py, specifically files containing the contacts and hydrogen bonds calculated for each structrual snapshot in the molecular dynamics trajectory, and files containing the intrinsic exchange rates for residues in the sequence. The folder contains the following files:

- TeaA_reweighting.py : Main executable, usable from the command line with provided arguments.

For usage instructions and to regenerate example data, see the README.md file inside the HDXer folder 

# Clustering
The clustering folder contains analysis scripts used to cluster the final reweighted ensemble using a DBSCAN algorithm as implemented in scikit-learn. A Euclidean distance in pairwise-C_alpha-RMSD space is used as a distance metric, and clustering is performed with various values of epsilon to explore a suitable cutoff, as qualitatively interpreted by silhouette score, and silhouette plot. The folder contains the following files:

- calculate_RMSDcluster_membership.py : Main executable to perform clustering analysis across multiple epsilon values and analyse final results

- extract_pops_and_frames.py : Extracts cluster sub-ensembles from the initial test ensemble, for subsequent visualization and analysis, e.g. density map calculations

For usage instructions and to regenerate example data, see the README.md file inside the Clustering folder

