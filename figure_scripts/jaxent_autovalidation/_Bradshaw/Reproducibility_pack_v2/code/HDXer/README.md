# Usage instructions for HDXer
---

TeaA_reweighting.py is a Python 3.7 script that reweights a structural ensemble to best fit a target set of deuterated fractions, using a Best & Vendruscolo-style model for calculation of protection factors and a maximum entropy criterion to enforce agreement of the reference and target ensembles within a given level of error. In addition to the standard Python libraries it has been used with the following dependencies:

Numpy 1.16.4

For a description of TeaA_reweighting.py command line options, run `TeaA_reweighting.py -h`

To perform an example reweighting that fits an initial ensemble of TeaA to recreate an artificially-generated set of HDX-MS data at a residue-resolved level and with gamma = 1000.0 (data used in Fig. 3 in the article ""), first run calc_hdx on the *full* TeaA reference trajectory according to the instructions in ../calc_hdx. This will calculate contacts and H-bonds for all the closed, open, and semi-open frames. Then run:

`./TeaA_reweighting.py -f ../calc_hdx/data -exp ../../data/artificial_HDX_data/mixed_60-40_artificial_expt_resfracs.dat -r ../calc_hdx/data/TeaA_initial_Intrinsic_rates.dat -out "mixed_gamma_1x10^3_" -g 1000 -mcmin -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0`

Note that thanks to the Monte-Carlo-based optimization of beta parameters used in the calculation of protection factors, repeated refinements may not give identical results. Reproducibility can be enforced by using a defined random seed (TeaA_reweighting.py, line 267).
 


# Alexi


rm -rf _output
mkdir _output ; cd _output
<!-- This command runs HDXer without optimising beta parameters -->
python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^3_" -g 1000  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -nopm


python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^2_" -g 100  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -nopm

python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^1_" -g 10  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -nopm


python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^0_" -g 1  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -nopm

python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^-1_" -g 0.1  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -nopm

cd ..

mkdir _output_mcminBV; cd _output_mcminBV 


python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^3_" -g 1000  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -mcmin


python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^2_" -g 100  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -mcmin

python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^1_" -g 10  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -mcmin


python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^0_" -g 1  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -mcmin


python ../TeaA_reweighting.py -f ../../calc_hdx/_output -exp ../mixed_60-40_artificial_expt_resfracs.dat -r ../../calc_hdx/_output/TeaA_mixed_Intrinsic_rates.dat -out "mixed_gamma_1x10^-1_" -g 0.1  -rf 0.001 -dt 0.167 1.0 10.0 60.0 120.0 -mcmin

cd ..