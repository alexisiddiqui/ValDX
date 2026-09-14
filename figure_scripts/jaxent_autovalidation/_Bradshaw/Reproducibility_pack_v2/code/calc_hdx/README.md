# Usage instructions for calc_hdx
---

calc_hdx is a Python 3.7 package that calculates HDX-MS observables directly from MD simulation trajectories. In addition to the standard Python libraries it has been used with the following dependencies:

MDtraj 1.9.3
Numpy 1.16.4
Scipy 1.3.0
Matplotlib 3.1.0

For a description of calc_hdx command line options, run `calc_hdx.py -h`

To recreate predicted HDX for the open & closed trajectories used in the article, run:
`../calc_hdx.py -t ../../../data/trajectories/TeaA_closed_reimaged.xtc ../../../data/trajectories/TeaA_open_reimaged.xtc -p ../../../data/trajectories/TeaA_ref_closed_state.pdb -m Radou -dt 0.167 1.0 10.0 60.0 120.0 -mopt "{'save_detailed' : True}" -seg TeaA_byresidue_segments.dat -out TeaA_mixed_ -log TeaA_mixed_HDX_analysis.log`

...inside the 'data' subdirectory. Alternatively, predicted HDX for all TeaA structures can be recreated if the '../../../data/trajectories/TeaA_initial_reimaged.xtc' trajectory is used instead.



# Alexi:
<!--  we need to extract the  tarball  to get the residue segments though these could be genered simply by hand -->
tar -xzf data.tar.gz 
rm -rf _output
mkdir _output ; cd _output
<!-- cp ../data/TeaA_byresidue_segments.dat TeaA_byresidue_segments.dat -->


<!-- This command runs with the standard setting -->
python ../calc_hdx.py -t "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_filtered.xtc" -p "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_ref_open_state.pdb" -m Radou -dt 0.167 1.0 10.0 60.0 120.0 -mopt "{'save_detailed' : True}" -seg "../data/TeaA_byresidue_segments.dat" -out TeaA_mixed_ -log TeaA_mixed_HDX_analysis.log


<!-- This command runs without a cutoff (vanilla-BV-like) -->
python ../calc_hdx.py -t "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_filtered.xtc" -p "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_ref_open_state.pdb" -m Radou -dt 0.167 1.0 10.0 60.0 120.0 -mopt "{'save_detailed' : True, 'switch_method': 'cutoff'}" -seg "../data/TeaA_byresidue_segments.dat" -out TeaA_mixed_ -log TeaA_mixed_HDX_analysis.log
