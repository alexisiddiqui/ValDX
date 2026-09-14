# create figures of TeaA open and closed with ligand

# color blind colors 12-scale
set_color cbf_dark_teal,   [1, 110, 130]
set_color cbf_purple,      [125, 45, 145]
set_color cbf_softblue,    [47, 94, 171]
set_color cbf_skyblue,     [68, 152, 211]
set_color cbf_violet,      [205, 133, 185]
set_color cbf_cyan,        [70, 195, 208]
set_color cbf_redrusset,   [170, 29, 63]
set_color cbf_peach,       [244, 119, 82]
set_color cbf_green,       [25, 179, 90]
set_color cbf_lime,        [237, 232, 59]
set_color cbf_peagreen,    [171, 211, 122]
set_color cbf_wheat,       [249, 229, 190]

# setup for figures
bg_color white
set ray_opaque_background, off
set ray_shadow, off
set ray_trace_fog, -1
set ray_trace_fog_start, 3 
set specular, 0.00
set shininess, 100
set depth_cue, on
set surface_quality, 1
viewport 1000, 700
set sphere_scale, 0.3

load ../example_open_state_prot.cif 
load ../ref_closed_state.cif 

hide everything, *
show cartoon, *

# show ectoine from closed state
create lig, resname 4cs and ref_closed_state
color cbf_peach, lig and elem c
show sticks, lig
show spheres, lig
hide everything, elem h

color cbf_wheat, ref_closed_state
color cbf_cyan,  example_open_state

# color by atom type (except C)
util.cnc

# side view of cleft
set_view (\
     0.863236189,   -0.049228419,   -0.502387822,\
    -0.462195486,   -0.477200687,   -0.747430682,\
    -0.202944741,    0.877411723,   -0.434689403,\
    -0.000036787,   -0.000181097, -135.383132935,\
    50.718467712,   53.034870148,   -4.244140625,\
    99.819534302,  170.908187866,   20.000001907 )

# Creat morph & movie
disable example_open_state
enable lig
morph closedtoopen, ref_closed_state and chain A and resi 1-310 , example_open_state and chain A and resi 1-310, steps=60, match=in

frame 1
enable lig
movie.sweep 45, 2
movie.produce Supplementary_movie_1.mpg, ray, quality=100




