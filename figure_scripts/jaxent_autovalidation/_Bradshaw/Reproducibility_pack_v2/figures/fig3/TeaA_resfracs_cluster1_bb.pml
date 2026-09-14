# create figures of TeaA closed with cluster map overlay

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

load example_open_state_prot.cif 
load BBonly_cluster_1_script_amplitude.xplor, bb_open_map


hide everything, *
show cartoon, example_open_state_prot
isomesh bb_open_mesh, bb_open_map, 0.25

# Color by structure, wheat for closed
color cbf_cyan, example_open_state_prot
color cbf_cyan, bb_open_mesh

# color by atom type (except C)
util.cnc

# side view of cleft
set_view (\
     0.863236189,   -0.049228419,   -0.502387822,\
    -0.462195486,   -0.477200687,   -0.747430682,\
    -0.202944741,    0.877411723,   -0.434689403,\
    -0.000038667,   -0.000188872, -135.388198853,\
    50.791168213,   52.244544983,   -2.957550049,\
    99.819534302,  170.908187866,   20.000001907 )

scene F1, store
ray 1000, 700
png TeaA_resfracs_open_cluster_1.png



