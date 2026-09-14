# create figures of TeaA closed with ligand, colored by B-factor (HDX-uptake)

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


### 1 min ###

load ref_closed_closed-open_full_1.0min.pdb, closed 

hide everything, *
show cartoon, *

# show ectoine from closed state
create lig, resname 4cs and closed
#color cbf_peach, lig and elem c
show sticks, lig
show spheres, lig
hide everything, elem h

# color by atom type (except C)
#util.cnc

# side view of cleft
set_view (\
     0.863236189,   -0.049228419,   -0.502387822,\
    -0.462195486,   -0.477200687,   -0.747430682,\
    -0.202944741,    0.877411723,   -0.434689403,\
    -0.000036787,   -0.000181097, -135.383132935,\
    50.718467712,   53.034870148,   -4.244140625,\
    99.819534302,  170.908187866,   20.000001907 )

## spectrumany allows list of colors instead of defined palette
## RdYlBu
run spectrumany.py
#spectrumany b, red paleyellow blue, minimum=-1.0, maximum=1.0

#color cbf_peach, lig and elem c
## color by atom type (except C)
#util.cnc

#scene F1, store
#ray 1000, 700
#png ref_closed_closed-open_RdYlBu_1.0min.png, dpi=300, ray=1

# RdBu
spectrumany b, red white blue, minimum=-1.0, maximum=1.0

color cbf_peach, lig and elem c
# color by atom type (except C)
util.cnc

scene F2, store
ray 1000, 700
png ref_closed_closed-open_RdBu_1.0min.png, dpi=300, ray=1

## PiYG
#spectrumany b, hotpink paleyellow forest, minimum=-1.0, maximum=1.0

#color cbf_peach, lig and elem c
## color by atom type (except C)
#util.cnc

#scene F3, store
#ray 1000, 700
#png ref_closed_closed-open_PiYG_1.0min.png, dpi=300, ray=1

#delete all

