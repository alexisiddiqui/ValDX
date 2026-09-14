
from matplotlib import colors as mcolors

def rgb_to_fractionrgb(vals):
    """Converts RGB values to fractions for matplotlib.
       Assumes normalization to 255"""
    currfrac = []
    for i in vals:
        if i > 255:
            raise ValueError("RGB value can't be > 255")
        currfrac.append(i/255.)
    return tuple(currfrac)

# Colorblind friendly 8-colors from https://jfly.uni-koeln.de/color/#pallet
# Values are in RGB fractions
cnames = [ 'cbf_key', \
           'cbf_orange', \
           # NOTE difference to cbf_skyblue in 12-color!
           'cbf_sky_blue', \
           'cbf_bluish_green', \
           'cbf_yellow', \
           'cbf_blue', \
           'cbf_vermillion', \
           'cbf_reddish_purple' ]
cvals =  [ (0.0, 0.0, 0.0), \
           (0.9, 0.6, 0.0), \
           (0.35, 0.7, 0.9), \
           (0.0, 0.6, 0.5), \
           (0.95, 0.9, 0.25), \
           (0.0, 0.45, 0.7), \
           (0.8, 0.4, 0.0), \
           (0.8, 0.6, 0.7) ]  
chexes = list(map(mcolors.to_hex, cvals))
mcolors.get_named_colors_mapping().update(zip(cnames, chexes))

# Colorblind friendly 12-colors from http://mkweb.bcgsc.ca/colorblind/
# Values are in RGB values
cnames = [ 'cbf_dark_teal', \
           'cbf_purple', \
           'cbf_softblue', \
           # NOTE difference to cbf_sky_blue in 8-color!
           'cbf_skyblue', \
           'cbf_violet', \
           'cbf_cyan', \
           'cbf_redrusset', \
           'cbf_peach', \
           'cbf_green', \
           'cbf_lime', \
           'cbf_peagreen', \
           'cbf_wheat' ]
cvals =  [ (1, 110, 130), \
           (125, 45, 145), \
           (47, 94, 171), \
           (68, 152, 211), \
           (205, 133, 185), \
           (70, 195, 208), \
           (170, 29, 63), \
           (244, 119, 82), \
           (25, 179, 90), \
           (237, 232, 59), \
           (171, 211, 122), \
           (249, 229, 190) ]  
cval_fracs = list(map(rgb_to_fractionrgb, cvals))
chexes = list(map(mcolors.to_hex, cval_fracs))
mcolors.get_named_colors_mapping().update(zip(cnames, chexes))

# If the YlOrBr Brewer colormap already available in matplotlib
# isn't suitable, it can be recreated with the following 9 colors from http://mkweb.bcgsc.ca/brewer/
# Values are in RGB values
cnames = [ 'cbf_ylorbr-9-seq-1', \
           'cbf_ylorbr-9-seq-2', \
           'cbf_ylorbr-9-seq-3', \
           'cbf_ylorbr-9-seq-4', \
           'cbf_ylorbr-9-seq-5', \
           'cbf_ylorbr-9-seq-6', \
           'cbf_ylorbr-9-seq-7', \
           'cbf_ylorbr-9-seq-8', \
           'cbf_ylorbr-9-seq-9', ]
cvals =  [ (255, 255, 229), \
           (255, 247, 188), \
           (254, 227, 145), \
           (254, 196, 79), \
           (254, 153, 41), \
           (236, 112, 20), \
           (204, 76, 2), \
           (153, 52, 4), \
           (102, 37, 6) ]  
cval_fracs = list(map(rgb_to_fractionrgb, cvals))
chexes = list(map(mcolors.to_hex, cval_fracs))
mcolors.get_named_colors_mapping().update(zip(cnames, chexes))

# If the RdYlBu Brewer colormap already available in matplotlib
# isn't suitable, it can be recreated with the following 9 colors from http://mkweb.bcgsc.ca/brewer/
# Values are in RGB values
cnames = [ 'cbf_rdylbu-9-seq-1', \
           'cbf_rdylbu-9-seq-2', \
           'cbf_rdylbu-9-seq-3', \
           'cbf_rdylbu-9-seq-4', \
           'cbf_rdylbu-9-seq-5', \
           'cbf_rdylbu-9-seq-6', \
           'cbf_rdylbu-9-seq-7', \
           'cbf_rdylbu-9-seq-8', \
           'cbf_rdylbu-9-seq-9', ]
cvals =  [ (215, 48, 39), \
           (244, 109, 67), \
           (253, 174, 39), \
           (254, 224, 144), \
           (255, 255, 191), \
           (224, 243, 248), \
           (171, 217, 233), \
           (116, 173, 209), \
           (69, 117, 180) ]  
cval_fracs = list(map(rgb_to_fractionrgb, cvals))
chexes = list(map(mcolors.to_hex, cval_fracs))
mcolors.get_named_colors_mapping().update(zip(cnames, chexes))

# If the PiYG Brewer colormap already available in matplotlib
# isn't suitable, it can be recreated with the following 9 colors from http://mkweb.bcgsc.ca/brewer/
# Values are in RGB values
cnames = [ 'cbf_piyg-9-seq-1', \
           'cbf_piyg-9-seq-2', \
           'cbf_piyg-9-seq-3', \
           'cbf_piyg-9-seq-4', \
           'cbf_piyg-9-seq-5', \
           'cbf_piyg-9-seq-6', \
           'cbf_piyg-9-seq-7', \
           'cbf_piyg-9-seq-8', \
           'cbf_piyg-9-seq-9', ]
cvals =  [ (197, 27, 125), \
           (222, 119, 174), \
           (241, 182, 218), \
           (253, 224, 239), \
           (247, 247, 247), \
           (230, 245, 208), \
           (184, 225, 134), \
           (127, 188, 65), \
           (77, 146, 33) ]  
cval_fracs = list(map(rgb_to_fractionrgb, cvals))
chexes = list(map(mcolors.to_hex, cval_fracs))
mcolors.get_named_colors_mapping().update(zip(cnames, chexes))
