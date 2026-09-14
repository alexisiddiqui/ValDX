import numpy as np
import pandas as pd
import pytest

from figure_scripts.kinetic_simulation.gamma_selection import select_group


def test_selection_is_independent_of_units_and_order():
    candidates = pd.DataFrame({"gamma": [1, 2, 3, 4, 5],
                               "mse": [10, 4, 2, 1.5, 1], "work": [0, 1, 2, 5, 10]})
    assert select_group(candidates).gamma == 3
    candidates.mse = candidates.mse * .001 + 7
    candidates.work = candidates.work * 1000 + 50
    assert select_group(candidates.iloc[::-1]).gamma == 3


def test_constant_axis_ties_and_invalid_candidates():
    candidates = pd.DataFrame({"gamma": [3, 2, 1, 0],
                               "mse": [np.nan, 1, 1, 0], "work": [0, 0, 0, 0]})
    assert select_group(candidates).gamma == 1
    candidates.loc[candidates.gamma == 2, "mse"] = .5
    assert select_group(candidates).gamma == 2
    with pytest.raises(ValueError):
        select_group(candidates.iloc[:1])
