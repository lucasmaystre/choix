import choix
import numpy as np

from testutils import data_path, parse_pairwise


LOGIT_TRUE_MEAN = np.array(
    [-2.24876774, -0.54278959, 0.5427896, 2.24876774])
LOGIT_TRUE_COV = np.array(
    [[ 5.13442507, 1.78941794, 0.7363867,  0.33977029],
     [ 1.78941794, 3.87821895, 1.59597641, 0.73638669],
     [ 0.7363867,  1.59597641, 3.87821895, 1.78941794],
     [ 0.33977029, 0.73638669, 1.78941794, 5.13442508]])


def test_ep_logit():
    with open(data_path("simpletrans-4.dat")) as f:
        num_items, data = parse_pairwise(f.read())
    mean, cov = choix.ep_pairwise(num_items, data, 8.0, model='logit')
    assert np.allclose(mean, LOGIT_TRUE_MEAN)
    assert np.allclose(cov, LOGIT_TRUE_COV)
