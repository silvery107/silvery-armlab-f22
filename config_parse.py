import sys
import numpy as np
import yaml

from utils import DTYPE

def parse_dh_param_file(dh_config_file):
    assert(dh_config_file is not None)
    f_line_contents = None
    with open(dh_config_file, "r") as f:
        f_line_contents = f.readlines()

    assert(f.closed)
    assert(f_line_contents is not None)
    # maybe not the most efficient/clean/etc. way to do this, but should only have to be done once so NBD
    dh_params = np.asarray([line.rstrip().split(',') for line in f_line_contents[1:]])
    dh_params = dh_params.astype(float)
    return dh_params


def parse_pox_param_file(pox_config_file):
    assert(pox_config_file is not None)
    f_line_contents = None
    with open(pox_config_file, "r") as f:
        f_line_contents = f.readlines()

    assert(f.closed)
    assert(f_line_contents is not None)
    # maybe not the most efficient/clean/etc. way to do this, but should only have to be done once so NBD
    M_matrix = np.asarray([line.split() for line in f_line_contents[1:5]], dtype=DTYPE)
    M_matrix = M_matrix.astype(float)
    S_list = np.asarray([line.split() for line in f_line_contents[6:]], dtype=DTYPE)
    S_list = S_list.astype(float)
    return M_matrix, S_list
