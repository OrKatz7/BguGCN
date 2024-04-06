import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train import *
from utils import *
from data import *
from model import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file path')
    args = parser.parse_args()
    old_stdout = sys.stdout
    log_file = open(f"./logs/{args.config_file}.log","w")
    sys.stdout = log_file
    
    print('###################### UltraGCN Plus Plus ######################')

    print('Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items,train_mat = data_param_prepare(args.config_file)
    
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    ultragcn = UltraGCNPlusPlus(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    # ultragcn.load_state_dict(torch.load("ultragcn_gowalla_0.pt"))
    ultragcn = ultragcn.to(params['device'])
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params,train_mat)

    print('END')
    sys.stdout = old_stdout

    log_file.close()
