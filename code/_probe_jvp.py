# -*- coding: utf-8 -*-
"""Probe: is torch.func.jvp usable through the layer (CG-LS inside), and does an
eigsh-based sigma_max converge to a self-consistent value (>= block maxima)?"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bjac import make_f, flat_of, THETA
from run_experiments import load_data, init_state, NX, NY, N_INC
from isp_core import UnrolledISP
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

t0 = time.time()
m, y, chi0, w0 = load_data(None) if False else (None,)*4
