#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# import larcv
from larcv import larcv

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

sys.path.append("../models")
from sparseinfill import SparseInfill

def test(a):
    print a
    return a

def forwardpassu(data_t):
    # function to load the sparse infill network and run a forward pass of one image
    # make tensor for coords (row,col,batch)
    ncoords = np.size(data_t,0)
    coord_t = torch.ones( (ncoords,3), dtype=torch.int )
    # tensor for input pixels
    input_t = torch.zeros( (ncoords,1), dtype=torch.float)

    coord_t[0:ncoords,0:2] \
        = torch.from_numpy(data_t[:,0:2].astype(np.int) )
    input_t[0:ncoords,0]  = torch.from_numpy(data_t[:,2])

    # loading model with hard coded parameters used in training
    # ( (height,width),reps,ninput_features, noutput_features,nplanes, show_sizes=False)
    CHECKPOINT_FILE = "../training/uplane_26000.tar"
    model = SparseInfill( (512,496), 1,16,16,5, show_sizes=False)

    # load checkpoint data
    checkpoint = torch.load( CHECKPOINT_FILE, {"cuda:0":"cuda:0","cuda:1":"cuda:1"} )
    best_prec1 = checkpoint["best_prec1"]
    model.load_state_dict(checkpoint["state_dict"])

    # run the forward pass
    out_t = model(coord_t, input_t, 1)
    out_t = out_t.data.numpy()
    predict_t = np.zeros( (ncoords,3), dtype=np.float32)
    predict_t[:,2]   = out_t[:,0]
    predict_t[:,0:2] = data_t[:,0:2]

    return predict_t

def forwardpassv(data_t):
    # function to load the sparse infill network and run a forward pass of one image
    # make tensor for coords (row,col,batch)
    ncoords = np.size(data_t,0)
    coord_t = torch.ones( (ncoords,3), dtype=torch.int )
    # tensor for input pixels
    input_t = torch.zeros( (ncoords,1), dtype=torch.float)

    coord_t[0:ncoords,0:2] \
        = torch.from_numpy(data_t[:,0:2].astype(np.int) )
    input_t[0:ncoords,0]  = torch.from_numpy(data_t[:,2])

    # loading model with hard coded parameters used in training
    # ( (height,width),reps,ninput_features, noutput_features,nplanes, show_sizes=False)
    CHECKPOINT_FILE = "../training/vplane_32000.tar"
    model = SparseInfill( (512,496), 1,16,16,5, show_sizes=False)

    # load checkpoint data
    checkpoint = torch.load( CHECKPOINT_FILE, {"cuda:0":"cuda:0","cuda:1":"cuda:1"} )
    best_prec1 = checkpoint["best_prec1"]
    model.load_state_dict(checkpoint["state_dict"])

    # run the forward pass
    out_t = model(coord_t, input_t, 1)
    out_t = out_t.data.numpy()
    predict_t = np.zeros( (ncoords,3), dtype=np.float32)
    predict_t[:,2]   = out_t[:,0]
    predict_t[:,0:2] = data_t[:,0:2]

    return predict_t

def forwardpassy(data_t):
    # function to load the sparse infill network and run a forward pass of one image
    # make tensor for coords (row,col,batch)
    ncoords = np.size(data_t,0)
    coord_t = torch.ones( (ncoords,3), dtype=torch.int )
    # tensor for input pixels
    input_t = torch.zeros( (ncoords,1), dtype=torch.float)

    coord_t[0:ncoords,0:2] \
        = torch.from_numpy(data_t[:,0:2].astype(np.int) )
    input_t[0:ncoords,0]  = torch.from_numpy(data_t[:,2])

    # loading model with hard coded parameters used in training
    # ( (height,width),reps,ninput_features, noutput_features,nplanes, show_sizes=False)
    CHECKPOINT_FILE = "../training/yplane_50000_1st.tar"
    model = SparseInfill( (512,496), 1,16,16,5, show_sizes=False)

    # load checkpoint data
    checkpoint = torch.load( CHECKPOINT_FILE, {"cuda:0":"cuda:0","cuda:1":"cuda:1"} )
    best_prec1 = checkpoint["best_prec1"]
    model.load_state_dict(checkpoint["state_dict"])

    # run the forward pass
    out_t = model(coord_t, input_t, 1)
    out_t = out_t.data.numpy()
    predict_t = np.zeros( (ncoords,3), dtype=np.float32)
    predict_t[:,2]   = out_t[:,0]
    predict_t[:,0:2] = data_t[:,0:2]

    return predict_t
