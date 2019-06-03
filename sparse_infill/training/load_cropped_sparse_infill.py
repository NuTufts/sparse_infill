import os,sys,time
import ROOT as rt
import numpy as np
from larcv import larcv

"""
Load infill sparse data
"""
def load_cropped_sparse_infill(io,input_producer= "ADCMasked", true_producer= "ADC", plane=0, threshold=0.0):
    """
    we need the input data to be a pixel list for Sparse Convolutions

    we import cropped sparseimage objects from a LArCV rootfile

    the sparseimg is assumed to have at 7 features
    (0,1): (row,col)
    2: adc value


    inputs
    ------
    io[larcv.IOManager] access to IOManager. assumed to have the entry set already.
    producer[str]       producer name
    plane[int]          plane number
    threshold[float]    threshold ADC value for source pixel

    outputs
    -------

    """

    data    = {}
    verbose = False

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    # get data from iomanager
    tio = time.time()
    ev_sparse_input = io.get_data(larcv.kProductSparseImage,input_producer)
    ev_sparse_true = io.get_data(larcv.kProductSparseImage,true_producer)
    dtio += time.time()-tio

    # get instance, convert to numpy array, nfeatures per flow
    # numpy array is (N,7) with 2nd dimension is (row,col,source_adc,target_adc,truth_flow)
    sparsedata_input = ev_sparse_input.at(plane)
    sparsedata_true = ev_sparse_true.at(plane)

    sparse_input_np  = larcv.as_ndarray( sparsedata_input, larcv.msg.kNORMAL )
    sparse_true_np  = larcv.as_ndarray( sparsedata_true, larcv.msg.kNORMAL )
    nfeatures  = sparsedata_true.nfeatures()
    #print "nfeatures: ",nfeatures
    # print "np size: ",sparse_input_np.shape

    # source meta, same for flow directions
    meta  = sparsedata_input.meta_v().front()

    # has truth, 3rd feature is truth flow

    tnpmanip  = time.time()
    data["ADCMasked"] = sparse_input_np[:,:] # (row,col,value)
    data["ADC"] = sparse_true_np[:,:] # (row,col,value)
    # print "adcmasked", data["ADCMasked"].shape
    # print "adc", data["ADC"].shape

    # check pixel Values
    numdead = (data["ADCMasked"]==0).sum()
    numdeadcharge =((data["ADCMasked"]==0)*(data["ADC"] > 0)).sum()
    numdeadnocharge =((data["ADCMasked"]==0)*(data["ADC"]==0)).sum()

    if numdead == 0 or numdeadcharge == 0 or numdeadnocharge == 0:
        return None
    # if has_truth:
    #     data["flowy2u"] = sparse_np[:,5].astype(np.float32).reshape( (sparse_np.shape[0],1) )
    #     data["flowy2v"] = sparse_np[:,6].astype(np.float32).reshape( (sparse_np.shape[0],1) )
    #
    #     if checkpix:
    #         nbadpix_y2u  = ( data["flowy2u"]<=-4000 ).sum()
    #         ngoodpix_y2u = ( data["flowy2u"]>-4000 ).sum()
    #         ngoodpix_y2v = ( data["flowy2v"]>-4000 ).sum()
    #         nbadpix_y2v  = ( data["flowy2v"]<=-4000 ).sum()
    # if checkpix and verbose:
    #     print "  ngoodpix_y2u=",ngoodpix_y2u," nbadpix_y2u=",nbadpix_y2u," tot=",ngoodpix_y2u+nbadpix_y2u," npts=",sparse_np.shape[0]
    #     print "  ngoodpix_y2v=",ngoodpix_y2v," nbadpix_y2v=",nbadpix_y2v," tot=",ngoodpix_y2v+nbadpix_y2v," npts=",sparse_np.shape[0]
    # if checkpix:
    #     if ngoodpix_y2u==0 or ngoodpix_y2v==0:
    #         return None

    dtnpmanip += time.time()-tnpmanip
    tottime = time.time()-tottime

    #
    # print "[load cropped sparse infill]"
    # print "  nfeatures=",nfeatures," npts=",sparse_input_np.shape[0]
    # print "  io time: %.3f secs"%(dtio)
    # print "  tot time: %.3f secs"%(tottime)


    return data
