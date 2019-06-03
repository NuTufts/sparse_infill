import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
sys.path.append("/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larflow/larcvdataset")
from larcvdataset.larcvserver import LArCVServer
import torch
from torch.utils import data as torchdata
from load_cropped_sparse_infill import load_cropped_sparse_infill



def load_infill_larcvdata( name, inputfile, batchsize, nworkers,
                            input_producer_name,true_producer_name, plane,
                            tickbackward=False, readonly_products=None):
    feeder = SparseInfillPyTorchDataset(inputfile, batchsize,
                                         input_producer_name=input_producer_name,
                                         true_producer_name=true_producer_name,
                                         plane=plane,
                                         tickbackward=tickbackward, nworkers=nworkers,
                                         readonly_products=readonly_products,
                                         feedername=name)
    return feeder


###################################
## SparseLArFlowPyTorchDataset
###################################
class SparseInfillPyTorchDataset(torchdata.Dataset):
    idCounter = 0
    def __init__(self,inputfile,batchsize,input_producer_name, true_producer_name,
                 plane, tickbackward=False,nworkers=4,
                 readonly_products=None,
                 feedername=None):
        super(SparseInfillPyTorchDataset,self).__init__()

        if type(inputfile) is str:
            self.inputfiles = [inputfile]
        elif type(inputfile) is list:
            self.inputfiles = inputfile

        if type(input_producer_name) is not str:
            raise ValueError("producer_name type must be str")

        # get length by querying the tree
        self.nentries  = 0
        tchain = rt.TChain("sparseimg_{}_tree".format(input_producer_name))
        for finput in self.inputfiles:
            tchain.Add(finput)
        self.nentries = tchain.GetEntries()
        #print "nentries: ",self.nentries
        del tchain

        if feedername is None:
            self.feedername = "SparseInfillImagePyTorchDataset_%d"%\
                                (SparseImagePyTorchDataset.idCounter)
        else:
            self.feedername = feedername
        self.batchsize = batchsize
        self.nworkers  = nworkers
        readonly_products = None
        params = {"inputproducer":input_producer_name, "trueproducer":true_producer_name,"plane":plane}

        # note, with way LArCVServer workers, must always use batch size of 1
        #   because larcvserver expects entries in each batch to be same size,
        #   but in sparse representations this is not true
        # we must put batches together ourselves for sparseconv operations
        self.feeder = LArCVServer(1,self.feedername,
                                  load_cropped_sparse_infill,
                                  self.inputfiles,self.nworkers,
                                  server_verbosity=-1,worker_verbosity=-1,
                                  io_tickbackward=tickbackward,
                                  func_params=params)

        SparseInfillPyTorchDataset.idCounter += 1

    def __len__(self):
        #print "return length of sample:",self.nentries
        return self.nentries

    def __getitem__(self,index):
        """ we do not have a way to get the index (can change that)"""
        #print "called get item for index=",index," ",self.feeder.identity,"pid=",os.getpid()
        data = self.feeder.get_batch_dict()
        # remove the feeder variable
        del data["feeder"]
        #print "called get item: ",data.keys()
        return data


    def get_tensor_batch(self,device):
        """
        get batch, convert into torch tensors

        inputs
        -------
        device: torch.device specifies either gpu or cpu

        output
        -------
        data [dict of torch tensors]
        """

        # we will fill this dict to return with batch
        datalen   = [] # store length of each sparse data instance
        ncoords   = 0  # total number of points over all batches

        # first collect data
        data_v = []
        for ibatch in xrange(self.batchsize):
            batch = None
            ntries = 0
            while batch is None and ntries<10:
                batch = self.feeder.get_batch_dict()
                ntries += 1
            if batch is not None:
                data_v.append( batch )


        # now calc total points in each sparse image instance
        for data in data_v:
            datalen.append( data["ADCMasked"][0].shape[0] )
            ncoords += datalen[-1]
        # print "NCOORDS: ",ncoords
        # print "shape: ", data["ADCMasked"][0].shape

        # if len(data_v)>0 and data_v[0]["ADC"][0] is not None:
        #     has_truth = True
        # else:
        #     has_truth = False
        has_truth = True

        # make tensor for coords (row,col,batch)
        coord_t = torch.zeros( (ncoords,3), dtype=torch.int ).to(device)

        # tensor for input pixel s
        input_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)

        # tensor for true values
        if has_truth:
            truth_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        else:
            truth_t = None

        # fill tensors above
        nfilled = 0
        for ib,batch in enumerate(data_v):
            srcpix    = batch["ADCMasked"][0]
            # print type(srcpix),
            # print srcpix.shape," "

            start = nfilled
            end   = nfilled+datalen[ib]
            coord_t[start:end,0:2] \
                = torch.from_numpy( srcpix[:,0:2].astype(np.int) )
            coord_t[start:end,2]        = ib
            # print coord_t.shape," "
            input_t[start:end,0]        = torch.from_numpy(srcpix[:,2])

            if has_truth:
                truepix    = batch["ADC"][0]
                truth_t[start:end,0]      = torch.from_numpy(truepix[:,2])

            nfilled += datalen[ib]

        flowdata = {"coord":coord_t, "ADCMasked":input_t, "ADC":truth_t}

        return flowdata


if __name__== "__main__":

    # "testing"
    inputfiles = ["/mnt/disk1/nutufts/kmason/data/sparseinfill_data_test.root"]
    batchsize = 10
    nworkers  = 3
    tickbackward = True
    readonly_products=None
    nentries = 10
    plane = 2
    #
    TEST_VANILLA = True
    #
    if TEST_VANILLA:

        feeder = load_infill_larcvdata( "infillsparsetest", inputfiles, batchsize, nworkers,"ADCMasked", "ADC" ,plane,
                                         tickbackward=tickbackward,
                                         readonly_products=readonly_products)
        tstart = time.time()

        print "TEST Infill LARCVDATASET SERVER"
        for n in xrange(nentries):
            print "=============================================="
            batch = feeder.get_tensor_batch(torch.device("cpu"))
            print "ENTRY[",n,"] from ",batch.keys()
            for name,arr in batch.items():
                print "  ",name," ",type(arr),": npts=",len(arr),"; ",
                if type(arr) is np.ndarray or type(arr) is torch.Tensor:
                    print arr.shape,
                print

        tend = time.time()-tstart
        print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
        del feeder
