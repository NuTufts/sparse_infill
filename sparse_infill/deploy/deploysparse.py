#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT as rt
from ROOT import TH1F,TTree,TFile,TH2F,TCanvas,TLine,TAttFill,TPad,TLegend
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

# tensorboardX
from tensorboardX import SummaryWriter

# dataset interface
sys.path.append("/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larflow/larcvdataset")
from larcvdataset.larcvserver import LArCVServer

sys.path.append("../models")
sys.path.append("../training")
from sparseinfill import SparseInfill
from sparseinfilldata import load_infill_larcvdata
from loss_sparse_infill import SparseInfillLoss

# first set variables
CHECKPOINT_MAP_LOCATIONS= {"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
batchsize = 1
nworkers  = 3
tickbackward = False
readonly_products=None
nentries = 10
plane = 1
image_height = 512
image_width = 496
# model params:
imgdims = 2
ninput_features  = 16
noutput_features = 16
nplanes = 5
reps = 1
# device variables
DEVICE_IDS=[0,1]
GPUID=DEVICE_IDS[0]
GPUMODE = True

def pixelloop(true_t,coord_t,predict_t, input_t,
        ADC_img,Input_img,Output_img, Overlay_img,
        trueadc_h,predictadc_h,diff2d_h,diffs2d_thresh_h):

    for i in range(0,true_t.detach().cpu().size()[0]):
        row = coord_t[i,0].item()
        col = coord_t[i,1].item()
        trueval = true_t[i].item()
        predval = predict_t[i].item()
        inputval = input_t[i].item()

        # fill root histograms (no threshold yet)
        if (inputval == 0): #dead channel
            trueadc_h.Fill(trueval)
            predictadc_h.Fill(predval)
            if (trueval > 0):
                diff2d_h.Fill(trueval,predval)
                diffs2d_thresh_h.Fill(trueval,predval)

        # threshold
        if (predval < 10):
            predval = 0
        if (trueval < 10):
            trueval = 0

        ADC_img.set_pixel(col,row,trueval)
        Input_img.set_pixel(col,row,inputval)
        Output_img.set_pixel(col,row,predval)

        # create overlay image
        if (inputval == 0):
            Overlay_img.set_pixel(col,row,predval)
        else:
            Overlay_img.set_pixel(col,row,trueval)


    return ADC_img,Input_img,Output_img,Overlay_img,trueadc_h,predictadc_h,diff2d_h,diffs2d_thresh_h

def main():
    inputfiles = ["/mnt/disk1/nutufts/kmason/data/sparseinfill_data_test.root"]
    outputfile = ["sparseoutput.root"]
    CHECKPOINT_FILE = "../training/vplane_24000.tar"

    trueadc_h = TH1F( 'adc value' , 'adc value', 100, 0., 100.)
    predictadc_h = TH1F( 'adc value' , 'adc value', 100, 0., 100.)
    diffs2d_thresh_h = TH2F('h2' , 'diff2d', 90,10.,100.,90,10.,100.)
    diff2d_h = TH2F('h2' , 'diff2d', 100,0.,100.,100,0.,100.)



    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(DEVICE_IDS[0]))
    else:
        DEVICE = torch.device("cpu")

    iotest = load_infill_larcvdata( "infillsparsetest", inputfiles, batchsize, nworkers,"ADCMasked", "ADC" ,plane,
                                     tickbackward=tickbackward,
                                     readonly_products=readonly_products)

    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( inputfiles[0])
    inputmeta.initialize()
    # setup model
    model = SparseInfill( (image_height,image_width), reps,
                            ninput_features, noutput_features,
                            nplanes, show_sizes=False).to(DEVICE)

    # load checkpoint data
    checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
    best_prec1 = checkpoint["best_prec1"]
    model.load_state_dict(checkpoint["state_dict"])

    tstart = time.time()
    # values for average accuracies
    totacc2 = 0
    totacc5 = 0
    totacc10 = 0
    totacc20 = 0
    totalbinacc = 0

    # output IOManager
    outputdata = larcv.IOManager( larcv.IOManager.kWRITE, "IOManager", larcv.IOManager.kTickForward )
    outputdata.set_out_file( "sparseoutput.root" )
    outputdata.initialize()

    # save to output file
    ev_out_ADC = outputdata.get_data(larcv.kProductImage2D,"ADC")
    ev_out_input = outputdata.get_data(larcv.kProductImage2D,"Input")
    ev_out_output = outputdata.get_data(larcv.kProductImage2D,"Output")
    ev_out_overlay = outputdata.get_data(larcv.kProductImage2D,"Overlay")

    totaltime = 0
    for n in xrange(nentries):
        starttime =time.time()
        print "On entry: ", n
        inputmeta.read_entry(n)
        ev_meta   = inputmeta.get_data(larcv.kProductSparseImage,"ADC")
        outmeta   = ev_meta.SparseImageArray()[0].meta_v()
        model.eval()

        infilldict = iotest.get_tensor_batch(DEVICE)
        coord_t  = infilldict["coord"]
        input_t = infilldict["ADCMasked"]
        true_t = infilldict["ADC"]

        # run through model
        predict_t = model(coord_t, input_t, 1)
        forwardpasstime = time.time()

        predict_t.detach().cpu().numpy()
        input_t.detach().cpu().numpy()
        true_t.detach().cpu().numpy()

        # calculate accuracies
        labels = input_t.eq(0).float()
        chargelabel = labels*(true_t>0).float()
        totaldeadcharge = chargelabel.sum().float()
        totaldead = labels.sum().float()
        predictdead = labels*predict_t
        truedead = true_t*labels
        predictcharge = chargelabel*predict_t
        truecharge = chargelabel*true_t
        err =(predictcharge-truecharge).abs()

        totacc2 +=  (err.lt(2).float()*chargelabel.float()).sum().item()/totaldeadcharge
        totacc5 +=  (err.lt(5).float()*chargelabel.float()).sum().item()/totaldeadcharge
        totacc10 +=  (err.lt(10).float()*chargelabel.float()).sum().item()/totaldeadcharge
        totacc20 +=  (err.lt(20).float()*chargelabel.float()).sum().item()/totaldeadcharge

        bineq0 = (truedead.eq(0).float() * predictdead.eq(0).float()*labels).sum().item()
        bingt0 = (truedead.gt(0).float() * predictdead.gt(0).float()).sum().item()
        totalbinacc += (bineq0+bingt0)/totaldead

        # construct dense images
        ADC_img = larcv.Image2D(image_width,image_height)
        Input_img = larcv.Image2D(image_width,image_height)
        Output_img = larcv.Image2D(image_width,image_height)
        Overlay_img = larcv.Image2D(image_width,image_height)

        ADC_img,Input_img,Output_img,Overlay_img,trueadc_h,predictadc_h,diff2d_h, diffs2d_thresh_h = pixelloop(
                                                    true_t,coord_t,predict_t, input_t,
                                                    ADC_img,Input_img,Output_img,Overlay_img,
                                                    trueadc_h,predictadc_h,diff2d_h,diffs2d_thresh_h)

        ev_out_ADC.Append( ADC_img )
        ev_out_input.Append( Input_img )
        ev_out_output.Append( Output_img )
        ev_out_overlay.Append( Overlay_img )

        outputdata.set_id( ev_meta.run(), ev_meta.subrun(), ev_meta.event() )
        outputdata.save_entry()
        endentrytime = time.time()
        print "total entry time: ", endentrytime-starttime
        print "forward pass time: ", forwardpasstime-starttime
        totaltime+=forwardpasstime-starttime

    avgacc2 = (totacc2/nentries)*100
    avgacc5 = (totacc5/nentries)*100
    avgacc10 = (totacc10/nentries)*100
    avgacc20 = (totacc20/nentries)*100
    avgbin = (totalbinacc/nentries)*100

    tend = time.time()-tstart
    print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
    print "average forward pass time: ", totaltime/nentries
    print "--------------------------------------------------------------------"
    print " For dead pixels that should have charge..."
    print "<2 ADC of true: ", avgacc2.item() , "%"
    print "<5 ADC of true: ", avgacc5.item() , "%"
    print "<10 ADC of true: ", avgacc10.item() , "%"
    print "<20 ADC of true: ", avgacc20.item() , "%"
    print "binary acc in dead: ", avgbin.item() , "%"
    print "--------------------------------------------------------------------"

    # create canvas to save as pngs

    # ADC values
    rt.gStyle.SetOptStat(0)
    c1 = TCanvas("ADC Values", "ADC Values", 600, 600)
    trueadc_h.GetXaxis().SetTitle("ADC Value")
    trueadc_h.GetYaxis().SetTitle("Number of pixels")
    c1.UseCurrentStyle()
    trueadc_h.SetLineColor(632)
    predictadc_h.SetLineColor(600)
    c1.SetLogy()
    trueadc_h.Draw()
    predictadc_h.Draw("SAME")
    legend = TLegend(0.1,0.7,0.48,0.9);
    legend.AddEntry(trueadc_h,"True Image","l");
    legend.AddEntry(predictadc_h,"Output Image","l");
    legend.Draw();
    c1.SaveAs(("ADCValues.png"))

    # 2d ADC difference histogram
    c2 = TCanvas("diffs2D", "diffs2D", 600, 600)
    c2.UseCurrentStyle()
    line = TLine(0,0,80,80)
    line.SetLineColor(632)
    diff2d_h.SetOption("COLZ")
    c2.SetLogz()
    diff2d_h.GetXaxis().SetTitle("True ADC value")
    diff2d_h.GetYaxis().SetTitle("Predicted ADC value")
    diff2d_h.Draw()
    line.Draw()
    c2.SaveAs(("diffs2d.png"))

    # 2d ADC difference histogram - thresholded
    c3 = TCanvas("diffs2D_thresh", "diffs2D_thresh", 600, 600)
    c3.UseCurrentStyle()
    line = TLine(10,10,80,80)
    line.SetLineColor(632)
    diffs2d_thresh_h.SetOption("COLZ")
    diffs2d_thresh_h.GetXaxis().SetTitle("True ADC value")
    diffs2d_thresh_h.GetYaxis().SetTitle("Predicted ADC value")
    diffs2d_thresh_h.Draw()
    line.Draw()
    c3.SaveAs(("diffs2d_thresh.png"))
    # save results
    outputdata.finalize()


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
