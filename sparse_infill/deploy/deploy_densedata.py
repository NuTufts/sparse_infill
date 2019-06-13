# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from ROOT import TH1F,TTree,TFile,TH2F,TCanvas,TLine,TAttFill,TPad,TLegend
from larcv import larcv

# pytorch
import torch

# # numba
from numba import jit

# util functions
# also, implicitly loads dependencies, pytorch larflow model definition
from infill_funcs import load_model

import time

sys.path.append("/mnt/disk1/nutufts/kmason/ubresnet/larcvdataset")
from larcvdataset import LArCVDataset


# @jit
def load_pre_cropped_data( larcvdataset_configfile, batchsize=1 ):
    larcvdataset_config="""ThreadProcessor: {
        Verbosity:3
        NumThreads: 2
        NumBatchStorage: 2
        RandomAccess: false
        InputFiles: ["/mnt/disk1/nutufts/kmason/data/crop_test.root"]
        ProcessName: ["ADC_valid","ADCmasked_valid","weights_valid","labelsbasic_valid"]
        ProcessType: ["BatchFillerImage2D","BatchFillerImage2D","BatchFillerImage2D","BatchFillerImage2D"]
        ProcessList: {
            weights_valid: {
                Verbosity:3
                ImageProducer: "Weights"
                Channels: [0]
                EnableMirror: false
            }
            ADC_valid: {
                Verbosity:3
                ImageProducer: "ADC"
                Channels: [0]
                EnableMirror: false
            }
            labelsbasic_valid: {
                Verbosity:3
                ImageProducer: "Labels"
                Channels: [0]
                EnableMirror: false
            }
            ADCmasked_valid: {
                Verbosity:3
                ImageProducer: "ADCMasked"
                Channels: [0]
                EnableMirror: false
            }
        }
    }

    """

    with open("larcv_dataloader.cfg",'w') as f:
        print >> f,larcvdataset_config
    iotest = LArCVDataset( "larcv_dataloader.cfg","ThreadProcessor") #, store_eventids=True

    return iotest


def pixelloop(within2,
    within5,
    within10,
    within20,
    chargetotal,
    labelbasic_numpy,
    weights_numpy,
    ADC_numpy,
    ADCvalue_numpy,
    overlay_numpy,
    thresh_numpy,
    h, h2, h3, h4,
    width,
    height,
    threshold):

    #loop through all pixels
    # calculate accuracies
    # create acc image
    chargetotal = 0.0
    for rows in range(height):
        for cols in range(width):
            if (plane ==0):
                factor = (53.0/43.0)
            if (plane ==1):
                factor = (52.0/43.0)
            if (plane == 2):
                factor = (59.0/48.0)
            factor = 1
            ADCvalue_numpy[ib,0,rows,cols] = ADCvalue_numpy[ib,0,rows,cols]*factor
            truepix = ADC_numpy[ib,0,rows,cols].item()
            outpix = ADCvalue_numpy[ib,0,rows,cols].item()
            weightpix = weights_numpy[ib,0,rows,cols].item()
            diff = abs(truepix - outpix)

            if labelbasic_numpy[ib,0,rows,cols].item() == 1.0:
                h3.Fill(truepix)
                h4.Fill(outpix)

                overlay_numpy[ib,0,rows,cols] = outpix
                # calculate accuracies
                if diff < 2.0 and truepix > 0:
                    within2 = within2 + 1
                if diff < 5.0 and truepix > 0:
                    within5 = within5 + 1
                if diff < 10.0 and truepix > 0:
                    within10 = within10 + 1
                if diff < 20.0 and truepix > 0:
                    within20 = within20 + 1

                # make diff histogram
                if truepix > 0:
                    # if truepix < threshold:
                    #     truepix=0
                    # if outpix < threshold:
                    #     outpix=0
                    h.Fill(truepix - outpix)
                    h2.Fill(truepix,outpix)
                    chargetotal +=1.0

            # make overlay image
            if overlay_numpy[ib,0,rows,cols] < threshold:
                overlay = 0
                overlay_numpy[ib,0,rows,cols] = 0
            else:
                overlay = overlay_numpy[ib,0,rows,cols]

            if ADC_numpy[ib,0,rows,cols] < threshold:
                ADC = 0
            else:
                ADC = ADC_numpy[ib,0,rows,cols]

            # make threshold image
            diffoverlay = abs(overlay-ADC)
            if diffoverlay < 2.0:
                thresh_numpy[ib,0,rows,cols] = 0
            elif diffoverlay < 5.0:
                thresh_numpy[ib,0,rows,cols] = 1
            elif diffoverlay < 10.0:
                thresh_numpy[ib,0,rows,cols] = 2
            elif diffoverlay < 20.0:
                thresh_numpy[ib,0,rows,cols] = 3
            else:
                thresh_numpy[ib,0,rows,cols] = 4
    if (chargetotal == 0):
        chargetotal =1.0

    return within2, within5, within10, within20, chargetotal,overlay_numpy, thresh_numpy, h, h2, h3,h4

if __name__=="__main__":
    # ARGUMENTS DEFINTION/PARSER
    start = time.time()

    # for checkpoint files for original infill for ADC see:
    # /mnt/disk1/nutufts/kmason/ubresnet/training/ on nudot

    input_larcv_filename = "/mnt/disk1/nutufts/kmason/data/crop_test.root"
    output_larcv_filename = "output_infill_dense.root"
    checkpoint_data = "/mnt/disk1/nutufts/kmason/ubdl/ublarcvserver/networks/infill/uplane_MC_40000.tar"
    batch_size = 1
    gpuid = 1
    checkpoint_gpuid = 0
    verbose = True
    nprocess_events = 228
    plane = 0

    # load data
    inputdata = load_pre_cropped_data( input_larcv_filename, batchsize=batch_size )
    inputmeta = larcv.IOManager(larcv.IOManager.kREAD )
    inputmeta.add_in_file( input_larcv_filename )
    inputmeta.initialize()
    width=496
    height=512
    threshold = 10

    # load model
    model = load_model( checkpoint_data, gpuid=gpuid, checkpointgpu=checkpoint_gpuid )
    model.to(device=torch.device("cuda:%d"%(gpuid)))
    model.eval()

    # output IOManager
    outputdata = larcv.IOManager( larcv.IOManager.kWRITE )
    outputdata.set_out_file( output_larcv_filename )
    outputdata.initialize()

    inputdata.start(batch_size)

    nevts = len(inputdata)
    if nprocess_events>=0:
        nevts = nprocess_events
    nbatches = nevts/batch_size
    if nevts%batch_size!=0:
        nbatches += 1

    # root hist to inspect diff
    f = TFile( 'test.root', 'recreate' )
    h = TH1F( 'h1' , 'diff', 150, -25., 25.)
    h2 = TH2F('h2' , 'diff2d', 100,0.,100.,100,0.,100.)
    h3 = TH1F( 'h3' , 'adc value', 100, 0., 100.)
    h4 = TH1F( 'h4' , 'adc value', 100, 0., 100.)
    t = TTree('t1', 'diff histos')

    ientry = 0
    averageacc2 = 0.0
    averageacc5 = 0.0
    averageacc10 = 0.0
    averageacc20 = 0.0

    for ibatch in range(nbatches):

        if verbose:
            print "=== [BATCH %d] ==="%(ibatch)
        #
        data = inputdata[0]
        # diff_h= array( 'diff', [ 0 ] )
        t.Branch( 'diff', h, 'diff' )


        # get input ADC(masked) images
        ADCmasked_resize = torch.from_numpy( data["ADCmasked_valid"].reshape( (batch_size,1,height,width) ) ) # source image ADC

        # get ADC images for accuracy calculation
        ADC_resize = torch.from_numpy( data["ADC_valid"].reshape( (batch_size,1,height,width) ) )

        #get labels images
        labelbasic_resize = torch.from_numpy( data["labelsbasic_valid"].reshape( (batch_size,1,height,width) ) )

        #get weights images
        weights_resize = torch.from_numpy( data["weights_valid"].reshape( (batch_size,1,height,width) ) )

        ADCmasked_t = torch.zeros(1,1,512,832)
        ADC_t = torch.zeros(1,1,512,832)
        labelbasic_t = torch.zeros(1,1,512,832)
        weights_t = torch.zeros(1,1,512,832)
        factor = 1
        for rows in range(height):
            for cols in range(width):
                if (plane ==0):
                    factor = (43.0/53.0)
                if (plane ==1):
                    factor = (43.0/52.0)
                if (plane == 2):
                    factor = (48.0/59.0)
                maskedpix = ADCmasked_resize[0,0,rows,cols].item()
                ADCmasked_t[0,0,rows,cols+8]=maskedpix*factor
                adcpix = ADC_resize[0,0,rows,cols].item()
                ADC_t[0,0,rows,cols+8]=adcpix*factor
                labelpix = labelbasic_resize[0,0,rows,cols].item()
                labelbasic_t[0,0,rows,cols+8]=labelpix
                weightspix = weights_resize[0,0,rows,cols].item()
                weights_t[0,0,rows,cols+8]=weightspix

        ADCmasked_t = ADCmasked_t.to(device=torch.device("cuda:%d"%(gpuid)))
        ADC_t = ADC_t.to(device=torch.device("cuda:%d"%(gpuid)))
        labelbasic_t = labelbasic_t.to(device=torch.device("cuda:%d"%(gpuid)))
        weights_t = weights_t.to(device=torch.device("cuda:%d"%(gpuid)))


        # run model
        pred_ADCvalue = model.forward( ADCmasked_t) #ADC_t

        # get predictions from gpu
        ADCvalue_np = pred_ADCvalue.detach().cpu().numpy().astype(np.float32)
        ADC_t = ADC_t.detach().cpu()
        labelbasic_t = labelbasic_t.detach().cpu()
        weights_t = weights_t.detach().cpu()


        for ib in range(batch_size):
            if ientry>=nevts:
                # skip last portion of last batch
                break

            # get meta
            inputmeta.read_entry(ientry)
            ev_meta   = inputmeta.get_data("image2d","ADC")
            outmeta   = larcv.ImageMeta(496, 832,496,832,plane, 0,0)


            beforeloop = time.time()

            #variables for accuracy check
            within2= 0.0
            within5= 0.0
            within10= 0.0
            within20= 0.0
            chargetotal = 0.0

            labelbasic_numpy= labelbasic_t.numpy()
            weights_numpy= weights_t.numpy()
            ADC_numpy= ADC_t.numpy()

            #save a copy of labels for use in creating diff and threshold images
            thresh_t = torch.from_numpy( data["weights_valid"].reshape( (batch_size,1,height,width) ) )
            #save a copy of adc for creating overlay
            overlay_t = torch.from_numpy( data["ADCmasked_valid"].reshape( (batch_size,1,height,width) ) )

            overlay_numpy = overlay_t.numpy()
            thresh_numpy = thresh_t.numpy()

            within2, within5, within10, within20, chargetotal, overlay_numpy, thresh_numpy, h, h2,h3,h4 = pixelloop(
                        within2,
                        within5,
                        within10,
                        within20,
                        chargetotal,
                        labelbasic_numpy,
                        weights_numpy,
                        ADC_numpy,
                        ADCvalue_np,
                        overlay_numpy,
                        thresh_numpy,
                        h, h2, h3, h4,
                        width,
                        height,
                        threshold)

            overlay_t = torch.from_numpy(overlay_numpy)
            thresh_t = torch.from_numpy(thresh_numpy)
            #calculate accuracies
            accuracy2 = (within2/chargetotal)*100
            accuracy5 = (within5/chargetotal)*100
            accuracy10 = (within10/chargetotal)*100
            accuracy20 = (within20/chargetotal)*100
            averageacc2 += accuracy2
            averageacc5 += accuracy5
            averageacc10 += accuracy10
            averageacc20 += accuracy20

            # outputdata.set_id( ev_meta.run(), ev_meta.subrun(), ev_meta.event() )
            # outputdata.save_entry()
            ientry += 1
            afterloop = time.time()

    print "-------------------------"
    print "Average Accuracies"
    print "<2: ", (averageacc2/nprocess_events)
    print "<5: ", (averageacc5/nprocess_events)
    print "<10: ", (averageacc10/nprocess_events)
    print "<20: ", (averageacc20/nprocess_events)
    print "-------------------------"

    rt.gStyle.SetOptStat(0)

    # save 2d hist as png image
    c1 = TCanvas("diffs2D", "diffs2D", 600, 600)
    c1.UseCurrentStyle()
    line = TLine(0,0,80,80)
    line.SetLineColor(632)
    h2.SetOption("COLZ")
    c1.SetLogz()
    h2.GetXaxis().SetTitle("True ADC value")
    h2.GetYaxis().SetTitle("Predicted ADC value")
    h2.Draw()
    line.Draw()
    c1.SaveAs(("diffs2d_dense.png"))


    # save ADC values as histogram
    c2 = TCanvas("ADC Values", "ADC Values", 600, 600)
    h3.GetXaxis().SetTitle("ADC Value")
    h3.GetYaxis().SetTitle("Number of pixels")
    c2.UseCurrentStyle()
    h3.SetLineColor(632)
    h4.SetLineColor(600)
    c2.SetLogy()
    h3.Draw()
    h4.Draw("SAME")
    legend2 = TLegend(0.1,0.7,0.48,0.9);
    legend2.AddEntry(h3,"True Image","l");
    legend2.AddEntry(h4,"Output Image","l");
    legend2.Draw();
    c2.SaveAs(("ADCValues_dense.png"))


    # stop input
    inputdata.stop()

    # save results
    outputdata.finalize()

    f.Write()
    f.Close()

    print "DONE."
    end = time.time()
    print "time to loop start: ", (beforeloop - start)
    print "time to do loop: ", (afterloop - beforeloop)
    print "time to finalize: ", (end - afterloop)
    print "total time: ", (end - start)
