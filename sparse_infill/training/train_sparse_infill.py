#!/bin/env python

## IMPORT

# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
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
from sparseinfill import SparseInfill
from sparseinfilldata import load_infill_larcvdata
from loss_sparse_infill import SparseInfillLoss

# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False

CHECKPOINT_FILE="test_y_5000.tar"
INPUTFILE_TRAIN=["/mnt/disk1/nutufts/kmason/data/sparseinfill_data_train.root"]
INPUTFILE_VALID="/mnt/disk1/nutufts/kmason/data/sparseinfill_data_valid.root"
TICKBACKWARD=False
PLANE = 2
start_iter  = 0
num_iters   = 15000
IMAGE_WIDTH=496
IMAGE_HEIGHT=512
BATCHSIZE_TRAIN=1#20
BATCHSIZE_VALID=1 #10
NWORKERS_TRAIN=1
NWORKERS_VALID=1
ADC_THRESH=0.0
DEVICE_IDS=[0]
GPUID=DEVICE_IDS[0]
# map multi-training weights
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=500
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

def main():

    global best_prec1
    global writer
    global num_iters

    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(DEVICE_IDS[0]))
    else:
        DEVICE = torch.device("cpu")

    # create model, mark it to run on the GPU
    imgdims = 2
    ninput_features  = 16
    noutput_features = 16
    nplanes = 5
    reps = 1
    # self, inputshape, reps, nin_features, nout_features, nplanes,show_sizes
    model = SparseInfill( (IMAGE_HEIGHT,IMAGE_WIDTH), reps,
                           ninput_features, noutput_features,
                           nplanes, show_sizes=False).to(DEVICE)

    # Resume training option
    if RESUME_FROM_CHECKPOINT:
        print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
        checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
        best_prec1 = checkpoint["best_prec1"]
        if CHECKPOINT_FROM_DATA_PARALLEL:
            model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        model.load_state_dict(checkpoint["state_dict"])

    if not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
        model = nn.DataParallel( model, device_ids=DEVICE_IDS ).to(device=DEVICE) # distribute across device_ids

    # uncomment to dump model
    if False:
        print "Loaded model: ",model
        return

    # define loss function (criterion) and optimizer
    criterion = SparseInfillLoss().to(device=DEVICE)

    # training parameters
    lr = 1.0e-4
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchsize_train = BATCHSIZE_TRAIN
    batchsize_valid = BATCHSIZE_VALID#*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    iter_per_epoch = None # determined later
    iter_per_valid = 10


    nbatches_per_itertrain = 5
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = -1

    nbatches_per_itervalid = 5
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = -1

    # SETUP OPTIMIZER

    # SGD w/ momentum
    #optimizer = torch.optim.SGD(model.parameters(), lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)

    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    #optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=lr,
    #                             weight_decay=weight_decay)
    # RMSPROP
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = True

    # LOAD THE DATASET
    iotrain = load_infill_larcvdata( "train_adc", INPUTFILE_TRAIN,
                                      BATCHSIZE_TRAIN, NWORKERS_TRAIN,
                                      input_producer_name="ADCMasked",
                                      true_producer_name="ADC",
                                      plane = 0,
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )
    iovalid = load_infill_larcvdata( "valid_adc", INPUTFILE_TRAIN,
                                      BATCHSIZE_TRAIN, NWORKERS_TRAIN,
                                      input_producer_name="ADCMasked",
                                      true_producer_name="ADC",
                                      plane = 0,
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )

    print "pause to give time to feeders"

    NENTRIES = len(iotrain)
    #NENTRIES = 100000
    print "Number of entries in training set: ",NENTRIES

    if NENTRIES>0:
        iter_per_epoch = NENTRIES/(itersize_train)
        if num_iters is None:
            # we set it by the number of request epochs
            num_iters = (epochs-start_epoch)*NENTRIES
        else:
            epochs = num_iters/NENTRIES
    else:
        iter_per_epoch = 1

    print "Number of epochs: ",epochs
    print "Iter per epoch: ",iter_per_epoch

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        # Resume training option
        if RESUME_FROM_CHECKPOINT:
           print "RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE
           checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS )
           best_prec1 = checkpoint["best_prec1"]
           model.load_state_dict(checkpoint["state_dict"])
           optimizer.load_state_dict(checkpoint['optimizer'])
        # if GPUMODE:
        #    optimizer.cuda(GPUID)

        for ii in range(start_iter, num_iters):

            adjust_learning_rate(optimizer, ii, lr)
            print "MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),
            for param_group in optimizer.param_groups:
                print "lr=%.3e"%(param_group['lr']),
                print

            # train for one iteration
            try:
                _ = train(iotrain, DEVICE, BATCHSIZE_TRAIN, model,
                          criterion, optimizer,
                          nbatches_per_itertrain, ii, trainbatches_per_print)

            except Exception,e:
                print "Error in training routine!"
                print e.message
                print e.__class__.__name__
                traceback.print_exc(e)
                break

            # evaluate on validation set
            if ii%iter_per_valid==0 and ii>0:
                try:
                    totloss, acc5 = validate(iovalid, DEVICE, BATCHSIZE_VALID, model,
                              criterion, optimizer,
                              nbatches_per_itervalid, ii, validbatches_per_print)
                except Exception,e:
                    print "Error in validation routine!"
                    print e.message
                    print e.__class__.__name__
                    traceback.print_exc(e)
                    break

                # remember best prec@1 and save checkpoint
                prec1   =acc5
                is_best =  prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                # check point for best model
                if is_best:
                    print "Saving best model"
                    save_checkpoint({
                        'iter':ii,
                        'epoch': ii/iter_per_epoch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, -1)

            # periodic checkpoint
            if ii>0 and ii%ITER_PER_CHECKPOINT==0:
                print "saving periodic checkpoint"
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, ii)
            # flush the print buffer after iteration
            sys.stdout.flush()

        # end of profiler context
        print "saving last state"
        save_checkpoint({
            'iter':num_iters,
            'epoch': num_iters/iter_per_epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False, num_iters)


    print "FIN"
    print "PROFILER"
    if RUNPROFILER:
        print prof
    writer.close()


def train(train_loader, device, batchsize, model, criterion, optimizer, nbatches, iiter, print_freq):

    global writer

    # timers for profiling
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()

    # accruacy and loss meters

    accnames = ("infillacc2",
                "infillacc5",
                "infillacc10",
                "infillacc20")

    acc_meters  = {}
    for n in accnames:
        acc_meters[n] = AverageMeter()

    lossnames = ("total" ,
                "nondeadloss",
                "deadnochargeloss",
                "deadlowchargeloss",
                "deadhighchargeloss")

    loss_meters = {}
    for l in lossnames:
        loss_meters[l] = AverageMeter()

    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    # switch to train mode
    model.train()

    nnone = 0
    for i in range(0,nbatches):
        #print "iiter ",iiter," batch ",i," of ",nbatches
        batchstart = time.time()

        # GET THE DATA
        end = time.time()
        time_meters["data"].update(time.time()-end)

        infilldict = train_loader.get_tensor_batch(device)
        coord_t  = infilldict["coord"]
        input_t = infilldict["ADCMasked"]
        true_t = infilldict["ADC"]

        # compute output
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()

        predict_t = model(coord_t, input_t,batchsize )

        nondeadloss, deadnochargeloss, deadlowchargeloss, deadhighchargeloss, totloss = criterion(predict_t, true_t, input_t)
        # print "Loss: ", loss.item()

        if RUNPROFILER:
            torch.cuda.synchronize()
        time_meters["forward"].update(time.time()-end)

        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        optimizer.zero_grad()
        totloss.backward()
        optimizer.step()
        if RUNPROFILER:
            torch.cuda.synchronize()
        time_meters["backward"].update(time.time()-end)

        # measure accuracy and record loss
        end = time.time()

        # update loss meters
        loss_meters["total"].update( totloss.item() )
        loss_meters["nondeadloss"].update( nondeadloss.item() )
        loss_meters["deadnochargeloss"].update( deadnochargeloss.item() )
        loss_meters["deadlowchargeloss"].update( deadlowchargeloss.item() )
        loss_meters["deadhighchargeloss"].update( deadhighchargeloss.item() )

        # measure accuracy and update meters
        acc_values = accuracy(predict_t.detach(),
                         true_t.detach(),
                         input_t.detach(),
                         acc_meters,True)


        # update time meter
        time_meters["accuracy"].update(time.time()-end)

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)

        # print status
        if print_freq>0 and i%print_freq == 0:
            prep_status_message( "train-batch", i, acc_meters, loss_meters, time_meters,True )

    prep_status_message( "Train-Iteration", iiter, acc_meters, loss_meters, time_meters, True )

    # write to tensorboard
    loss_scalars = { x:y.avg for x,y in loss_meters.items() }
    writer.add_scalars('data/train_loss', loss_scalars, iiter )

    acc_scalars = { x:y.avg for x,y in acc_meters.items() }
    writer.add_scalars('data/train_accuracy', acc_scalars, iiter )

    return loss_meters['total'].avg


def validate(val_loader, device, batchsize, model, criterion, optimizer, nbatches, iiter, print_freq):
    """
    inputs
    ------
    val_loader: instance of LArCVDataSet for loading data
    batchsize (int): image (sets) per batch
    model (pytorch model): network
    criterion (pytorch module): loss function
    nbatches (int): number of batches to process
    print_freq (int): number of batches before printing output
    iiter (int): current iteration number of main loop

    outputs
    -------
    average percent of predictions within 5 pixels of truth
    """
    global writer



    # accruacy and loss meters
    accnames = ("infillacc2",
                "infillacc5",
                "infillacc10",
                "infillacc20")

    acc_meters  = {}
    for n in accnames:
        acc_meters[n] = AverageMeter()

    lossnames = ("total" ,
                "nondeadloss",
                "deadnochargeloss",
                "deadlowchargeloss",
                "deadhighchargeloss")

    loss_meters = {}
    for l in lossnames:
        loss_meters[l] = AverageMeter()

    # timers for profiling
    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    # switch to evaluate mode
    model.eval()

    iterstart = time.time()
    nnone = 0
    for i in range(0,nbatches):
        batchstart = time.time()

        tdata_start = time.time()

        infilldict = val_loader.get_tensor_batch(device)
        coord_t  = infilldict["coord"]
        input_t = infilldict["ADCMasked"]
        true_t = infilldict["ADC"]

        time_meters["data"].update( time.time()-tdata_start )

        # compute output
        tforward = time.time()
        predict_t = model(coord_t, input_t,batchsize )
        nondeadloss, deadnochargeloss, deadlowchargeloss, deadhighchargeloss, totloss = criterion(predict_t, true_t, input_t)

        time_meters["forward"].update(time.time()-tforward)

        # measure accuracy and update meters
        # update loss meters
        loss_meters["total"].update( totloss.item() )
        loss_meters["nondeadloss"].update( nondeadloss.item() )
        loss_meters["deadnochargeloss"].update( deadnochargeloss.item() )
        loss_meters["deadlowchargeloss"].update( deadlowchargeloss.item() )
        loss_meters["deadhighchargeloss"].update( deadhighchargeloss.item() )


        # measure accuracy and update meters
        acc_values = accuracy(predict_t.detach(),
                         true_t.detach(),
                         input_t.detach(),
                         acc_meters,True)


        # update time meter
        end = time.time()
        time_meters["accuracy"].update(time.time()-end)

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)

        # measure elapsed time for batch
        time_meters["batch"].update( time.time()-batchstart )
        if print_freq>0 and i % print_freq == 0:
            prep_status_message( "valid-batch", i, acc_meters, loss_meters, time_meters, False )


    prep_status_message( "Valid-Iter", iiter, acc_meters, loss_meters, time_meters, False )

    # write to tensorboard
    loss_scalars = { x:y.avg for x,y in loss_meters.items() }
    writer.add_scalars('data/valid_loss', loss_scalars, iiter )

    acc_scalars = { x:y.avg for x,y in acc_meters.items() }
    writer.add_scalars('data/valid_accuracy', acc_scalars, iiter )

    return loss_meters['total'].avg,acc_meters['infillacc5'].avg

def save_checkpoint(state, is_best, p, filename='checkpoint2.pth.tar'):
    if p>0:
        filename = "checkpoint2.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(predict,true,input,acc_meters,istrain):
    """Computes the accuracy metrics."""
    # inputs:
    #  assuming all pytorch tensors
    # metrics:
    #  (20,10,5,2) pixel accuracy fraction. adc within X value of target.

    if istrain:
        accvals = (2,5,10,20)
    else:
        accvals = (2,5,10,20)

    profile = False

    # needs to be as gpu as possible!
    if profile:
        start = time.time()
    labels = input.eq(0).float()
    chargelabel = labels*(true>0).float()
    totaldeadcharge = chargelabel.sum().float()
    totaldead = labels.sum().float()
    predictdead = labels*predict
    truedead = true*labels
    predictcharge = chargelabel*predict
    truecharge = chargelabel*true
    err =(predictcharge-truecharge).abs()

    for level in accvals:
        name = "infillacc%d"%(level)
        acc_meters[name].update( (err.lt(level).float()*chargelabel.float()).sum().item()/totaldeadcharge )


    if profile:
        torch.cuda.synchronize()
        start = time.time()

    return acc_meters["infillacc2"],acc_meters["infillacc5"],acc_meters["infillacc10"],acc_meters["infillacc20"]
def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print "Epoch [%d] lr=%.3e"%(epoch,lr)
    print "Epoch [%d] lr=%.3e"%(epoch,lr)
    return

def prep_status_message( descripter, iternum, acc_meters, loss_meters, timers, istrain ):
    print "------------------------------------------------------------------------"
    print " Iter[",iternum,"] ",descripter
    print "  Time (secs): iter[%.2f] batch[%.3f] Forward[%.3f/batch] Backward[%.3f/batch] Acc[%.3f/batch] Data[%.3f/batch]"%(timers["batch"].sum,
                                                                                                                             timers["batch"].avg,
                                                                                                                             timers["forward"].avg,
                                                                                                                             timers["backward"].avg,
                                                                                                                             timers["accuracy"].avg,
                                                                                                                             timers["data"].avg)
    print "  Loss: Total[%.2f]"%(loss_meters["total"].avg)
    print "  Accuracy: <2[%.1f] <5[%.1f] <10[%.1f] <20[%.1f]"%(acc_meters["infillacc2"].avg*100,acc_meters["infillacc5"].avg*100,acc_meters["infillacc10"].avg*100,acc_meters["infillacc20"].avg*100)

    print "------------------------------------------------------------------------"


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
