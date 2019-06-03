import os,sys

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

# sparse submanifold convnet library
import sparseconvnet as scn
# -------------------------------------------------------------------------
# HolePixelLoss
# This loss mimics nividia's pixelwise loss for holes (L1)
# used in the infill network
# how well does the network do in dead regions?
# -------------------------------------------------------------------------

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class SparseInfillLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=False, ignore_index=-100 ):
        super(SparseInfillLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        self.size_average = size_average
        #self.mean = torch.mean.cuda()

    def forward(self,predict,adc,input):
        """
        predict: (b,1,h,w) tensor with output from logsoftmax
        adc:  (b,h,w) tensor with true adc values

        """
        _assert_no_grad(adc)
        # print "size of predict: ",predict.size()
        # print "size of adc: ",adc.size()[0]

        # want three losses: non-dead, dead w/o charge, dead w/ charge
        L1loss=torch.nn.L1Loss(self.size_average)
        nondeadweight = 1.0
        deadnochargeweight = 1000.0
        deadlowchargeweight = 100.0
        deadhighchargeweight = 1000.0

        goodch = (input > 0).float()
        predictgood = goodch * predict
        adcgood = goodch* adc
        totnondead = goodch.sum().float()
        if (totnondead == 0):
                totnondead = 1.0
        nondeadloss = (L1loss(predictgood, adcgood)*nondeadweight)/totnondead

        deadch = (input.eq(0)).float()

        deadchhighcharge = deadch * (adc > 40).float()
        predictdeadhighcharge = predict*deadchhighcharge
        adcdeadhighcharge = adc*deadchhighcharge
        totdeadhighcharge = deadchhighcharge.sum().float()
        if (totdeadhighcharge == 0):
                totdeadhighcharge = 1.0
        deadhighchargeloss = (L1loss(predictdeadhighcharge,adcdeadhighcharge)*deadhighchargeweight)/totdeadhighcharge

        deadchlowcharge = deadch * (adc > 10).float() *(adc<40).float()
        predictdeadlowcharge = predict*deadchlowcharge
        adcdeadlowcharge = adc*deadchlowcharge
        totdeadlowcharge = deadchlowcharge.sum().float()
        if (totdeadlowcharge == 0):
                totdeadlowcharge = 1.0
        deadlowchargeloss = (L1loss(predictdeadlowcharge,adcdeadlowcharge)*deadlowchargeweight)/totdeadlowcharge

        deadchnocharge = deadch * (adc<10).float()
        predictdeadnocharge = predict*deadchnocharge
        adcdeadnocharge = adc*deadchnocharge
        totdeadnocharge = deadchnocharge.sum().float()
        if (totdeadnocharge == 0):
                totdeadnocharge = 1.0
        deadnochargeloss = (L1loss(predictdeadnocharge,adcdeadnocharge)*deadnochargeweight)/totdeadnocharge

        totloss = nondeadloss + deadnochargeloss + deadlowchargeloss +deadhighchargeloss
        return nondeadloss, deadnochargeloss, deadlowchargeloss, deadhighchargeloss, totloss
