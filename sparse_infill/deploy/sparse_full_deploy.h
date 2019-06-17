#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
//#include "/usr/include/python2.7/unicodeobject.h"

#include "numpy/arrayobject.h"
// larcv
#include "larcv/core/Base/ConfigManager.h"
#include "larcv/core/Base/LArCVBaseUtilFunc.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventROI.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/IOManager.h"
// ublarcvapp
#include "ublarcvapp/UBImageMod/UBSplitDetector.h"
#include "ublarcvapp/UBImageMod/InfillImageStitcher.h"
#include "ublarcvapp/UBImageMod/InfillDataCropper.h"


larcv::Image2D Netork(larcv::Image2D Pred_img,larcv::SparseImage img_adcmasked, int plane);
