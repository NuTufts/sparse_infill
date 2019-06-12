#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
//#include "/usr/include/python2.7/unicodeobject.h"

#include "numpy/arrayobject.h"
// larcv
#include "/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larcv/larcv/core/DataFormat/EventImage2D.h"
#include "/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larcv/larcv/core/DataFormat/EventChStatus.h"
#include "/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larcv/larcv/core/DataFormat/EventSparseImage.h"
#include "/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larcv/larcv/core/DataFormat/ImageMeta.h"
#include "/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larcv/larcv/core/DataFormat/IOManager.h"

int main(){
  Py_Initialize();
  import_array1(0);
  // input IOManager
  larcv::IOManager *ioin = new larcv::IOManager(larcv::IOManager::kBOTH,"IOManager");
  ioin->add_in_file( "/mnt/disk1/nutufts/kmason/data/sparseinfill_data_test.root" );
  ioin->set_out_file("sparseinfill_cxx_test.root");
  ioin->initialize();

  // // output file
  larcv::IOManager *foutIO = new larcv::IOManager( larcv::IOManager::kWRITE,"OutManager");
  foutIO->set_out_file( "sparseinfill_cxx_test.root" );
  foutIO->initialize();

  // loop through entries
  for (int n = 0; n<ioin->get_n_entries();n++){
    ioin->read_entry(n);
    larcv::EventBase *ev_meta  = ioin->get_data(larcv::kProductSparseImage,"ADCMasked");

    auto ev_in_adcmasked = (larcv::EventSparseImage*)(ioin->get_data(larcv::kProductSparseImage, "ADCMasked"));
    if (!ev_in_adcmasked) {
       std::cout << "No Input SparseImage found with a name: " << "ADCMasked" << std::endl;
     }
     std::vector< larcv::SparseImage > img_adcmasked_v = ev_in_adcmasked->SparseImageArray();

    larcv::EventImage2D* ev_out_pred  = (larcv::EventImage2D*)foutIO->get_data(larcv::kProductImage2D,"Infill");
    ev_out_pred->clear();

    // loop through planes
    for (int plane = 0;plane<3;plane++){
      larcv::Image2D Pred_img(512,496);

      // Convert array into numpy array
      int ND = img_adcmasked_v[plane].pixellist().size();

      float npts    = ND/3;
      float c_arr[int(npts)][3];
      for (int start = 0; start <int(npts); start++){
        c_arr[start][0] = img_adcmasked_v[plane].pixellist()[start*3];
        c_arr[start][1] = img_adcmasked_v[plane].pixellist()[start*3+1];
        c_arr[start][2] = img_adcmasked_v[plane].pixellist()[start*3+2];
      }


      npy_intp dims[2]{int(npts),3};

      PyObject *pArray = PyArray_SimpleNewFromData(2,dims,NPY_FLOAT,reinterpret_cast<void*>(c_arr));

      // import forward pass python module
      PyObject *pName = PyUnicode_FromString("Infill_ForwardPass");
      PyObject *pModule = PyImport_Import(pName);
      Py_DECREF(pName);
      // std::cout<<pModule <<std::endl;

      // choose which function depending on plane
      PyObject *pFunc;
      if (plane == 0){
        pFunc = PyObject_GetAttrString(pModule,"forwardpassu");
      }
      else if (plane == 1){
        pFunc = PyObject_GetAttrString(pModule,"forwardpassv");
      }
      else{
        pFunc = PyObject_GetAttrString(pModule,"forwardpassy");
      }
      // std::cout << pFunc<<std::endl;

      PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc,pArray,NULL);
      // std::cout<<pReturn<<std::endl;
      PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);


      // converting back to c++ array
      float *c_out;
      c_out = reinterpret_cast<float*>(PyArray_DATA(np_ret));


      // save to output_img
      for (int i =0; i<int(npts);i++){
        // setting threshold at 10
        if (*(c_out + 3*i + 2) >10){
          Pred_img.set_pixel(*(c_out + 3*i + 0),*(c_out + 3*i + 1),*(c_out + 3*i + 2));}
      }

      ev_out_pred->Append(Pred_img);
    }
    foutIO->set_id( ev_meta->run(), ev_meta->subrun(), ev_meta->event() );
    foutIO->save_entry();
    std::cout<<"finished entry: "<< n <<std::endl;
  }

  foutIO->finalize();
  ioin->finalize();

  Py_Finalize();

  return 0;
}
