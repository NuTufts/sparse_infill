#include "sparse_full_deploy.h"
#include "argparse.hpp"

#include "TStyle.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"

#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"

// larcv::Image2D Network(larcv::Image2D Pred_img,
//                        larcv::SparseImage img_adcmasked,
//                        int plane,
//                        std::string weightfile ){
//   // std::cout << "In network loop" <<std::endl;
//   // Convert array into numpy array
//   int ND = img_adcmasked.pixellist().size();
//   // std::cout <<ND <<std::endl;
//   clock_t startloop = clock();

//   float npts    = ND/3;
//   float c_arr[int(npts)][3];
//   for (int start = 0; start <int(npts); start++){
//     c_arr[start][0] = img_adcmasked.pixellist()[start*3];
//     c_arr[start][1] = img_adcmasked.pixellist()[start*3+1];
//     c_arr[start][2] = img_adcmasked.pixellist()[start*3+2];
//   }


//   npy_intp dims[2]{int(npts),3};
//   // std::cout <<*c_arr<<std::endl;

//   PyObject *pArray = PyArray_SimpleNewFromData(2,dims,NPY_FLOAT,reinterpret_cast<void*>(c_arr));
//   // std::cout<<pArray<<std::endl;
//   clock_t parray = clock();
//   // import forward pass python module
//   PyObject *pName   = PyUnicode_FromString("Infill_ForwardPass");
//   PyObject *pWeight = PyUnicode_FromString(weightfile.c_str());
//   PyObject *pModule = PyImport_Import(pName);  
//   Py_DECREF(pName);
//   if ( pModule==0 || pModule==NULL || pModule==nullptr ) {
//     throw std::runtime_error("Could not import Infill_ForwardPass. Did you run 'setenv.sh'?");
//   }

//   // choose which function depending on plane
//   PyObject *pFunc = nullptr;
//   if (plane == 0){
//     try {
//       pFunc = PyObject_GetAttrString(pModule,"forwardpassu");
//     }
//     catch ( const std::exception& e ) {
//       std::cout << "ERROR LOADING FUNCTION 'forwardpassu' FROM MODULE" << std::endl;
//       std::cout << e.what() << std::endl;
//       throw e;
//     }
//   }
//   else if (plane == 1){
//     pFunc = PyObject_GetAttrString(pModule,"forwardpassv");
//   }
//   else{
//     pFunc = PyObject_GetAttrString(pModule,"forwardpassy");
//   }
//   if ( !pFunc ) {
//     throw std::runtime_error("ERROR loading forward pass function");
//   }
//   std::cout << "forward pass module: " << pFunc <<std::endl;  
  


//   PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc,pArray,pWeight,NULL);
//   if ( !pReturn ) {
//     throw std::runtime_error("ERROR in return array from forwardpass. Is NULL");
//   }
//   clock_t returned = clock();
//   // std::cout<<pReturn<<std::endl;
//   PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);


//   // converting back to c++ array
//   float *c_out;
//   c_out = reinterpret_cast<float*>(PyArray_DATA(np_ret));


//   // save to output_img
//   for (int i =0; i<int(npts);i++){
//     // setting threshold at 10
//     if (*(c_out + 3*i + 2) >10){
//       Pred_img.set_pixel(*(c_out + 3*i + 0),*(c_out + 3*i + 1),*(c_out + 3*i + 2));}
//   }
//   clock_t predtime = clock();
//   // std::cout << "time to make array: " << float(parray - startloop)/CLOCKS_PER_SEC <<std::endl;
//   // std::cout << "time to forward pass: " << float(returned-parray)/CLOCKS_PER_SEC <<std::endl;
//   // std::cout << "time to set final: " << float(predtime-returned)/CLOCKS_PER_SEC <<std::endl;

//   return Pred_img;
// }

void RunNetwork( PyObject* pForwardFunc,
                 PyObject* pModel,
                 larcv::SparseImage img_adcmasked,
                 larcv::Image2D& Pred_img,
                 std::string devname )
{
  // std::cout << "In network loop" <<std::endl;
  // Convert array into numpy array
  int ND = img_adcmasked.pixellist().size();
  // std::cout <<ND <<std::endl;
  clock_t startloop = clock();

  float npts    = ND/3;
  float c_arr[int(npts)][3];
  for (int start = 0; start <int(npts); start++){
    c_arr[start][0] = img_adcmasked.pixellist()[start*3];
    c_arr[start][1] = img_adcmasked.pixellist()[start*3+1];
    c_arr[start][2] = img_adcmasked.pixellist()[start*3+2];
  }

  npy_intp dims[2]{int(npts),3};
  // std::cout <<*c_arr<<std::endl;

  PyObject* pDEVNAME = PyUnicode_FromString(devname.c_str());
  PyObject *pArray = PyArray_SimpleNewFromData(2,dims,NPY_FLOAT,reinterpret_cast<void*>(c_arr));
  // std::cout<<pArray<<std::endl;
  clock_t parray = clock();
  // import forward pass python module
  
  PyObject *pReturn = PyObject_CallFunctionObjArgs(pForwardFunc,pModel,pArray,pDEVNAME,NULL);
  PyObject* perr = PyErr_Occurred();
  if ( perr ) {
    PyErr_Print();
    throw std::runtime_error("ERROR running forwardpass_all function");
  }
  
  if ( !pReturn ) {
    throw std::runtime_error("ERROR in return array from forwardpass. Is NULL");
  }
  clock_t returned = clock();
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
  clock_t predtime = clock();
  // std::cout << "time to make array: " << float(parray - startloop)/CLOCKS_PER_SEC <<std::endl;
  // std::cout << "time to forward pass: " << float(returned-parray)/CLOCKS_PER_SEC <<std::endl;
  // std::cout << "time to set final: " << float(predtime-returned)/CLOCKS_PER_SEC <<std::endl;

  Py_DECREF( pDEVNAME );
  Py_DECREF( pArray );
  Py_DECREF( pReturn );

}

int main( int argc, const char** argv ){

  Py_Initialize();
  import_array1(0);

  gStyle->SetOptStat(0);

  // Argument parser
  /** taken from https://github.com/hbristow/argparse **/
  ArgumentParser parser;
  parser.addArgument("-i","--input-larcv",1,false);
  parser.addArgument("-o","--output",1,false);
  parser.addArgument("-u","--u-weight",1,false);
  parser.addArgument("-v","--v-weight",1,false);
  parser.addArgument("-y","--y-weight",1,false);
  parser.addArgument("-n","--nentries",1,true);
  parser.addArgument("-t","--tick-backwards",1,true);
  parser.addArgument("-d","--dump-images",1,true);
  parser.addArgument("-g","--gpu-name",1,true);
  parser.parse(argc, argv);

  std::cout << "Weight files loaded:" << std::endl;
  std::cout << " u:" << parser.retrieve<std::string>("u-weight") << std::endl;
  std::cout << " v:" << parser.retrieve<std::string>("v-weight") << std::endl;
  std::cout << " y:" << parser.retrieve<std::string>("y-weight") << std::endl;


  bool dump_images = false;
  if ( parser.retrieve<std::string>("dump-images")=="1" )
    dump_images = true;
  else if ( parser.retrieve<std::string>("dump-images")!="0" && !parser.retrieve<std::string>("dump-images").empty() ) {
    std::cout << "dump-images argument must be 0 or 1" << std::endl;
    return 0;
  }
  
  bool tick_backwards = false;
  if ( parser.retrieve<std::string>("tick-backwards")=="1")
    tick_backwards = true;
  else if ( parser.retrieve<std::string>("tick-backwards")!="0" && !parser.retrieve<std::string>("tick-backwards").empty() ) {
    std::cout << "tick-backwards argument must be 0 or 1"  << std::endl;
    return 0;
  }
    
  std::string device_name  = "cpu";
  if ( !parser.retrieve<std::string>("gpu-name").empty() ) {
    device_name = parser.retrieve<std::string>("gpu-name");
  }
  
  clock_t begin = clock();
  // input IOManager
  
  larcv::IOManager *ioin = nullptr;
  if ( tick_backwards ) {
    ioin = new larcv::IOManager(larcv::IOManager::kBOTH,"IOManager",larcv::IOManager::kTickBackward);
  }
  else {
    ioin = new larcv::IOManager(larcv::IOManager::kBOTH,"IOManager");
  }
  ioin->add_in_file( parser.retrieve<std::string>("input-larcv") ); // e.g. "supera-Run004955-SubRun000079.root"
  
  if ( tick_backwards ) {
    ioin->reverse_all_products();
  }
  ioin->set_out_file( parser.retrieve<std::string>("output") ); //e.g. sparseinfill_cxx_test.root
  //ioin->set_out_file("sparseinfill_cxx_test.root"); ///< this overwritten by output iomanager?


  /**
  KEY: TTree	image2d_wire_tree;1	wire tree
  KEY: TTree	chstatus_wire_tree;1	wire tree
  KEY: TTree	image2d_ubspurn_plane0_tree;1	ubspurn_plane0 tree
  KEY: TTree	image2d_ubspurn_plane1_tree;1	ubspurn_plane1 tree
  KEY: TTree	image2d_ubspurn_plane2_tree;1	ubspurn_plane2 tree
  KEY: TTree	image2d_thrumu_tree;1	thrumu tree
  KEY: TTree	image2d_segment_tree;1	segment tree
  KEY: TTree	image2d_instance_tree;1	instance tree
  KEY: TTree	image2d_ancestor_tree;1	ancestor tree
  KEY: TTree	partroi_segment_tree;1	segment tree
  KEY: TTree	image2d_larflow_tree;1	larflow tree
  KEY: TTree	image2d_wcshower_tpc_tree;1	wcshower_tpc tree
  KEY: TTree	image2d_wctrack_tpc_tree;1	wctrack_tpc tree    
   */
  ioin->specify_data_read( larcv::kProductImage2D,  "wire");
  ioin->specify_data_read( larcv::kProductImage2D,  "ubspurn_plane0" );
  ioin->specify_data_read( larcv::kProductImage2D,  "ubspurn_plane1" );
  ioin->specify_data_read( larcv::kProductImage2D,  "ubspurn_plane2" );
  ioin->specify_data_read( larcv::kProductImage2D,  "thrumu" );
  ioin->specify_data_read( larcv::kProductImage2D,  "segment" );
  ioin->specify_data_read( larcv::kProductImage2D,  "instance" );
  ioin->specify_data_read( larcv::kProductImage2D,  "ancestor" );
  ioin->specify_data_read( larcv::kProductImage2D,  "larflow" );
  ioin->specify_data_read( larcv::kProductImage2D,  "wcshower_tpc" );
  ioin->specify_data_read( larcv::kProductImage2D,  "wctrack_tpc" );      
  ioin->specify_data_read( larcv::kProductROI,      "segment" );
  ioin->specify_data_read( larcv::kProductChStatus, "wire");
  
  ioin->initialize();

  // // output file
  // larcv::IOManager *foutIO = new larcv::IOManager( larcv::IOManager::kWRITE,"OutManager");
  // foutIO->set_out_file( parser.retrieve<std::string>("output") ); //e.g. sparseinfill_cxx_test.root
  // //foutIO->set_out_file( "sparseinfill_cxx_test.root" );
  // foutIO->initialize();

  int nentries = ioin->get_n_entries();
  int arg_nentries = (int)std::atoi(parser.retrieve<std::string>("nentries").c_str());
  if ( arg_nentries>0 && arg_nentries<nentries )
    nentries = arg_nentries;
  std::cout << "Number of entries to run: " << nentries << std::endl;

  // ------------------------------------------------------------
  // LOAD NETWORK
  // load module
  PyObject *pName   = PyUnicode_FromString("Infill_ForwardPass");
  PyObject *pModule = PyImport_Import(pName);  
  if ( pModule==0 || pModule==NULL || pModule==nullptr ) {
    throw std::runtime_error("Could not import Infill_ForwardPass. Did you run 'setenv.sh'?");
  }
  //Py_DECREF(pName);  

  // load network
  std::string u_weightfile = parser.retrieve<std::string>("u-weight");
  std::string v_weightfile = parser.retrieve<std::string>("v-weight");
  std::string y_weightfile = parser.retrieve<std::string>("y-weight");
  PyObject *pWeight_u = PyUnicode_FromString( u_weightfile.c_str() );
  PyObject *pWeight_v = PyUnicode_FromString( v_weightfile.c_str() );
  PyObject *pWeight_y = PyUnicode_FromString( y_weightfile.c_str() );
  PyObject* pDevName  = PyUnicode_FromString( device_name.c_str() );
  PyObject* pFunc_load_model  = PyObject_GetAttrString(pModule,"load_model");
  PyObject* pFunc_forwardpass = PyObject_GetAttrString(pModule,"forwardpass_all");

  PyObject *pModel_u = PyObject_CallFunctionObjArgs(pFunc_load_model,pWeight_u,pDevName,NULL);
  PyObject *pModel_v = PyObject_CallFunctionObjArgs(pFunc_load_model,pWeight_v,pDevName,NULL);
  PyObject *pModel_y = PyObject_CallFunctionObjArgs(pFunc_load_model,pWeight_y,pDevName,NULL);

  std::cout << "Models loaded. [ENTER] to continue." << std::endl;
  std::cin.get();

  // loop through entries
  for (int n = 0; n<nentries;n++){
    ioin->read_entry(n);

    auto ev_in_wholeview = (larcv::EventImage2D*)(ioin->get_data(larcv::kProductImage2D, "wire"));
    if (!ev_in_wholeview) {
       std::cout << "No Input found with a name: " << "ADCMasked" << std::endl;
     }
    const std::vector< larcv::Image2D >& wholeview_v = ev_in_wholeview->Image2DArray();

    // need chstatus to make overlays
    auto ev_chstatus = (larcv::EventChStatus*)(ioin->get_data(larcv::kProductChStatus, "wire"));
    if (!ev_chstatus) {
       std::cout << "No Input found with a name: " << "ChStatus" << std::endl;
     }

    larcv::EventBase *ev_meta  = ioin->get_data(larcv::kProductImage2D,"wire");


    int nplanes = wholeview_v.size();
    int run    = ev_meta->run();
    int subrun = ev_meta->subrun();
    int event  = ev_meta->event();

    std::cout<<"Number of planes in entry " << run << ", " << subrun << ", " << event << ": "<<nplanes<<std::endl;

    // create labels_image_v
    // std::vector<larcv::Image2D> image_label_v = {wholeview_v[0].meta() , wholeview_v[1].meta() , wholeview_v[2].meta()};
    // image_label_v = ublarcvapp::InfillDataCropper::ChStatusToLabels(image_label_v,ev_chstatus);

    ublarcvapp::EmptyChannelAlgo badchmaker;
    std::vector<larcv::Image2D> image_label_v = badchmaker.makeGapChannelImage( wholeview_v, *ev_chstatus, 4, 3, 2400, 6*1008, 3456, 6, 1, 10.0, 100, -1.0 );
    for ( auto& img : image_label_v ) {
      for (size_t r=0; r<img.meta().rows(); r++) {
        for (size_t c=0; c<img.meta().cols(); c++) {
          if ( img.pixel(r,c)>0,0 )
            img.set_pixel(r,c,1.0);
        }
      }
    }
        

    // make output trees
    larcv::EventImage2D* ev_infill  = (larcv::EventImage2D*)ioin->get_data(larcv::kProductImage2D,"infill");
    ev_infill->clear();
    // larcv::EventImage2D* ev_input  = (larcv::EventImage2D*)foutIO->get_data(larcv::kProductImage2D,"wire");
    // ev_input->clear();
    // larcv::EventChStatus* evout_chstatus = (larcv::EventChStatus*)foutIO->get_data(larcv::kProductChStatus,"wire");

    // // copy chstatus
    // for ( int p=0; p<nplanes; p++ )
    //   evout_chstatus->Insert( ev_chstatus->Status( (larcv::PlaneID_t)p ) );
    
    
    // crop using UBSplit for infill network
    // we want to break the image into set crops to send in
    // we split the entire image using UBSplitDetector

    larcv::PSet cfg = larcv::CreatePSetFromFile( "ubsplit.cfg", "UBSplitDetector" );
    ublarcvapp::UBSplitDetector algo;
    algo.initialize();
    algo.configure(cfg);


    // define the bbox_v images and cropped images for wholeview_v
    std::vector<std::vector<larcv::ROI>> bbox_list;
    std::vector<std::vector<larcv::Image2D>> img2d_list;
    std::vector<larcv::ROI> bbox_v;
    std::vector<larcv::Image2D> img2d_v;
    std::vector<larcv::Image2D> img2du;
    std::vector<larcv::Image2D> img2dv;
    std::vector<larcv::Image2D> img2dy;
    algo.process( wholeview_v,img2d_v,bbox_v );

    // define the bbox_v images and cropped images for labels image
    std::vector<std::vector<larcv::ROI>> bbox_labels_list;
    std::vector<std::vector<larcv::Image2D>> img2d_labels_list;
    std::vector<larcv::ROI> bbox_labels_v;
    std::vector<larcv::Image2D> img2d_labels_v;
    std::vector<larcv::Image2D> img2du_labels;
    std::vector<larcv::Image2D> img2dv_labels;
    std::vector<larcv::Image2D> img2dy_labels;
    algo.process( image_label_v,img2d_labels_v,bbox_labels_v );


    algo.finalize();

    // change format of output to seperate by planes
    for (int j =0;j<img2d_v.size();j++){
      int p = img2d_v[j].meta().plane();
      if (p == 0){
        img2du.push_back(img2d_v[j]);
        // std::cout<<"original meta: "<<std::endl;
        // std::cout << img2d_v[j].meta().dump() <<std::endl;
        img2du_labels.push_back(img2d_labels_v[j]);
      }
      else if (p == 1) {
        img2dv.push_back(img2d_v[j]);
        img2dv_labels.push_back(img2d_labels_v[j]);
      }
      else if (p == 2) {
        img2dy.push_back(img2d_v[j]);
        img2dy_labels.push_back(img2d_labels_v[j]);
      }
    }
    img2d_list.push_back(img2du);
    img2d_list.push_back(img2dv);
    img2d_list.push_back(img2dy);
    img2d_labels_list.push_back(img2du_labels);
    img2d_labels_list.push_back(img2dv_labels);
    img2d_labels_list.push_back(img2dy_labels);


    for (int i = 0; i <img2d_list.size();i++ ){
      std::cout <<"in list: "<< img2d_list[i].size()<<std::endl;
    }
    for (int i = 0; i <img2d_labels_list.size();i++ ){
      std::cout <<"in label list: "<< img2d_labels_list[i].size()<<std::endl;
    }
  clock_t crop = clock();
  std::cout<< "Time to finish crops: "<<float(crop-begin)/CLOCKS_PER_SEC<<std::endl;

  // Sparsify Image2D
  std::vector<float> thresholds (1,10);
  larcv::SparseImage sparse_img;
  larcv::EventSparseImage usparse_v;
  larcv::EventSparseImage vsparse_v;
  larcv::EventSparseImage ysparse_v;

  for (int i = 0;i<img2d_list.size();i++){
    for (int j =0; j<img2d_list[i].size();j++){
      sparse_img = larcv::SparseImage(img2d_list[i][j],img2d_labels_list[i][j],thresholds);
      if (i ==0) usparse_v.Append(sparse_img);
      else if (i ==1) vsparse_v.Append(sparse_img);
      else if (i ==2) ysparse_v.Append(sparse_img);
    }
  }
  clock_t sparse = clock();
  std::cout<< "Time to finish sparsify: "<<float(sparse-crop)/CLOCKS_PER_SEC<<std::endl;
  // send in to network


  // loop through planes
  std::vector<larcv::Image2D> u_out;
  std::vector<larcv::Image2D> v_out;
  std::vector<larcv::Image2D> y_out;
  u_out.reserve(66);
  v_out.reserve(66);
  y_out.reserve(66);

  for (int plane = 0;plane<3;plane++){
    if (plane == 0 ){
      for (int i = 0; i< 66;i++){
        larcv::Image2D Pred_img =  larcv::Image2D(img2d_list[0][i].meta());
        //clock_t network1 = clock();
        std::cout << "Running plane[0] img[" << i  << "]" << std::endl;
        //Pred_img = Network(Pred_img,usparse_v.at(i),plane,parser.retrieve<std::string>("u-weight"));
        RunNetwork( pFunc_forwardpass, pModel_u, usparse_v.at(i), Pred_img, device_name );
        //clock_t network2 = clock();
        //std::cout<< "Time for crop ("<<i<<"): "<<float(network2-network1)/CLOCKS_PER_SEC<<std::endl;
        u_out.emplace_back( std::move(Pred_img) );
      }
    }
    else if (plane == 1 ){
      for (int i = 0; i< 66;i++){
        larcv::Image2D Pred_img=  larcv::Image2D(img2d_list[1][i].meta());
        std::cout << "Running plane[1] img[" << i  << "]" << std::endl;        
        //clock_t network1 = clock();
        //Pred_img = Network(Pred_img,vsparse_v.at(i),plane,parser.retrieve<std::string>("v-weight"));
        RunNetwork( pFunc_forwardpass, pModel_v, vsparse_v.at(i), Pred_img, device_name );
        //clock_t network2 = clock();
        //std::cout<< "Time for crop ("<<i<<"): "<<float(network2-network1)/CLOCKS_PER_SEC<<std::endl;
        v_out.emplace_back( std::move(Pred_img) );
      }
    }
    else if (plane == 2 ){
      for (int i = 0; i< 66;i++){
        larcv::Image2D Pred_img=  larcv::Image2D(img2d_list[2][i].meta());
        std::cout << "Running plane[2] img[" << i  << "]" << std::endl;                
        //clock_t network1 = clock();
        //Pred_img = Network(Pred_img,ysparse_v.at(i),plane,parser.retrieve<std::string>("y-weight"));
        RunNetwork( pFunc_forwardpass, pModel_y, ysparse_v.at(i), Pred_img, device_name );
        //clock_t network2 = clock();
        //std::cout<< "Time for crop ("<<i<<"): "<<float(network2-network1)/CLOCKS_PER_SEC<<std::endl;
        y_out.emplace_back( std::move(Pred_img) );
      }
    }
  }
  std::cout <<"finished network loop" <<std::endl;

  // create full images from crops
  clock_t startfull = clock();
  larcv::Image2D outputimg_u= larcv::Image2D(wholeview_v.at(0).meta());
  outputimg_u.paint(0);
  larcv::Image2D outputimg_v= larcv::Image2D(wholeview_v.at(1).meta());
  outputimg_v.paint(0);
  larcv::Image2D outputimg_y= larcv::Image2D(wholeview_v.at(2).meta());
  outputimg_y.paint(0);

  // temp image to use for averaging later
  larcv::Image2D overlapcountimg_u = larcv::Image2D( wholeview_v.at(0).meta());
  overlapcountimg_u.paint(0);
  larcv::Image2D overlapcountimg_v = larcv::Image2D( wholeview_v.at(1).meta());
  overlapcountimg_v.paint(0);
  larcv::Image2D overlapcountimg_y = larcv::Image2D( wholeview_v.at(2).meta());
  overlapcountimg_y.paint(0);

  larcv::ImageMeta output_u_meta=outputimg_u.meta();
  larcv::ImageMeta output_v_meta=outputimg_v.meta();
  larcv::ImageMeta output_y_meta=outputimg_y.meta();
  // std::cout<<wholeview_v.at(0).meta().min_x()<<" "<<wholeview_v.at(0).meta().max_x()<<" "<<wholeview_v.at(0).meta().min_y()<<" "<<wholeview_v.at(0).meta().max_y()<<std::endl;

  for (int i =0; i<66;i++){
    ublarcvapp::InfillImageStitcher().Croploop(output_u_meta, u_out[i], outputimg_u,overlapcountimg_u);
    ublarcvapp::InfillImageStitcher().Croploop(output_v_meta, v_out[i], outputimg_v,overlapcountimg_v);
    ublarcvapp::InfillImageStitcher().Croploop(output_y_meta, y_out[i], outputimg_y,overlapcountimg_y);

    if ( dump_images ) {
      
      TH2D hu_out = larcv::rootutils::as_th2d( u_out[i], "hout_u" );
      TH2D hv_out = larcv::rootutils::as_th2d( v_out[i], "hout_v" );
      TH2D hy_out = larcv::rootutils::as_th2d( y_out[i], "hout_y" );

      //std::cout << "vsparse_v[" << i << "].len(): " << (int)vsparse_v.at(i).len()  << std::endl;
      std::vector< larcv::EventSparseImage* > sparse_container_v
        = { &usparse_v, &vsparse_v, &ysparse_v };
      std::vector< TH2D* > h_v = { &hu_out, &hv_out, &hy_out };
      
      for (int p=0; p<3; p++ ) {
        larcv::EventSparseImage& sparse_v = *sparse_container_v[p];
        TH2D& hout = *h_v[p];
        for ( int ipix=0; ipix<(int)sparse_v.at(i).len(); ipix++ ) {
          int row = (int)sparse_v.at(i).getfeature( ipix, 0 );
          int col = (int)sparse_v.at(i).getfeature( ipix, 1 );
          float val = sparse_v.at(i).getfeature(ipix,2);
          if ( hout.GetBinContent( col+1, row+1 )==0 )
            hout.SetBinContent( col+1, row+1, 5.0 );
        }
      }
      
      TCanvas c("c","c",1500,500);
      c.Divide(3,1);
      c.cd(1);
      hu_out.SetMaximum(120);
      hu_out.Draw("colz");
      c.cd(2);
      hv_out.SetMaximum(120);
      hv_out.Draw("colz");
      c.cd(3);
      hy_out.SetMaximum(120);
      hy_out.Draw("colz");
      char cname[256];
      sprintf(cname,"infillout_event%d_crop%d.png",(int)n,i);
      c.SaveAs(cname);
    }//end of if dump images
  }
  std::cout << "Finished Crop loop"<<std::endl;

  // creates overlay image and takes average where crops overlapped
  ublarcvapp::InfillImageStitcher().Overlayloop(0,output_u_meta,outputimg_u,overlapcountimg_u, wholeview_v, image_label_v);
  ublarcvapp::InfillImageStitcher().Overlayloop(1,output_v_meta,outputimg_v,overlapcountimg_v, wholeview_v, image_label_v);
  ublarcvapp::InfillImageStitcher().Overlayloop(2,output_y_meta,outputimg_y,overlapcountimg_y, wholeview_v, image_label_v);
  // ublarcvapp::InfillImageStitcher().Overlayloop(0,output_u_meta,outputimg_u,overlapcountimg_u, wholeview_v, *ev_chstatus);
  // ublarcvapp::InfillImageStitcher().Overlayloop(1,output_v_meta,outputimg_v,overlapcountimg_v, wholeview_v, *ev_chstatus);
  // ublarcvapp::InfillImageStitcher().Overlayloop(2,output_y_meta,outputimg_y,overlapcountimg_y, wholeview_v, *ev_chstatus);
  std::cout << "Finished Overlay loop" <<std::endl;

  ev_infill->Append(outputimg_u);
  ev_infill->Append(outputimg_v);
  ev_infill->Append(outputimg_y);

  // save bad ch image
  larcv::EventImage2D* ev_badchimg = (larcv::EventImage2D*)ioin->get_data(larcv::kProductImage2D,"badchimg");
  ev_badchimg->Emplace( std::move(image_label_v) );
  // ev_input->Append(wholeview_v.at(0));
  // ev_input->Append(wholeview_v.at(1));
  // ev_input->Append(wholeview_v.at(2));

  clock_t endfull = clock();
  std::cout<<"Time to finish stitching: "<<float(endfull-startfull)/CLOCKS_PER_SEC<<std::endl;

  // save to output file
  //foutIO->set_id( ev_meta->run(), ev_meta->subrun(), ev_meta->event() );
  //foutIO->save_entry();
  ioin->save_entry();
  std::cout<<"finished entry: "<< n <<std::endl;
  }

  //foutIO->finalize();
  ioin->finalize();

  Py_Finalize();
  clock_t final = clock();
  std::cout<< "TOTAL TIME: "<<float(final-begin)/CLOCKS_PER_SEC<<std::endl;

  return 0;
}
