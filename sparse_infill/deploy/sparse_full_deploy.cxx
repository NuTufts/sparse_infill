#include "sparse_full_deploy.h"

larcv::Image2D Network(larcv::Image2D Pred_img,larcv::SparseImage img_adcmasked,int plane){
  // std::cout << "In network loop" <<std::endl;
  // Convert array into numpy array
  int ND = img_adcmasked.pixellist().size();
  // std::cout <<ND <<std::endl;

  float npts    = ND/3;
  float c_arr[int(npts)][3];
  for (int start = 0; start <int(npts); start++){
    c_arr[start][0] = img_adcmasked.pixellist()[start*3];
    c_arr[start][1] = img_adcmasked.pixellist()[start*3+1];
    c_arr[start][2] = img_adcmasked.pixellist()[start*3+2];
  }


  npy_intp dims[2]{int(npts),3};
  // std::cout <<*c_arr<<std::endl;

  PyObject *pArray = PyArray_SimpleNewFromData(2,dims,NPY_FLOAT,reinterpret_cast<void*>(c_arr));
  // std::cout<<pArray<<std::endl;

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

  return Pred_img;
}

int main(){
  Py_Initialize();
  import_array1(0);
  clock_t begin = clock();
  // input IOManager
  larcv::IOManager *ioin = new larcv::IOManager(larcv::IOManager::kBOTH,"IOManager");
  ioin->add_in_file( "supera-Run004955-SubRun000079.root" );
  ioin->set_out_file("sparseinfill_cxx_test.root");
  ioin->initialize();

  // // output file
  larcv::IOManager *foutIO = new larcv::IOManager( larcv::IOManager::kWRITE,"OutManager");
  foutIO->set_out_file( "sparseinfill_cxx_test.root" );
  foutIO->initialize();

  // loop through entries
  // n<ioin->get_n_entries()
  for (int n = 0; n<1;n++){
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
    std::vector<larcv::Image2D> image_label_v = {wholeview_v[0].meta() , wholeview_v[1].meta() , wholeview_v[2].meta()};
    image_label_v = ublarcvapp::InfillDataCropper::ChStatusToLabels(image_label_v,ev_chstatus);

    // make output trees
    larcv::EventImage2D* ev_infill  = (larcv::EventImage2D*)foutIO->get_data(larcv::kProductImage2D,"infill");
    ev_infill->clear();
    larcv::EventImage2D* ev_input  = (larcv::EventImage2D*)foutIO->get_data(larcv::kProductImage2D,"wire");
    ev_input->clear();
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
  std::vector<float> thresholds (1,0);
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
  for (int plane = 0;plane<3;plane++){
    if (plane == 0 ){
      for (int i = 0; i< 3;i++){
        larcv::Image2D Pred_img =  larcv::Image2D(img2d_list[0][i].meta());
        clock_t network1 = clock();
        Pred_img = Network(Pred_img,usparse_v.at(i),plane);
        clock_t network2 = clock();
        std::cout<< "Time for crop ("<<i<<"): "<<float(network2-network1)/CLOCKS_PER_SEC<<std::endl;
        u_out.push_back(Pred_img);
      }
    }
    else if (plane == 1 ){
      for (int i = 0; i< 3;i++){
        larcv::Image2D Pred_img=  larcv::Image2D(img2d_list[1][i].meta());
        clock_t network1 = clock();
        Pred_img = Network(Pred_img,vsparse_v.at(i),plane);
        clock_t network2 = clock();
        std::cout<< "Time for crop ("<<i<<"): "<<float(network2-network1)/CLOCKS_PER_SEC<<std::endl;
        v_out.push_back(Pred_img);
      }
    }
    else if (plane == 2 ){
      for (int i = 0; i< 3;i++){
        larcv::Image2D Pred_img=  larcv::Image2D(img2d_list[2][i].meta());
        clock_t network1 = clock();
        Pred_img = Network(Pred_img,ysparse_v.at(i),plane);
        clock_t network2 = clock();
        std::cout<< "Time for crop ("<<i<<"): "<<float(network2-network1)/CLOCKS_PER_SEC<<std::endl;
        y_out.push_back(Pred_img);
      }
    }
  }

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

  for (int i =0; i<3;i++){
    // std::cout<<"output meta: "<<std::endl;
    // std::cout <<  u_out[i].meta().dump() <<std::endl;
    // std::cout <<  v_out[i].meta().dump() <<std::endl;
    // std::cout <<  y_out[i].meta().dump() <<std::endl;
    ublarcvapp::InfillImageStitcher().Croploop(output_u_meta, u_out[i], outputimg_u,overlapcountimg_u);
    ublarcvapp::InfillImageStitcher().Croploop(output_v_meta, v_out[i], outputimg_v,overlapcountimg_v);
    ublarcvapp::InfillImageStitcher().Croploop(output_y_meta, y_out[i], outputimg_y,overlapcountimg_y);
  }
  // std::cout << "Finished Crop loop"<<std::endl;

  // creates overlay image and takes average where crops overlapped
  ublarcvapp::InfillImageStitcher().Overlayloop(0,output_u_meta,outputimg_u,overlapcountimg_u, wholeview_v, *ev_chstatus);
  ublarcvapp::InfillImageStitcher().Overlayloop(1,output_v_meta,outputimg_v,overlapcountimg_v, wholeview_v, *ev_chstatus);
  ublarcvapp::InfillImageStitcher().Overlayloop(2,output_y_meta,outputimg_y,overlapcountimg_y, wholeview_v, *ev_chstatus);
  // std::cout << "Finished Overlay loop" <<std::endl;

  ev_infill->Append(outputimg_u);
  ev_infill->Append(outputimg_v);
  ev_infill->Append(outputimg_y);

  ev_input->Append(wholeview_v.at(0));
  ev_input->Append(wholeview_v.at(1));
  ev_input->Append(wholeview_v.at(2));

  clock_t endfull = clock();
  std::cout<<"Time to finish stitching: "<<float(endfull-startfull)/CLOCKS_PER_SEC<<std::endl;

  // save to output file
  foutIO->set_id( ev_meta->run(), ev_meta->subrun(), ev_meta->event() );
  foutIO->save_entry();
  std::cout<<"finished entry: "<< n <<std::endl;
  }

  foutIO->finalize();
  ioin->finalize();

  Py_Finalize();
  clock_t final = clock();
  std::cout<< "TOTAL TIME: "<<float(final-begin)/CLOCKS_PER_SEC<<std::endl;

  return 0;
}
