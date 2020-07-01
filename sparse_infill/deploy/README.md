# Deploy Sparse Infill network

The program to use is `sparse_full_deploy`.

You have to build it first (in the usual way).

    make

Of course, the `ubdl` repo is assumed to have been built and setup.

To run, paths with the model functions need to be put into `PYTHONPATH`. To do so, run

    source setenv.sh


You'll need weights.  To get them, go into `../weights`.

Usage:

```
$ ./sparse_full_deploy
Usage: ./sparse_full_deploy --input-larcv INPUT-LARCV --output OUTPUT --u-weight U-WEIGHT --v-weight V-WEIGHT 
                           --y-weight Y-WEIGHT [--nentries NENTRIES] [--tick-backwards]
```

`--tick-backwards` should be used if image2d made using LArCV1 and the image is stored in tick-backwards orientation. (Usually the case if part of DLRECO chain.)

Example call:

```
./sparse_full_deploy --input-larcv merged_dlreco_38b62814-ff15-4b1c-8433-334291ed6a13.root -o test.root --u-weight ../weights/prelim_june2019/sparseinfill_uplane_test.tar --v-weight ../weights/prelim_june2019/sparseinfill_vplane_test.tar --y-weight ../weights/prelim_june2019/sparseinfill_yplane_test.tar --nentries 1 -t
```

