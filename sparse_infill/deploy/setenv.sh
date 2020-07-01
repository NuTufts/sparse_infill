#!/bin/bash

# WE HAVE TO SETUP SOME PYTHONPATHs FOR THE DEPLOY TO WORK

__THIS_DIR__="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "SETTING UP SPARSE INFILL DEPLOY"
echo "  location of script: "${__THIS_DIR__}
export PYTHONPATH=${__THIS_DIR__}:${__THIS_DIR__}/../models:$PYTHONPATH
