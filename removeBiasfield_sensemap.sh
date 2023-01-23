#!/bin/bash
data=$1

echo ""
echo "------------------------------------------------------------------------------"
echo "| Performing bias field correction using SPM12"
echo "------------------------------------------------------------------------------"
matlab -nosplash -nodisplay -r "preproc_sensemap('"${data}"'); exit;"
