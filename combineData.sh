#!/bin/bash
Full=$1
Slab=$2
WM=$3
Mask=$4
Out=$5

echo ""
echo "------------------------------------------------------------------------------"
echo "| Combine slab and full brain data"
echo "------------------------------------------------------------------------------"

INPUT="'${Full}', '${Slab}', '${WM}', '${Mask}', '${Out}'"
matlab -nosplash -nodisplay -r "weightedAverage(${INPUT}); exit;"
