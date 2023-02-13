#!/bin/sh
#=======================================================================
# ensfcst.sh
#   This script runs the SPEEDY model with subdirectory $PROC
#=======================================================================

# Input for this shell
SPEEDY=$1
OUTPUT=$2/DATA
YMDH=$3
TYMDH=$4

TMPDIR=${OUTPUT}/tmp
cd ${TMPDIR} 
# Create directory for this process
cp /home/ucakdpg/Scratch/mogp-speedy/imp .
# Set up boundary files
SB=$SPEEDY/model/data/bc/t30/clim
SC=$SPEEDY/model/data/bc/t30/anom
ln -s $SB/sfc.grd   fort.20
ln -s $SB/sst.grd   fort.21
ln -s $SB/icec.grd  fort.22
ln -s $SB/stl.grd   fort.23
ln -s $SB/snowd.grd fort.24
ln -s $SB/swet.grd  fort.26
cp    $SC/ssta.grd  fort.30

# Run
ln -fs $OUTPUT/${YMDH}.grd fort.90
ln -fs $OUTPUT/fluxes.grd fluxes.grd
FORT2=2
echo $FORT2 > fort.2
echo $YMDH | cut -c1-4 >> fort.2
echo $YMDH | cut -c5-6 >> fort.2
echo $YMDH | cut -c7-8 >> fort.2
echo $YMDH | cut -c9-10 >> fort.2
./imp > out.lis 2> out.lis.2
mv ${TYMDH}.grd $OUTPUT
rm ${TYMDH}_p.grd
mv ${TYMDH}_fluxes.grd $OUTPUT
#rm ${TMPDIR}/*
exit 0
