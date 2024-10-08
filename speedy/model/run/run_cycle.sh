#!/bin/bash
#=======================================================================
# run_cycle
#   To run the SPEEDY model for forecast-forecast cycle, useful to
#   generate a nature run.
#=======================================================================

# Directory settings
cd ../..
SPEEDY=`pwd`
NATURE=$SPEEDY/DATA/nature
TMPDIR=$SPEEDY/model/tmp

# Source experiment configuration and time increment function
source $SPEEDY/config.sh
source $SPEEDY/common/timeinc.sh

# Work directory
rm -rf $TMPDIR
mkdir -p $TMPDIR
cd $TMPDIR

echo 'Building model'
cp $SPEEDY/model/source/makefile .
cp $SPEEDY/model/source/*.h .
cp $SPEEDY/model/source/*.f90 .
cp $SPEEDY/model/source/*.s .

# Set resolution
echo "Patching configuration parameters"
if [[ "$nat_res" = "t39" ]]
then
    sed -i '' -e "s/NTRUN/39/g" mod_atparam.f90
    sed -i '' -e "s/NLON/120/g" mod_atparam.f90
    sed -i '' -e "s/NLAT/60/g" mod_atparam.f90
    sed -i '' -e "s/NSTEPS/72/g" mod_tsteps.f90
elif [[ "$nat_res" = "t30" ]]
then
    sed -i '' -e "s/NTRUN/30/g" mod_atparam.f90
    sed -i '' -e "s/NLON/96/g" mod_atparam.f90
    sed -i '' -e "s/NLAT/48/g" mod_atparam.f90
    sed -i '' -e "s/NSTEPS/36/g" mod_tsteps.f90
fi
sed -i '' -e "s/NMONTS/3/g" mod_tsteps.f90
sed -i '' -e "s/NMONRS/3/g" mod_tsteps.f90
sed -i '' -e "s/IHOUT/.true./g" mod_tsteps.f90
sed -i '' -e "s/IPOUT/.true./g" mod_tsteps.f90
sed -i '' -e "s/SIXHRRUN/.true./g" mod_tsteps.f90

make -s imp

sh inpfiles.s $nat_res

# Cycle run 
YYYY=$IYYYY
MM=$IMM
DD=$IDD
HH=$IHH

# Extension for fluxes files
fluxes="_fluxes"

# Copy the executable
cp imp $NATURE

while test $YYYY$MM$DD$HH -le $FYYYY$FMM$FDD$FHH
do
    # Run 6-hour forecast
    FORT2=2
    ln -fs $NATURE/$YYYY$MM$DD$HH.grd fort.90
    echo "$YYYY/$MM/$DD/$HH"
    echo $FORT2 > fort.2
    echo $YYYY >> fort.2
    echo $MM >> fort.2
    echo $DD >> fort.2
    echo $HH >> fort.2
    ./imp &> out.lis

    # Date change
    TY=`timeinc6hr $YYYY $MM $DD $HH | cut -c1-4`
    TM=`timeinc6hr $YYYY $MM $DD $HH | cut -c5-6`
    TD=`timeinc6hr $YYYY $MM $DD $HH | cut -c7-8`
    TH=`timeinc6hr $YYYY $MM $DD $HH | cut -c9-10`
    YYYY=$TY
    MM=$TM
    DD=$TD
    HH=$TH

    # Move output file
    mv $YYYY$MM$DD$HH.grd $NATURE
    mv $YYYY$MM$DD$HH$fluxes.grd $NATURE
done

echo "Cleaning up"
cd ..
# rm -rf $TMPDIR
