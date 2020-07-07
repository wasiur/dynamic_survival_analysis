source ~/.zprofile
source activate dynamic_survival_analysis


DATAFILE="ohio_0706.csv"
LOCATION="Ohio"
OUTPUTFOLDER="plots_dsa_bayesian_0610"
DAYZERO="2020-06-01"
FINALDATE="2020-06-30"
NITER=7500
NCHAINS=4


time python DSA_Bayesian.py -d $DATAFILE -r -o $OUTPUTFOLDER --day-zero=$DAYZERO --final-date=$FINALDATE --niter=$NITER --nchains=$NCHAINS -l $LOCATION
