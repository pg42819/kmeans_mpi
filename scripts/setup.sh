echo module load intel/2020
module load intel/2020
echo module load gcc/5.3.0
module load gcc/5.3.0
echo export LC_ALL=C
export LC_ALL=C
echo module load papi/5.5.0
module load papi/5.5.0
echo try: papi_avail 
echo try: qsub -I -lnodes=1,walltime=30:00

echo ready for make
