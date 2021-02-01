#!/usr/bin/env bash
# Run the matrix program multiple times and collect results
current_dir=$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )
source ${current_dir}/set_env.sh

env_processes=${KMEANS_PROCESSES:-1}
processes=${1:-${KMEANS_PROCESSES}}
debug_level=${KMEANS_DEBUG:-debug}
infile=${KMEANS_IN:-six_points.csv}
testfile=${KMEANS_TEST:-six_points_clustered_csv}
clusters=${KMEANS_CLUSTERS:-3}
max_iterations=${KMEANS_MAX_ITERATIONS:-20}
max_points=${KMEANS_MAX_POINTS:-10}
prog_num=${KMEANS_PROG_NUM:-1}
prog_base=${KMEANS_PROG_BASE:-kmeans_mpi}
prog="${prog_base}${prog_num}"
#timeout_arg="--timeout 10 --report-state-on-timeout --get-stack-traces"
timeout_arg=""
mpi_extras=${MPI_EXTRAS:-}
# mpirun --help debug for more info
echo "Running ${prog} with $processes processes"
echo "with -k ${clusters} and input from ${infile} with debug level ${debug_level}"
command="mpirun -n ${processes} ${mpi_extras} ${KMEANS_BIN_DIR}/${prog} --${debug_level} \
 -f ${KMEANS_DATA_DIR}/${infile} -k ${clusters} -i ${max_iterations} -n ${max_points} \
 -t ${KMEANS_TEST_DIR}/${testfile}"
echo "running: $command"
${command}
