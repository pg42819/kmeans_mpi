#!/usr/bin/env bash
# Run the matrix program multiple times and collect results
current_dir=$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )
source ${current_dir}/set_env.sh

processes=${1:-1}
debug_level=${KMEANS_DEBUG:-debug}
#timeout_arg="--timeout 10 --report-state-on-timeout --get-stack-traces"
timeout_arg=""
mpi_extras=${MPI_EXTRAS:-}
# mpirun --help debug for more info
echo "Running with $processes processes"
echo mpirun -n ${processes} ${mpi_extras} ${timeout_arg} ${KMEANS_BIN_DIR}/kmeans_mpi1 --${debug_level} -f ${KMEANS_DATA_DIR}/six_points.csv -k 3 -i 20 -n 10 -t ${KMEANS_TEST_DIR}/six_points_clustered_csv
mpirun -n ${processes} ${timeout_arg} ${KMEANS_BIN_DIR}/kmeans_mpi1 --${debug_level} -f ${KMEANS_DATA_DIR}/six_points.csv -k 3 -i 20 -n 10 -t ${KMEANS_TEST_DIR}/six_points_clustered_csv
