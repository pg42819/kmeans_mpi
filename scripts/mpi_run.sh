#!/usr/bin/env bash
# Run the matrix program multiple times and collect results
current_dir=$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )
source ${current_dir}/set_env.sh

processes=${1:-1}
echo "Running with $processes processes"
mpirun -n ${processes} ${KMEANS_BIN_DIR}/kmeans_mpi1 -v -d -f ${KMEANS_DATA_DIR}/six_points.csv -k 3 -i 20 -n 10 -t ${KMEANS_TEST_DIR}/six_points_clustered_csv
