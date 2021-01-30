#!/usr/bin/env bash
# Run the matrix program once
# E.g. simple_run.sh -s 1024 --transpose --ikj
if [ -z "$KMEANS_HOME" ]; then
  current_dir=$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )
  export KMEANS_HOME=$( dirname ${current_dir} )
fi

export KMEANS_DATA_DIR=${KMEANS_HOME}/data
export KMEANS_OUT_DIR=${KMEANS_HOME}/outdata
export KMEANS_TEST_DIR=${KMEANS_HOME}/testdata
export KMEANS_METRICS_DIR=${KMEANS_HOME}/reports
export KMEANS_BIN_DIR=${KMEANS_HOME}/bin
export KMEANS_SCRIPTS_DIR=${KMEANS_HOME}/scripts
export KMEANS_ADVISOR_DIR=${KMEANS_HOME}/advisor

