#!/usr/bin/env bash
# Run the matrix program multiple times and collect results
matrix_simple=${KMEANS_BIN_DIR}/matrix_1
current_dir=$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )
source ${current_dir}/set_env.sh

multirun() {
  local indata=${KMEANS_DATA_DIR}/${in}

  local test_file=${KMEANS_TEST_DIR}/${test}
  local test_args="-t ${test_file}"
  if [ ! -f "${indata}" ]; then
    echo "Cannot find input at ${indata}"; return 1;
#  else
#    echo "Raw input data  :  ${indata}"
  fi

#  echo "clustered data  :  ${outdata}"
#  echo "metrics report  :  ${metrics_file}"

  # simple run first
  local program=${matrix_simple}
  local full_label="${label} simple"

  echo "LOOPING over $num_progs programs"
  for ((prognum=${prog_first}; prognum<=${prog_first}; prognum++)) do
    echo "- LOOP prog # $prognum "
    progname=matrix_${prognum}
    program=${KMEANS_BIN_DIR}/${progname}
    local thread_label=""
    echo "- LOOPING over threads from $min_threads up to $max_threads step = $thread_step"
    for ((t=${min_threads}; t<=${max_threads}; t+=${thread_step})) do
      echo "- - LOOP thread # $t "
      # always start at 1 then jump evenly
      if [ "$t" -eq "0" ];then
        threads=1
      else
        thread_label="__t=${t}"
        threads=$t
      fi

      # subtract 1 for round numbers
      export OMP_NUM_THREADS=$threads
#      export OMP_SCHEDULE=static #,chunk_size
      echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
      local omp_label="${label}_${progname}${thread_label}"
      local order
        for c in "${matrix_sizes[@]}"; do
          if [ "$c" -eq "0" ];then
            matrix_size=1
          else
            matrix_size=$c
          fi
          echo "- - - - LOOP matrix-size # $c "
#         full_label="${omp_label}; schedule=$order;$chunk_size"
          local transpose_arg=""
          local sub_label="${omp_label}__size_${matrix_size}__order_${order}${block_label}${papi_label}"
          full_label="${sub_label}"
          echo "- - - - - WITHOUT transpose"
          singlerun
          if $dotranspose; then
            transpose_arg="--transpose"
            full_label="${sub_label}_transpose"
            echo "- - - - - WITH transpose"
          singlerun
          fi
        done
    done
  done
}

singlerun() {
  local out_arg=""
  if [ -n "${out}" ]; then
    local filename="$(echo ${full_label} | sed 's/[ \/\+\.]/_/g')"
    out_arg="-o ${out}/${filename}.csv"
  fi
  for ((run=1; run<=${repeats}; run++)) do
    echo "RUNNING $full_label repetition $run of $repeats"
    run_label="${full_label}__rep_$run"
    to_run="${program} --silent --giga ${order_arg} -s ${matrix_size} ${debug_arg} ${verbose_arg} ${block_arg} ${papi_arg} ${out_arg} -m ${metrics_file} ${transpose_arg} ${test_args} -l ${run_label}"
    echo "About to: ${to_run}"
    if $interactive; then
      askcontinue
    fi
    if [ -z ${KMEANS_RUN_NOOP+x} ]; then
      ${to_run}
    else
      echo "DRY RUN. Not doing anything"
    fi
  done
}

askcontinue() {
  local question=${1:-"Do you want to continue?"}
	read -p "$question (waiting 5s then defaulting to YES)? " -n 1 -r -t 5
	echo    # (optional) move to a new line
	# if no reply (timeout) or yY then yes, else leave it false
	if [[ -z "$REPLY" || $REPLY =~ ^[Yy]$ ]];then
		return 0
	fi
	return 1
}

run_matrix() {
  # Metrics go in same file to build a full result set
  min_threads=${KMEANS_RUN_THREADS_MIN:-0}
  max_threads=${KMEANS_RUN_THREADS_MAX:-0}
  thread_step=${KMEANS_RUN_THREADS_STEP:-20} # must be non-zero to be non infinite
#  matrix_sizes=( "$1" "$2" "$3" "$4" )

  label=matrix
  # run all: matrix_omp1 and matrix_omp2
  num_progs=1
  test_args="--test-reverse-rows"
  multirun
}


run_jutland() {
  size=$1
  max_points=$2
  max_iterations=200

  in="jutland_${size}.csv"
  # If out is empty, no output file is written but metrics and test can still be used
  # out="jutland_${size}_clustered.csv"
  test="jutland_${size}_clustered_knime.csv"
  # Metrics go in same file to build a full result set
  num_clusters=22

  min_chunks=0
  max_chunks=200
  chunks_step=20

  label=jutland_${size}
  # run all: kmeans_omp1 and kmeans_omp2
  num_progs=2
  multirun
}

if [ -z ${KMEANS_RUN_INTERACTIVE+x} ]; then
  interactive=false
else
  interactive=true
fi

if [ -z ${KMEANS_RUN_NO_TRANSPOSE+x} ]; then
  dotranspose=true
else
  echo "SKIPPING transpose"
  dotranspose=false
fi

if [ -z ${KMEANS_RUN_VERBOSE+x} ]; then
  verbose_arg=""
else
  echo "VERBOSE"
  verbose_arg="--verbose"
fi

if [ -z ${KMEANS_RUN_DEBUG+x} ]; then
  debug_arg=""
else
  echo "DEBUG"
  debug_arg="--debug"
fi

if [ -n "${KMEANS_RUN_PROG_FIRST}" ]; then
  prog_first=${KMEANS_RUN_PROG_FIRST}
else
  echo "set KMEANS_RUN_PROG_FIRST to the number of the first program 1=simple, 2=block..."
  prog_first=1
fi

if [ -n "${KMEANS_RUN_REPEATS}" ]; then
  repeats=${KMEANS_RUN_REPEATS}
  echo "Running same data $repeats times"
else
  repeats=1
fi

report_name=${KMEANS_RUN_REPORT:-1}
metrics_file=${KMEANS_METRICS_DIR}/"kmeans_metrics${report_name}.csv"

if [ -f "${metrics_file}" ]; then
  if [ -z ${KMEANS_RUN_NOOP+x} ]; then
    echo "Found an existing metrics file at ${metrics_file}"
    askcontinue "Do you want to delete the report at ${metrics_file} and start clean?" && rm -f $metrics_file
  fi
fi

if [ -n "$*" ]; then
  echo "here"
  matrix_sizes_arg="$@"
else
  if [ -n "${KMEANS_RUN_SIZES}" ]; then
    echo "here $KMEANS_RUN_SIZES"
    matrix_sizes_arg="${KMEANS_RUN_SIZES}"
  else
    matrix_sizes_arg="3"
  fi
fi


matrix_sizes=( ${matrix_sizes_arg} )
#matrix_sizes="${matrix_sizes_arg}"
run_matrix

askcontinue "Want to see the results?"
cat ${metrics_file}