#!/usr/bin/env bash
# Run the kmeans program multiple times and collect results
if [ -z "$KMEANS_HOME" ]; then
  current_dir=$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )
  export KMEANS_HOME=$( dirname ${current_dir} )
fi
data_dir=${KMEANS_HOME}/data
out_dir=${KMEANS_HOME}/outdata
test_dir=${KMEANS_HOME}/testdata
metrics_dir=${KMEANS_HOME}/reports
bin_dir=${KMEANS_HOME}/bin
kmeans_simple=${bin_dir}/kmeans_simple

multirun() {
  local indata=${data_dir}/${in}
  local out_arg=""
  if [ -n "${out}" ]; then
    out_arg="-o ${out_dir}/${out}"
  fi

  local test_file=${test_dir}/${test}
  local test_args="-t ${test_file}"
  if [ ! -f "${indata}" ]; then
    echo "Cannot find input at ${indata}"; return 1;
#  else
#    echo "Raw input data  :  ${indata}"
  fi


#  echo "clustered data  :  ${outdata}"
#  echo "metrics report  :  ${metrics_file}"
#  echo "num clusters   : ${num_clusters}"
#  echo "max points     : ${max_points}"
#  echo "max iterations : ${max_iterations}"

  # simple run first
  local program=${kmeans_simple}
  local full_label="${label} simple"
  singlerun

  for ((prognum=1; prognum<=num_progs; prognum++)) do
    progname=kmeans_omp${prognum}
    program=${bin_dir}/${progname}
    for ((t=0; t<=max_threads; t+=thread_step)) do
      # always start at 1 then jump evenly
      if [ "$t" -eq "0" ];then
        threads=1
      else
        threads=$t
      fi

      #echo "THREAD_COUNT $threads  from 1 to $max_threads stepping $thread_step"
      # subtract 1 for round numbers
      export OMP_NUM_THREADS=$threads
      #echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
      local omp_label="${label} ${progname} t=$threads"
      local kind
      for kind in dynamic static guided; do
        for ((c=0; c<=max_chunks; c+=chunks_step)) do
           # always start at 1 then jump evenly
          if [ "$c" -eq "0" ];then
            chunk_size=1
          else
            chunk_size=$c
          fi
          export OMP_SCHEDULE=$kind,$chunk_size
          full_label="${omp_label}; schedule=$kind;$chunk_size"
          #echo "RUNNING $full_label"
          singlerun
        done
      done
    done
  done
}

singlerun() {
  # run the program with -s for silent
  "${program}" -s -f "${indata}" ${out_arg} -m "${metrics_file}" ${test_args} \
      -k ${num_clusters} -n ${max_points} -i ${max_iterations} -l "${full_label}"
  #echo "${program}" -f "${indata}" -o "${outdata}" -m "${metrics_file}" ${test_args} \
   #   -k ${num_clusters} -n ${max_points} -i ${max_iterations} -l "${full_label}"
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

  min_threads=0
  max_threads=200
  thread_step=20

  min_chunks=0
  max_chunks=200
  chunks_step=20

  label=jutland_${size}
  # run all: kmeans_omp1 and kmeans_omp2
  num_progs=2
  multirun
}

metrics_file=${metrics_dir}/"jutland_metrics.csv"
if [ -f "${metrics_file}" ]; then
  echo "Found an existing metrics file at ${metrics_file}"
  askcontinue "Do you want to delete the report at ${metrics_file} and start clean?" && rm -f $metrics_file
fi

run_jutland 50 100
run_jutland 500 1000
run_jutland 400k 1000000

