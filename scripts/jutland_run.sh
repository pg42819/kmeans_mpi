#!/usr/bin/env bash
# Run the kmeans program multiple times and collect results
if [ -z "$KMEANS_HOME" ]; then
  current_dir=$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )
  KMEANS_HOME=$( dirname ${current_dir} )
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
  local full_label="${label}_simple"
  local runner=""
  singlerun

  for ((prognum=min_prog; prognum<=max_prog; prognum++)) do
    progname="${prog_base}${prognum}"
    program=${bin_dir}/${progname}
    for ((t=min_threads; t<=max_threads; t=t+t)) do
#    for ((t=min_threads; t<=max_threads; t+=thread_step)) do
      # always start at 1 then jump evenly
      if [ "$t" -eq "0" ];then
        threads=1
      else
        threads=$t
      fi
      echo "Threads loop with ${threads} threads"
      local sublabel="${label}__${progname}__n_$threads"
      local map_by="node"
      local full_label="${sublabel}_c_s,0,0"
      runner="mpirun -np ${threads} ${mpi_base_args} --map-by ${map_by}"
      singlerun
      for c in 4 8; do
          echo "Cores loop with ${c} cores"
          local map_by="ppr:${c}:core"
          full_label="${sublabel}_c_s,$c,0"
          runner="mpirun -np ${threads} ${mpi_base_args} --map-by ${map_by}"
          singlerun
      done
      for s in 1 2 4 8; do
          echo "Sockets loop with ${s} sockets"
          local map_by="ppr:${s}:socket"
          full_label="${sublabel}_c_s,0,$s"
          runner="mpirun -np ${threads} ${mpi_base_args} --map-by ${map_by}"
          singlerun
      done
    done
  done
}

singlerun() {
    echo "RUNNING $full_label"
    command="${runner} ${program} --${debug_level} -f ${indata} ${out_arg} \
           -m ${metrics_file} ${test_args} -k ${num_clusters} -n ${max_points} \
            -i ${max_iterations} -l ${full_label}"
    echo "About to: ${command}"
    if $interactive; then
      askcontinue
    fi
    if [ -z ${KMEANS_RUN_NOOP+x} ]; then
      ${command}
    echo "Finished: ${command}"
    else
      echo "DRY RUN. Not doing anything"
    fi

  # run the program with -s for silent
#  "${program}" -s -f "${indata}" ${out_arg} -m "${metrics_file}" ${test_args} \
#      -k ${num_clusters} -n ${max_points} -i ${max_iterations} -l "${full_label}"
}

askcontinue() {
  local question=${1:-"Do you want to continue?"}
	read -p "$question (waiting a few seconds then defaulting to YES)? " -n 1 -r -t 10
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

  min_threads=${KMEANS_PROCESSES_MIN:-1}
  max_threads=${KMEANS_PROCESSES_MAX:-200}
  thread_step=${KMEANS_PROCESSES_STEP:-20}

  label=jutland_${size}
  min_prog=${KMEANS_PROG_NUM_MIN:-1}
  # default max == min prog
  max_prog=${KMEANS_PROG_NUM_MAX:-1}
  multirun
}

debug_level=${KMEANS_DEBUG:-debug}
num_clusters=${KMEANS_CLUSTERS:-3}
max_iterations=${KMEANS_MAX_ITERATIONS:-20}
prog_base=${KMEANS_PROG_BASE:-kmeans_mpi}
report_name=${KMEANS_REPORT:-jutland_metrics.csv}
mpi_base_args="-mca btl self,sm,tcp"

if [ -z ${KMEANS_RUN_INTERACTIVE+x} ]; then
  interactive=false
else
  interactive=true
fi

if [ -z ${KMEANS_OUT_BASE+x} ]; then
  out=""
else
  out=${KMEANS_OUT_BASE}
fi

metrics_file=${metrics_dir}/${report_name}
if [ -f "${metrics_file}" ]; then
  echo "Found an existing metrics file at ${metrics_file}"
  askcontinue "Do you want to delete the report at ${metrics_file} and start clean?" && rm -f $metrics_file
fi


# For sockets use 1,2,4,8
# for cores use 8, 16

if [ -z ${KMEANS_JUTLAND_SIZE+x} ]; then
  run_jutland 50 100
  run_jutland 500 1000
  run_jutland 400k 1000000
else
  run_jutland ${KMEANS_JUTLAND_SIZE} ${KMEANS_MAX_POINTS}
fi

askcontinue "Want to see the results in ${metrics_file}?"
cat ${metrics_file}
