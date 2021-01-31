#include <stdbool.h>
#include <stdlib.h>
#include "mpi_test.h"
#include "../src/kmeans.h"
#ifdef __APPLE__
#include "/opt/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif
#include "../src/mpi_log.h"

int log_level = MPI_LOG_DEBUG;
int mpi_rank = 0;

int main() {
    mpi_log(MPI_LOG_DEBUG, "simple debug");
    mpi_log(MPI_LOG_DEBUG, "debug something %d", 44);
    mpi_rank = 3;
    mpi_log(MPI_LOG_DEBUG, "debug string something %s", "foo");
    mpi_rank = 1;
    mpi_log(MPI_LOG_INFO, "info string something %s", "foo");
//    mpi_warn("warn something %d", 44);
}
