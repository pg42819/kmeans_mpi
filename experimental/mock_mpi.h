#include <stdbool.h>
#include <stdlib.h>
#ifdef __APPLE__
#include "/opt/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif
#include "../src/mpi_log.h"

#undef MPI_Datatype
#define MPI_Datatype int
#undef MPI_Comm
#define MPI_Comm int
