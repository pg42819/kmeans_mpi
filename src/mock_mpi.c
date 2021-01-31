#include <stdbool.h>
#include <stdlib.h>
#ifdef __APPLE__
//#include "/opt/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif
#include "mpi_log.h"
#include "mock_mpi.h"

#define NUM_POINTS 6

extern int log_level;
extern int mpi_rank = 0;

//((type) ((void *) &(global)))
//#define MPI_DOUBLE OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_double)
//#define MPI_INT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_int)

//#define MPI_Datatype int
//#define MPI_Comm int
//#define MPI_INT 0
//#define MPI_COMM_WORLD 0

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
                {
    mpi_info("Mock Scatter %d items", sendcount);
    return 0;
}
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm) {
    mpi_info("Mock Gather %d items", sendcount);
    return 0;
}

int MPI_Init(int *argc, char ***argv) {
    mpi_info("Mock Init");
    return 0;
}

int MPI_Barrier(MPI_Comm comm) {
    mpi_info("Mock Barrier");
    return 0;

}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                             int root, MPI_Comm comm) {
    mpi_info("Mock Bcast %d items", count);
}

int MPI_Comm_size(MPI_Comm comm, int *size) {
    mpi_info("Mock Size");
    *size = 1; // set to 1
    return 0;
}

int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    mpi_info("Mock Rank");
    *rank = 1; // set to 0  for root
    return 0;
}

int MPI_Get_processor_name(char *name, int *resultlen) {
    mpi_info("Mock processor name");
    sprintf(name, "Mock Processor Name");
    *resultlen = 15;
}

int MPI_Finalize(void) {
    mpi_info("Mock finalize");
}
