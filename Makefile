UNAME_S := $(shell uname -s)

#COMPILER := gcc
COMPILER := intel
#COMPILER := cuda
OMP := no
VEC := no
DEBUG := no
PAPI := no

DEBUG_FLAGS=
INCLUDES=
PAPI_INC=
PAPI_LIB=
OMP_INC=
OMP_LIB=
MPI_LIB=
MPI_INC=
OPTIMIZATION := -O1

SRC=src/
TESTDIR=test/
BIN=bin/

ifeq ($(DEBUG),yes)
		DEBUG_FLAGS=-DDEBUG -DPAPI_LOG_INFO=true -DPAPI_LOG_VERBOSE=true
endif

ifeq ($(OMP),yes)
		OMPFLAGS=-DMATRIX_OMP
endif

ifeq ($(UNAME_S),Linux)
	ifeq ($(PAPI), yes)
		# Papi libs and includes
		PAPI_INC=-I/share/apps/papi/5.5.0/include
		PAPI_LIB=-L/share/apps/papi/5.5.0/lib -lpapi
	endif

	ifeq ($(COMPILER),intel)
		#Intel requires module load intel/2020
		# BUT that CANNOT be done on search node (no qsub -I before make)
		CXX = icpc
		LD  = icpc
		MPICC=mpiicc
		MPI_LIB=
#		MPI_INC=/share/apps/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include
		OMP_FLAGS=-qopenmp $(OMP_EXTRA)
		ifeq ($(VEC),yes)
			CXXFLAGS_VECTOR  = -O3 -qopenmp -g -Wall -Wextra -std=c++11 -Wno-unused-parameter -qopt-report=5 -qopt-report-phase=vec $(DEBUG_FLAGS)
		else
			CXXFLAGS  = $(OPTIMIZATION) -qopenmp -g -Wall -Wextra -std=c99 -Wno-unused-parameter -qopt-report=2 $(DEBUG_FLAGS)
		endif
		#	include directories
		# library directories
		ifeq ($(DYNAMIC),yes)
			CXXFLAGS += -DD_DYNAMIC
		endif
		ifeq ($(IRREGULAR),yes)
			CXXFLAGS += -DD_IRREGULAR
		endif
	endif
	ifeq ($(COMPILER),cuda)
		# GPU NVidia compiler
		CXX = nvcc
		LD  = nvcc
		CXXFLAGS = $(OPTIMIZATION) -DD_GPU -arch=sm_30
		ifeq ($(DEBUG),yes)
			CXXFLAGS += -ggdb3
		endif
	endif
	ifeq ($(COMPILER),gcc)
		# GCC
		CXX=gcc
		# OPT: CXXFLAGS= $(OPTIMIZATION) -std=c99 -g -fopenmp $(INCLUDES) $(DEBUG_FLAGS)
		# NOOPT CXXFLAGS= -std=c99 -g -fopenmp $(INCLUDES) $(DEBUG_FLAGS)
		OMP_FLAGS=-fopenmp $(OMP_EXTRA)
		CXXFLAGS=$(OPTIMIZATION) -std=c99 -g $(DEBUG_FLAGS)
	endif
endif
ifeq ($(UNAME_S),Darwin)
	CXX=/usr/local/bin/gcc-10
	MPICC=/opt/openmpi/bin/mpicc
	MPI_INC=-I /opt/openmpi/include
	INCLUDES=
	OMP_FLAGS= -fopenmp
	CXXFLAGS= $(OPTIMIZATION) -std=c99 -g $(OMP_FLAGS)
	MPI_LIBS=/opt/openmpi/lib
endif
#CXXFLAGS= -O3 -std=c++11 -mavx -pg -qopenmp -qopt-report5 $(INCLUDES)

PROGS=$(BIN)kmeans

.PHONY: all
#all: $(BIN) kmeans_mpi1
all: $(BIN) mpitestpoints mpitest
#all: $(BIN) kmeans_simple
#kmeans_omp1 kmeans_omp2


kmeans_simple:
	$(CXX) $(CXXFLAGS) -o $(BIN)kmeans_simple $(SRC)kmeans.c $(SRC)csvhelper.c \
						  $(SRC)kmeans_config.c $(SRC)kmeans_support.c \
 						  $(SRC)kmeans_simple_impl.c $(HEADERS) $(LIBS)

kmeans_omp1:
	$(CXX) $(CXXFLAGS) -o $(BIN)kmeans_omp1 $(SRC)kmeans.c $(SRC)kmeans_support.c \
 						  $(SRC)kmeans_omp1_impl.c $(SRC)csvhelper.c $(HEADERS) $(LIBS)

kmeans_omp2:
	$(CXX) $(CXXFLAGS) -o $(BIN)kmeans_omp2 $(SRC)kmeans.c $(SRC)kmeans_support.c \
 						  $(SRC)kmeans_omp1_impl.c $(SRC)csvhelper.c $(HEADERS) $(LIBS)

kmeans_mpi1:
	$(MPICC) $(CXXFLAGS) -o $(BIN)kmeans_mpi1 $(SRC)kmeans.c \
						  $(SRC)kmeans_config.c $(SRC)kmeans_support.c \
 						  $(SRC)kmeans_mpi1_impl.c $(SRC)csvhelper.c \
 						  $(MPI_INC) $(MPI_LIB) $(HEADERS) $(LIBS)

mpitest:
	$(MPICC) $(CXXFLAGS) -o $(BIN)mpitest $(SRC)mpi_test.c $(MPI_INC) $(MPI_LIB) $(HEADERS) $(LIBS)

mpitestpoints:
	$(MPICC) $(CXXFLAGS) -o $(BIN)mpitestpoints $(SRC)mpi_test_points.c $(MPI_INC) $(MPI_LIB) $(HEADERS) $(LIBS)


$(BIN):
	mkdir $(BIN)

.PHONY: clean
clean:
	rm -r $(BIN)

