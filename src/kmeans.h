#ifndef KMEANS_H
#define KMEANS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <omp.h>

#define NUM_CLUSTERS 15
#define MAX_ITERATIONS 10000
#define MAX_POINTS 5000

struct point {
    double x, y;
    int cluster;
};

// rather than having a struct for single point, we use a
// struct for the whole dataset - making it easier to break int
// simple arrays of ints and doubles for marshalling/unmarshalling (over MPI)
struct pointset {
    int num_points;
    double *x_coords;
    double *y_coords;
    int *cluster_ids;
};

struct kmeans_config {
    int max_points;
    int num_clusters;
    int max_iterations;
    int num_processors;
    char *in_file;
    char *out_file;
    char *test_file;
    char *metrics_file;
    char *label;
    bool proper_distance; // true means perform square root in euclidean
};

struct kmeans_metrics {
    char *label; // label for metrics row from -l command line arg
    double assignment_seconds;    // total time spent assigning points to clusters in every iteration
    double centroids_seconds;     // total time spent calculating new centroids in every iteration
    double total_seconds;         // total time in seconds for the run
    double max_iteration_seconds; // time taken by the slowest iteration of the whole algo
    int used_iterations; // number of actual iterations needed to complete clustering
    int test_result;     // 0 = not tested, 1 = passed, -1 = failed comparison with expected data
    int num_points;      // number or points in the file limited to max from -n command line arg
    int num_clusters;    // number of clusters from  -k command line arg
    int max_iterations;  // max iterations from -i command line arg
    int num_processors; // number of processors that mpi is running on
};

#endif