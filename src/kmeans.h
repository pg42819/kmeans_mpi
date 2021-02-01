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

struct kmeans_timing {
    double main_start_time;
    double main_stop_time;
    double iteration_start;
    double iteration_start_assignment;
    double iteration_stop_assignment;
    double iteration_assignment_seconds;
    double iteration_start_centroids;
    double iteration_stop_centroids;
    double iteration_centroids_seconds;
    double accumulated_assignment_seconds;
    double accumulated_centroids_seconds;
    double max_iteration_seconds;
    double elapsed_total_seconds;
    int used_iterations;
};

extern int load_dataset(struct pointset *dataset);
extern void main_loop(int max_iterations, struct kmeans_timing *timing);
extern void main_finalize(struct pointset *dataset, struct kmeans_metrics *metrics, struct kmeans_timing *timing);

#endif