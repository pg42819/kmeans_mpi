#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#ifdef __APPLE__
#include "/opt/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif
#include "kmeans.h"
#include "kmeans_extern.h"
#include "log.h"

static char* headers[3];
static int dimensions;
struct kmeans_config *kmeans_config;
enum log_level_t log_level;

int load_dataset(struct pointset *dataset) {
    char *csv_file_name = valid_file('f', kmeans_config->in_file);
    int num_points = read_csv_file(csv_file_name, dataset, kmeans_config->max_points, headers, &dimensions);
    return num_points;
}

/**
 * Assigns each point in the dataset to a cluster based on the distance from that cluster.
 *
 * The return value indicates how many points were assigned to a _different_ cluster
 * in this assignment process: this indicates how close the algorithm is to completion.
 * When the return value is zero, no points changed cluster so the clustering is complete.
 *
 * @param dataset set of all points with current cluster assignments
 * @param centroids pointset that holds the current centroids
 * @return the number of points for which the cluster assignment was changed
 */
extern int assign_clusters(struct pointset* dataset, struct pointset *centroids);

// TODO move initialisze and assign into a sort of interface externs
extern void initialize(int max_data);
extern void initialize_representatives(int num_clusters);
extern int assign();

extern void run(int max_iterations, struct kmeans_metrics *metrics);
extern void finalize(struct kmeans_metrics *metrics);

/**
 * Calculates new centroids for the clusters of the given dataset by finding the
 * mean x and y coordinates of the current members of the cluster for each cluster.
 *
 * The centroids are set in the array passed in, which is expected to be pre-allocated
 * and contain the previous centroids: these are overwritten by the new values.
 */
extern void calculate_centroids();

void main_loop(int max_iterations, struct kmeans_metrics *metrics) {

    // we deliberately skip the centroid initialization phase in calculating the
    // total time as it is constant and never optimized
    double start_time = omp_get_wtime();
    int cluster_changes = MAX_POINTS; // start at a max then work down to zero chagnes
    int iterations = 0;

    while (cluster_changes > 0 && iterations < max_iterations) {
        DEBUG("Starting iteration %d. %d change in last iteration", iterations, cluster_changes);
        // K-Means Algo Step 2: assign every point to a cluster (closest centroid)
        double start_iteration = omp_get_wtime();
        double start_assignment = start_iteration;
        DEBUG("Main loop calling assign_clusters");
        cluster_changes = assign();
        double assignment_seconds = omp_get_wtime() - start_assignment;

        metrics->assignment_seconds += assignment_seconds;

        // K-Means Algo Step 3: calculate new centroids: one at the center of each cluster
        double start_centroids = omp_get_wtime();
        DEBUG("Main loop calling calculate_centroids");
        calculate_centroids();
        double centroids_seconds = omp_get_wtime() - start_centroids;
        metrics->centroids_seconds += centroids_seconds;

#ifndef SKIP_MAX_ITERATION_CALC
        // potentially costly calculation may skew stats, hence only in ifdef
        double iteration_seconds = omp_get_wtime() - start_iteration;
        if (iteration_seconds > metrics->max_iteration_seconds) {
            metrics->max_iteration_seconds = iteration_seconds;
        }
#endif
        iterations++;
    }
    metrics->total_seconds = omp_get_wtime() - start_time;
    metrics->used_iterations = iterations;
    INFO("Ended after %d iterations with %d changed clusters\n", iterations, cluster_changes);
}

void main_finalize(struct pointset *dataset, struct kmeans_metrics *metrics)
{
    // output file is not always written: sometimes we only run for metrics and compare with test data
    if (kmeans_config->out_file) {
        INFO("Writing output to %s\n", kmeans_config->out_file);
        write_csv_file(kmeans_config->out_file, dataset, headers, dimensions);
    }

    if (IS_DEBUG) {
        write_csv(stdout, dataset, headers, dimensions);
    }

    if (kmeans_config->test_file) {
        char *test_file_name = valid_file('t', kmeans_config->test_file);
        INFO("Comparing results against test file: %s\n", kmeans_config->test_file);
        metrics->test_result = test_results(test_file_name, dataset);
    }

    if (kmeans_config->metrics_file) {
        // metrics file may or may not already exist
        INFO("Reporting metrics to: %s\n", kmeans_config->metrics_file);
        write_metrics_file(kmeans_config->metrics_file, metrics);
    }

    if (IS_WARN) {
        print_metrics_headers(stdout);
        print_metrics(stdout, metrics);
    }
}

void debug_setup(struct pointset *dataset, struct pointset *centroids)
{
    if (IS_DEBUG) {
        DEBUG("\nDatabase Setup:\n");
        print_headers(stdout, headers, dimensions);
        print_points(stdout, dataset, "Setup ");
        DEBUG("\nCentroids Setup:\n");
        print_centroids(stdout, centroids, "Setup ");
    }
}

int main(int argc, char* argv [])
{
    kmeans_config = new_kmeans_config();
    parse_kmeans_cli(argc, argv, kmeans_config, &log_level);

    DEBUG("Initializing dataset");
    initialize(kmeans_config->max_points);

    // K-Means Lloyds alorithm  Step 1: initialize the centroids
    initialize_representatives(kmeans_config->num_clusters);

    // set up a metrics struct to hold timing and other info for comparison
    struct kmeans_metrics *metrics = new_kmeans_metrics(kmeans_config);

//     run the main loop
    run(kmeans_config->max_iterations, metrics);
//    assign(); // TODO remove and fix run

    // finalize with the metrics
    finalize(metrics);
    free(kmeans_config);
    return 0;
}



