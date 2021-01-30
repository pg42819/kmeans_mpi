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
struct log_config *log_config;

/**
 * Initializes the given array of points to act as initial centroid "representatives" of the
 * clusters, by selecting the first num_clusters points in the dataset.
 *
 * Note that there are many ways to do this for k-means, most of which are better than the
 * approach used here - the most common of which is to use a random sampling of points from
 * the dataset. Since we are performance tuning, however, we want a consistent performance
 * across difference runs of the algorithm, so we simply select the first K points in the dataset
 * where K is the number of clusters.
 *
 * WARNING: The kmeans can fail if there are equal points in the first K in the dataset
 *          such that two or more of the centroids are the same... try to avoid this in
 *          your dataset (TODO fix this later to skip equal centroids)
 *
 * @param dataset array of all points
 * @param centroids uninitialized array of centroids to be filled
 * @param num_clusters the number of clusters (K) for which centroids are created
 */
void initialize_centroids(struct pointset* dataset, struct pointset *centroids)
{
    if (dataset->num_points < centroids->num_points) {
        FAIL("There cannot be fewer points than clusters");
    }

    copy_points(dataset, centroids, 0, centroids->num_points, false);
}

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
//        cluster_changes = assign_clusters(dataset, centroids);
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
    if (!log_config->quiet) {
        printf("\nEnded after %d iterations with %d changed clusters\n", iterations, cluster_changes);
    }
}

void main_finalize(struct pointset *dataset, struct kmeans_metrics *metrics) {

#ifdef KMEANS_MPI
    MPI_Finalize();
#endif
    // output file is not always written: sometimes we only run for metrics and compare with test data
    if (kmeans_config->out_file) {
        if (!log_config->silent) {
            printf("Writing output to %s\n", kmeans_config->out_file);
        }
        write_csv_file(kmeans_config->out_file, dataset, headers, dimensions);
    }

    if (log_config->debug) {
        write_csv(stdout, dataset, headers, dimensions);
    }

    if (kmeans_config->test_file) {
        char *test_file_name = valid_file('t', kmeans_config->test_file);
        if (!log_config->quiet) {
            printf("Comparing results against test file: %s\n", kmeans_config->test_file);
        }
        metrics->test_result = test_results(test_file_name, dataset);
    }

    if (kmeans_config->metrics_file) {
        // metrics file may or may not already exist
        if (!log_config->quiet) {
            printf("Reporting metrics to: %s\n", kmeans_config->metrics_file);
        }
        write_metrics_file(kmeans_config->metrics_file, metrics);
    }

    if (!log_config->silent) {
        print_metrics_headers(stdout);
        print_metrics(stdout, metrics);
    }
}

void debug_setup(struct pointset *dataset, struct pointset *centroids)
{
    if (log_config->debug) {
        printf("\nDatabase Setup:\n");
        print_headers(stdout, headers, dimensions);
        print_points(stdout, dataset, "Setup ");
        printf("\nCentroids Setup:\n");
        print_centroids(stdout, centroids, "Setup ");
    }
}

int main(int argc, char* argv [])
{
    log_config = new_log_config();
    kmeans_config = new_kmeans_config();
    parse_kmeans_cli(argc, argv, kmeans_config, log_config);

    DEBUG("Initializing dataset");
    initialize(kmeans_config->max_points);

    // K-Means Lloyds alorithm  Step 1: initialize the centroids
    initialize_representatives(kmeans_config->num_clusters);

    // set up a metrics struct to hold timing and other info for comparison
    struct kmeans_metrics *metrics = new_kmeans_metrics(kmeans_config);

    // run the main loop
    run(kmeans_config->max_iterations, metrics);

    // finalize with the metrics
    finalize(metrics);
    free(log_config);
    free(kmeans_config);
    return 0;
}

