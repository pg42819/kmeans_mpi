#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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
void initialize_centroids(struct point* dataset, struct point *centroids, int num_clusters)
{
    for (int k = 0; k < num_clusters; ++k) {
        centroids[k] = dataset[k];
    }
}

/**
 * Assigns each point in the dataset to a cluster based on the distance from that cluster.
 *
 * The return value indicates how many points were assigned to a _different_ cluster
 * in this assignment process: this indicates how close the algorithm is to completion.
 * When the return value is zero, no points changed cluster so the clustering is complete.
 *
 * @param dataset set of all points with current cluster assignments
 * @param num_points number of points in the dataset
 * @param centroids array that holds the current centroids
 * @param num_clusters number of clusters - hence size of the centroids array
 * @return the number of points for which the cluster assignment was changed
 */
extern int assign_clusters(struct point* dataset, int num_points, struct point *centroids, int num_clusters);

/**
 * Calculates new centroids for the clusters of the given dataset by finding the
 * mean x and y coordinates of the current members of the cluster for each cluster.
 *
 * The centroids are set in the array passed in, which is expected to be pre-allocated
 * and contain the previous centroids: these are overwritten by the new values.
 *
 * @param dataset set of all points with current cluster assignments
 * @param num_points number of points in the dataset
 * @param centroids array to hold the centroids - already allocated
 * @param num_clusters number of clusters - hence size of the centroids array
 */
extern void calculate_centroids(struct point* dataset, int num_points, struct point *centroids, int num_clusters);

int main(int argc, char* argv [])
{
    log_config = new_log_config();
    kmeans_config = new_kmeans_config();
    parse_kmeans_cli(argc, argv, kmeans_config, log_config);

    struct point *dataset = malloc(kmeans_config->max_points * sizeof(struct point));
    char* csv_file_name = valid_file('f', kmeans_config->in_file);
    int num_points = read_csv_file(csv_file_name, dataset, kmeans_config->max_points, headers, &dimensions);

    // K-Means Algo Step 1: initialize the centroids
    struct point *centroids = malloc(kmeans_config->num_clusters * sizeof(struct point));
    initialize_centroids(dataset, centroids, kmeans_config->num_clusters);

    // we deliberately skip the centroid initialization phase in calculating the
    // total time as it is constant and never optimized
    double start_time = omp_get_wtime();
    if (log_config->debug) {
        printf("\nDatabase:\n");
        print_headers(stdout, headers, dimensions);
        print_points(stdout, dataset, num_points);
        printf("\nCentroids:\n");
        print_centroids(stdout, centroids, kmeans_config->num_clusters);
    }
    int cluster_changes = num_points;
    int iterations = 0;

    // set up a metrics struct to hold timing and other info for comparison
    struct kmeans_metrics *metrics = new_kmeans_metrics(kmeans_config);
    metrics->num_points = num_points;
    metrics->omp_max_threads = omp_get_max_threads();
    // get kind: dynamic, static, auto.. and the chunk size
    metrics->omp_schedule_kind = omp_schedule_kind(&metrics->omp_chunk_size);

    while (cluster_changes > 0 && iterations < kmeans_config->max_iterations) {
        // K-Means Algo Step 2: assign every point to a cluster (closest centroid)
        double start_iteration = omp_get_wtime();
        double start_assignment = start_iteration;
        cluster_changes = assign_clusters(dataset, num_points, centroids, kmeans_config->num_clusters);
        double assignment_seconds = omp_get_wtime() - start_assignment;

        metrics->assignment_seconds += assignment_seconds;

        if (log_config->debug) {
            printf("\n%d clusters changed after assignment phase. New assignments:\n", cluster_changes);
            print_points(stdout, dataset, num_points);
            printf("Time taken: %.3f seconds total in assignment so far: %.3f seconds",
                   assignment_seconds, metrics->assignment_seconds);
        }
        // K-Means Algo Step 3: calculate new centroids: one at the center of each cluster
        double start_centroids = omp_get_wtime();
        calculate_centroids(dataset, num_points, centroids, kmeans_config->num_clusters);
        double centroids_seconds = omp_get_wtime() - start_centroids;
        metrics->centroids_seconds += centroids_seconds;

        if (log_config->verbose) {
            printf("New centroids calculated New assignments:\n");
            print_centroids(stdout, centroids, kmeans_config->num_clusters);
            printf("Time taken: %.3f seconds total in centroid calculation so far: %.3f seconds",
                   centroids_seconds, metrics->centroids_seconds);
        }
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

    // output file is not always written: sometimes we only run for metrics and compare with test data
    if (kmeans_config->out_file) {
        if (!log_config->silent) {
            printf("Writing output to %s\n", kmeans_config->out_file);
        }
        write_csv_file(kmeans_config->out_file, dataset, num_points, headers, dimensions);
    }

    if (log_config->debug) {
        write_csv(stdout, dataset, num_points, headers, dimensions);
    }

    if (kmeans_config->test_file) {
        char* test_file_name = valid_file('t', kmeans_config->test_file);
        if (!log_config->quiet) {
            printf("Comparing results against test file: %s\n", kmeans_config->test_file);
        }
        metrics->test_result = test_results(test_file_name, dataset, num_points);
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

    free(log_config);
    free(kmeans_config);
    free(dataset);
    return 0;
}

