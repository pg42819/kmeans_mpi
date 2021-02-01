#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "kmeans.h"
#include "kmeans_support.h"
#include "log.h"
#include <float.h>

/**
 * Simple sequential re-calculation of centroids for Lloyds K-Means algorithm
 * @param dataset dataset of points
 * @param centroids centroids to be updated
 */
void simple_calculate_centroids(struct pointset *dataset, struct pointset *centroids)
{
    int num_points = dataset->num_points;
    int num_clusters = centroids->num_points;
    double sum_of_x_per_cluster[num_clusters];
    double sum_of_y_per_cluster[num_clusters];
    int num_points_in_cluster[num_clusters];
    for (int k = 0; k < num_clusters; ++k) {
        sum_of_x_per_cluster[k] = 0.0;
        sum_of_y_per_cluster[k] = 0.0;
        num_points_in_cluster[k] = 0;
    }

    // loop over all points in the database and sum up
    // the x coords of clusters to which each belongs
    for (int n = 0; n < num_points; ++n) {
        int k = dataset->cluster_ids[n];
        sum_of_x_per_cluster[k] += dataset->x_coords[n];
        sum_of_y_per_cluster[k] += dataset->y_coords[n];
        // count the points in the cluster to get a mean later
        num_points_in_cluster[k]++;
    }

    // the new centroids are at the mean x and y coords of the clusters
    for (int k = 0; k < num_clusters; ++k) {
        int cluster_size = num_points_in_cluster[k];
        TRACE("Cluster %d has %d points", k, cluster_size);
        // ignore empty clusters (otherwise div by zero!)
        if (cluster_size > 0) {
            // mean x, mean y => new centroid
            double new_centroid_x = sum_of_x_per_cluster[k] / cluster_size;
            double new_centroid_y = sum_of_y_per_cluster[k] / cluster_size;
            set_point(centroids, k, new_centroid_x, new_centroid_y, IGNORE_CLUSTER_ID);
        }
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
 * @param centroids set of current centroids
 * @return the number of points for which the cluster assignment was changed
 */
int simple_assign_clusters(struct pointset *dataset, struct pointset *centroids)
{
    TRACE("Starting simple assignment");
    int cluster_changes = 0;

    int num_points = dataset->num_points;
    int num_clusters = centroids->num_points;
    for (int n = 0; n < num_points; ++n) {
        double min_distance = DBL_MAX; // init the min distance to a big number
        int closest_cluster = -1;
        for (int k = 0; k < num_clusters; ++k) {
            // calc the distance passing pointers to points since the distance does not modify them
            double distance_from_centroid = point_distance(dataset, n, centroids, k);
            if (distance_from_centroid < min_distance) {
                min_distance = distance_from_centroid;
                closest_cluster = k;
                TRACE("Closest cluster to point %d is %d", n, k);
            }
        }
        // if the point was not already in the closest cluster, move it there and count changes
        if (dataset->cluster_ids[n] != closest_cluster) {
            dataset->cluster_ids[n] = closest_cluster;
            cluster_changes++;
            TRACE("Assigning (%s) to cluster %d with centroid (%s) d = %f\n",
                  p_to_s(dataset, n),  closest_cluster, p_to_s(centroids, closest_cluster), min_distance);
        }
    }
    TRACE("Leaving simple assignment with %d cluster changes", cluster_changes);
    return cluster_changes;
}



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

void simple_start_iteration_timing(struct kmeans_timing *timing)
{
    double now = omp_get_wtime();
    timing->iteration_start = now;
    timing->iteration_start_assignment = now;
}

void simple_between_assignment_centroids(struct kmeans_timing *timing)
{
    double now = omp_get_wtime();
    double assignment_seconds = now - timing->iteration_start_assignment;
    timing->iteration_assignment_seconds = assignment_seconds;
    timing->accumulated_assignment_seconds += assignment_seconds;
    timing->iteration_start_centroids = now;
}

void simple_end_iteration_timing(struct kmeans_timing *timing)
{
        double now = omp_get_wtime();
        double centroids_seconds = now - timing->iteration_start_centroids;
        timing->iteration_centroids_seconds = centroids_seconds;
        timing->accumulated_centroids_seconds += centroids_seconds;

        // potentially costly calculation, but testing shows does not skew stats significantly  and worth it for metrics
        double iteration_seconds = now - timing->iteration_start;
        if (iteration_seconds > timing->max_iteration_seconds) {
            timing->max_iteration_seconds = iteration_seconds;
        }
}

void simple_start_main_timing(struct kmeans_timing *timing)
{
        double now = omp_get_wtime();
        timing->main_start_time = now;
}

void simple_end_main_timing(struct kmeans_timing *timing, int iterations)
{
    double now = omp_get_wtime();
    timing->main_stop_time = now;
    timing->elapsed_total_seconds = now - timing->main_start_time;
    timing->used_iterations = iterations;
}
