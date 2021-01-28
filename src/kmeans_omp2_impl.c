#include <float.h>
#include <math.h>
#include "kmeans.h"
#include "kmeans_extern.h"
#include "log.h"

/**
 * OpenMP performance version 2:
 * - share the thread team across for loops of the same size
 * - use reduction for the point counter
 */

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
int assign_clusters(struct pointset* dataset, struct pointset *centroids)
{
    DEBUG("\nStarting assignment phase:\n");
    int cluster_changes = 0;
    int num_points = dataset->num_points;
    int num_clusters = centroids->num_points;
    // use reduction + for cluster changes which are added up
#pragma omp parallel for schedule(runtime) reduction(+:cluster_changes)
    for (int n = 0; n < num_points; ++n) {
#ifdef DEBUG_OMP
        char msg[50];
        sprintf(msg, "Point %d", n);
        omp_debug(msg);
#endif
        double min_distance = DBL_MAX; // init the min distance to a big number
        int closest_cluster = -1;

// Nested for loop parallelism - consider collapse instead of this or nowait
#pragma omp parallel for schedule(runtime) shared(n)
        for (int k = 0; k < num_clusters; ++k) {
            // calc the distance passing pointers to points since the distance does not modify them
            double distance_from_centroid = point_distance(dataset, n, centroids, k);
            if (distance_from_centroid < min_distance) {
                min_distance = distance_from_centroid;
                closest_cluster = k;
            }
        }
        // if the point was not already in the closest cluster, move it there and count changes

        if (dataset->cluster_ids[n] != closest_cluster) {
            dataset->cluster_ids[n] = closest_cluster;
            cluster_changes++;
            if (log_config->verbose) {
                debug_assignment(dataset, n, centroids, closest_cluster, min_distance);
            }
        }
    }
    return cluster_changes;
}

/**
 * Calculates new centroids for the clusters of the given dataset by finding the
 * mean x and y coordinates of the current members of the cluster for each cluster.
 *
 * @param dataset set of all points with current cluster assignments
 * @param centroids array to hold the centroids - already allocated
 */
void calculate_centroids(struct pointset* dataset, struct pointset *centroids)
{
    int num_points = dataset->num_points;
    int num_clusters = centroids->num_points;
    double sum_of_x_per_cluster[num_clusters];
    double sum_of_y_per_cluster[num_clusters];
    int num_points_in_cluster[num_clusters];

// reuse the thread team across the for loops
#pragma omp parallel
{

#pragma for schedule(runtime)
    for (int k = 0; k < num_clusters; ++k) {
        sum_of_x_per_cluster[k] = 0.0;
        sum_of_y_per_cluster[k] = 0.0;
        num_points_in_cluster[k] = 0;
    }

    // loop over all points in the database and sum up
    // the x coords of clusters to which each belongs
#pragma omp for schedule(runtime)
    for (int n = 0; n < num_points; ++n) {
        int k = dataset->cluster_ids[n];
        sum_of_x_per_cluster[k] += dataset->x_coords[n];
        sum_of_y_per_cluster[k] += dataset->y_coords[n];
        // count the points in the cluster to get a mean later
        num_points_in_cluster[k]++;
    }

    // the new centroids are at the mean x and y coords of the clusters
#pragma omp for schedule(runtime)
    for (int k = 0; k < num_clusters; ++k) {
        // mean x, mean y => new centroid
        double new_centroid_x = sum_of_x_per_cluster[k] / num_points_in_cluster[k];
        double new_centroid_y = sum_of_y_per_cluster[k] / num_points_in_cluster[k];
        set_point(centroids, k, new_centroid_x, new_centroid_y, IGNORE_CLUSTER_ID);
    }
}
}




