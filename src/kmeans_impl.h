#ifndef KMEANS_IMPL_H
#define KMEANS_IMPL_H

#include <stdbool.h>

extern void initialize(int max_data);
extern void initialize_representatives(int num_clusters);
extern int assign_clusters();
extern void run(int max_iterations, struct kmeans_metrics *metrics);
extern void finalize(struct kmeans_metrics *metrics);
extern bool is_done(int changes, int iterations, int max_iterations);

/**
 * Calculates new centroids for the clusters of the given dataset by finding the
 * mean x and y coordinates of the current members of the cluster for each cluster.
 *
 * The centroids are set in the array passed in, which is expected to be pre-allocated
 * and contain the previous centroids: these are overwritten by the new values.
 */
extern void calculate_centroids();

#endif
