/**
 * Simple Sequential Implementation of the K-Means Lloyds Algorithm
 */
#include "kmeans.h"
#include "kmeans_support.h"
#include "kmeans_sequential.h"
#include "log.h"

int num_points_total = 0;

struct pointset main_dataset;
struct pointset centroids;

void initialize(int max_points, struct kmeans_metrics *metrics)
{
    metrics->num_processors=1; // sequential - always one processor
    // for root we actually load the dataset, for others we just return the empty one
    allocate_pointset_points(&main_dataset, max_points);
    DEBUG("Allocated %d point space", max_points);
    num_points_total = load_dataset(&main_dataset);
    INFO("Loaded main dataset with %d points (confirmation: %d)", num_points_total, main_dataset.num_points);
}

/**
 * Assigns each point in the dataset to a cluster based on the distance from that cluster.
 *
 * The return value indicates how many points were assigned to a _different_ cluster
 * in this assignment process: this indicates how close the algorithm is to completion.
 * When the return value is zero, no points changed cluster so the clustering is complete.
 */
int assign_clusters()
{
    TRACE("Starting assign_clusters with %d datapoints", main_dataset.num_points);
    int total_reassignments = simple_assign_clusters(&main_dataset, &centroids);
    TRACE("Leaving assign_clusters with %d changes", total_reassignments);
    return total_reassignments;
}

/**
 * Calculates new centroids for the clusters of the given dataset by finding the
 * mean x and y coordinates of the current members of the cluster for each cluster.
 */
void calculate_centroids()
{
    TRACE("Starting calculate_centroids");
    simple_calculate_centroids(&main_dataset, &centroids);
    TRACE("Leaving calculate_centroids");
}

void initialize_representatives(int num_clusters)
{
    allocate_pointset_points(&centroids, num_clusters);
    initialize_centroids(&main_dataset, &centroids);
}


bool is_done(int changes, int iterations, int max_iterations)
{
    // only root completes the loop
    if (changes == 0 || iterations >= max_iterations) {
        INFO("Done with %d changes after %d iterations", changes, iterations);
        return true;
    }
    else {
        return false;
    }
}

void start_main_timing(struct kmeans_timing *timing)
{
    simple_start_main_timing(timing);
}

void start_iteration_timing(struct kmeans_timing *timing)
{
    simple_start_iteration_timing(timing);
}

void between_assignment_centroids(struct kmeans_timing *timing)
{
    simple_between_assignment_centroids(timing);
}

void end_iteration_timing(struct kmeans_timing *timing)
{
    simple_end_iteration_timing(timing);
}

void end_main_timing(struct kmeans_timing *timing, int iterations)
{
    simple_end_main_timing(timing, iterations);
}

void run(int max_iterations, struct kmeans_timing *timing)
{
    main_loop(max_iterations, timing);
}

void finalize(struct kmeans_metrics *metrics, struct kmeans_timing *timing)
{
    metrics->num_points = num_points_total;
    main_finalize(&main_dataset, metrics, timing);
}



