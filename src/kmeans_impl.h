#ifndef KMEANS_IMPL_H
#define KMEANS_IMPL_H

#include <stdbool.h>
#include "kmeans.h"

extern void initialize(int max_data, struct kmeans_metrics *metrics);
extern void initialize_representatives(int num_clusters);
extern int assign_clusters();
extern void run(int max_iterations, struct kmeans_timing *timing);
extern void finalize(struct kmeans_metrics *metrics, struct kmeans_timing *p_timing);
extern bool is_done(int changes, int iterations, int max_iterations);
extern void calculate_centroids();
extern void start_iteration_timing(struct kmeans_timing *p_timing);
extern void between_assignment_centroids(struct kmeans_timing *timing);
extern void end_iteration_timing(struct kmeans_timing *timing);
extern void start_main_timing(struct kmeans_timing *timing);
extern void end_main_timing(struct kmeans_timing *timing, int iterations);

#endif
