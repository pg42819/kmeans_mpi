#ifndef KMEANS_SEQUENTIAL_H
#define KMEANS_SEQUENTIAL_H


extern void simple_calculate_centroids(struct pointset *dataset, struct pointset *centroids);
extern int simple_assign_clusters(struct pointset *dataset, struct pointset *centroids);
extern void initialize_centroids(struct pointset* dataset, struct pointset *centroids);
extern void simple_start_iteration_timing(struct kmeans_timing *timing);
extern void simple_between_assignment_centroids(struct kmeans_timing *timing);
extern void simple_end_iteration_timing(struct kmeans_timing *timing);
extern void simple_start_main_timing(struct kmeans_timing *timing);
extern void simple_end_main_timing(struct kmeans_timing *timing, int iterations);

#endif