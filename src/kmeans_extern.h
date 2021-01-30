#ifndef KMEANS_EXTERN_H
#define KMEANS_EXTERN_H

#include "kmeans.h"
#include "csvhelper.h"
#include "log.h"

// indicates that the cluster ID should not be set or modified
#define IGNORE_CLUSTER_ID -2
// indicates that no cluster ID has been assigned to this element of a pointset
#define NO_CLUSTER_ID -1

// global config
extern struct log_config *log_config;
extern struct kmeans_config *kmeans_config;


extern struct log_config *new_log_config();
extern struct kmeans_config *new_kmeans_config();
extern void parse_kmeans_cli(int argc, char *argv[], struct kmeans_config *kmeans_config, struct log_config *log_config);
extern struct kmeans_metrics *new_kmeans_metrics(struct kmeans_config *config);

// basic pointset management
extern struct pointset *allocate_pointset(int num_points);
extern void check_bounds(struct pointset *pointset, int index);
extern void set_point(struct pointset *pointset, int index, double x, double y, int cluster_id);
extern void set_cluster(struct pointset *pointset, int index, int cluster_id);
extern void copy_points(struct pointset *source, struct pointset *target, int start_index, int size, bool include_cluster);
extern void copy_point(struct pointset *source, struct pointset *target, int index, bool include_cluster);
extern bool same_point(struct pointset *pointset1, struct pointset *pointset2, int index);
extern bool same_cluster(struct pointset *pointset1, struct pointset *pointset2, int index);
extern double point_distance(struct pointset *pointset1, int index1, struct pointset *pointset2, int index2);

extern const char *p_to_s(struct pointset *dataset, int index);
extern double euclidean_distance(double x2, double y2, double x1, double y1);
extern void kmeans_usage();
extern void print_points(FILE *out, struct pointset *dataset, const char *label);
extern void print_headers(FILE *out, char **headers, int dimensions);
extern void print_metrics_headers(FILE *out);
extern void print_centroids(FILE *out, struct pointset *centroids, char *label);


extern void debug_setup(struct pointset *dataset, struct pointset *centroids);
extern void print_metrics(FILE *out, struct kmeans_metrics *metrics);
extern int read_csv_file(char* csv_file_name, struct pointset *dataset, int max_points, char *headers[], int *dimensions);
extern int read_csv(FILE* csv_file, struct pointset *dataset, int max_points, char *headers[], int *dimensions);
extern void write_csv_file(char *csv_file_name, struct pointset *dataset, char *headers[], int dimensions);
extern void write_csv(FILE *csv_file, struct pointset *dataset, char *headers[], int dimensions);

extern void write_metrics_file(char *metrics_file_name, struct kmeans_metrics *metrics) ;

extern char* valid_file(char opt, char *filename);
extern int valid_count(char opt, char *arg);
extern void validate_config(struct log_config config);

extern int test_results(char *test_file_name, struct pointset *dataset);

extern void trace_assignment(struct pointset *dataset, int dataset_index,
                             struct pointset *centroids, int centroid_index, double min_distance);

// help with debugging OMP
extern int omp_schedule_kind(int *chunk_size);
extern void omp_debug(char *msg);

#endif