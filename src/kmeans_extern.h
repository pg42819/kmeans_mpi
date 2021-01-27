#ifndef KMEANS_EXTERN_H
#define KMEANS_EXTERN_H

#include "kmeans.h"
#include "csvhelper.h"
#include "log.h"

// global config
extern struct log_config *log_config;
extern struct kmeans_config *kmeans_config;


extern struct log_config *new_log_config();
extern struct kmeans_config *new_kmeans_config();
extern void parse_kmeans_cli(int argc, char *argv[], struct kmeans_config *kmeans_config, struct log_config *log_config);
extern struct kmeans_metrics *new_kmeans_metrics(struct kmeans_config *config);

extern const char *p_to_s(struct point *p);
extern double euclidean_distance(struct point *p1, struct point *p2);
extern void kmeans_usage();
extern void print_points(FILE *out, struct point *dataset, int num_points);
extern void print_headers(FILE *out, char **headers, int dimensions);
extern void print_metrics_headers(FILE *out);
extern void print_centroids(FILE *out, struct point *centroids, int num_points);

extern void print_metrics(FILE *out, struct kmeans_metrics *metrics);
extern int read_csv_file(char* csv_file_name, struct point *dataset, int max_points, char *headers[], int *dimensions);
extern int read_csv(FILE* csv_file, struct point *dataset, int max_points, char *headers[], int *dimensions);
extern void write_csv_file(char *csv_file_name, struct point *dataset, int num_points, char *headers[], int dimensions);
extern void write_csv(FILE *csv_file, struct point *dataset, int num_points, char *headers[], int dimensions);

extern void write_metrics_file(char *metrics_file_name, struct kmeans_metrics *metrics) ;

extern char* valid_file(char opt, char *filename);
extern int valid_count(char opt, char *arg);
extern void validate_config(struct log_config config);

extern int test_results(char *test_file_name, struct point *dataset, int num_points);


extern void debug_assignment(struct point *p, int closest_cluster, struct point *centroid, double min_distance);

// help with debugging OMP
extern int omp_schedule_kind(int *chunk_size);
extern void omp_debug(char *msg);

#endif