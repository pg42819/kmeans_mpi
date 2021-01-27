#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <unistd.h>
#include "csvhelper.h"
#include "kmeans.h"
#include "kmeans_extern.h"
#include "log.h"
#include <math.h>
#include <omp.h>

/**
 * Convert the omp schedule kind to an int for easy graphing and
 * handle the OMP 4.5 introduction of monotonic for static by returning zero.
 *
 * @param chunk_size pointer to an int to hold the chunk size
 * @return an integer representing the OpenMP schedule kind:
 *      0 = monotonic (OMP 4.5+)
 *      1 = static
 *      2 = dynamic (default)
 *      3 = guided
 *      4 = auto
 */
int omp_schedule_kind(int *chunk_size)
{
    int chunk_s = -1; // create our own in case chunk_size is a null pointer
    enum omp_sched_t kind = omp_sched_static;
    omp_get_schedule(&kind, &chunk_s);

//    printf("kind   : %d\n", kind);
//    printf("omp_sched_static    : %d\n", omp_sched_static);
//    printf("omp_sched_dynamic   : %d\n", omp_sched_dynamic);
//    printf("omp_sched_guided    : %d\n", omp_sched_guided);
//    printf("omp_sched_auto      : %d\n", omp_sched_auto);
//    printf("omp_sched_monotonic : %d\n", omp_sched_monotonic);
    // allow for chunk_size null if we don't care about it otherwise assign it
    if (chunk_size != NULL) {
        *chunk_size = chunk_s;
    }

    if (kind < -1) {
        // on mac os the OMP_SCHEDULE variable value "static" results in -2147483647 (-MAX_INT)
        // which is probably meant to match the omp_sched_monotonic enum but misses by 1
        // But we switch it for 1 anyway to simulate static on the Linux SEARCH server which is
        // where this program is finally run anyway (the Mac OS run is just for dev/debug)
        // Note that monotonic was introduced in OpenMP 4.5
        return 1;
    }
    return (int)kind;
}

/**
 * Print debug information for the assignment of a point to a cluster
 */
void debug_assignment(struct point *p, int closest_cluster, struct point *centroid, double min_distance)
{
    if (log_config->verbose) {
        printf("Assigning (%s) to cluster %d with centroid (%s) d = %f\n",
               p_to_s(p), closest_cluster, p_to_s(centroid), min_distance);
    }
}

void omp_debug(char *msg) {
    int chunks = -1;
    int kind = omp_schedule_kind(&chunks);
    printf("%s: OMP schedule kind %d with chunk size %d on thread %d of %d\n",
           msg, kind, chunks, omp_get_thread_num(), omp_get_num_threads());
}

/**
 * Calculate the SQAURE of euclidean distance between two points.
 *
 * That is, the sum of the squares of the distances between coordinates
 *
 * Note we work with the pure sum of squares - so the _square_ of the distance,
 * since we really only need the _relative_ distances to assign a point to
 * the right cluster - and square-root is a slow function.
 * We have an option to enable the proper square if we need to
 *
 * The points are expected as pointers since the method does not change them and memory and time is
 * saved by not copying the structs unnecessarily.
 *
 * @param p1 pointer to first point in 2 dimensions
 * @param p2 ponter to second point in 2 dimensions
 * @return geometric distance between the 2 points
 */
double euclidean_distance(struct point *p1, struct point *p2)
{
    double square_diff_x = (p2->x - p1->x) * (p2->x - p1->x);
    double square_diff_y = (p2->y - p1->y) * (p2->y - p1->y);
    double square_dist = square_diff_x + square_diff_y;
    // most k-means algorithms would stop here and return the square of the euclidean distance
    // because its faster, and we only need comparative values for clustering
    // Here we have the option of doing it either way - defaulting to the performant no-sqrt
    double dist = kmeans_config->proper_distance ? sqrt(square_dist) : square_dist;
    VERBOSE("Distance from (%s) -> (%s) = %f\n", p_to_s(p1), p_to_s(p2), dist);
    return dist;
}

/**
 * Convert a point to a string with a standard precision
 *
 * @param p point to print to string
 * @return allocated string holding point
 */
const char *p_to_s(struct point *p)
{
    // TODO this eats a lot of memory when in a big loop - consider passing in a string to reuse
    // but for now we keep string printing out of timed sections so it won't have much affect
    char *result = malloc(50 * sizeof(char)); // big enough for 2 points
    sprintf(result, "%.7f,%.7f", p->x, p->y);
    return result;
}

/**
 * Print dataset of points to a file pointer (may be stdout) including cluster assignment
 *
 * @param out file pointer for output
 * @param dataset array of points
 * @param num_points size of the array
 */
void print_points(FILE *out, struct point *dataset, int num_points) {
    for (int i = 0; i < num_points; ++i) {
        fprintf(out, "%s,cluster_%d\n", p_to_s(&dataset[i]), dataset[i].cluster);
    }
}

/**
 * Print the set of centroids for the clusters to a file pointer (may be stdout)
 *
 * @param out file pointer for output
 * @param centroids array of centroid points
 * @param num_points number of centroids == number of clusters
 */
void print_centroids(FILE *out, struct point *centroids, int num_points) {
    for (int i = 0; i < num_points; ++i) {
        fprintf(out, "centroid[%d] is at %s\n", i, p_to_s(&centroids[i]));
    }
}

/**
 * Print headers for output CSV files
 * @param out file pointer for output
 * @param headers array of strings
 * @param dimensions number of strings in the array
 */
void print_headers(FILE *out, char **headers, int dimensions) {
    if (headers == NULL) return;

    fprintf(out, "%s", headers[0]);
    for (int i = 1; i < dimensions; ++i) {
        fprintf(out, ",%s", headers[i]);
    }
    // add a 3rd header called "Cluster" to match the Knime output for easier comparison
    fprintf(out, ",Cluster\n");
}

/**
 * Print the headers for the metrics table to a file pointer.
 * Used for the first run to use a metrics file to produce the header row
 *
 * @param out file pointer
 */
void print_metrics_headers(FILE *out)
{
    fprintf(out, "label,used_iterations,total_seconds,assignments_seconds,"
                 "centroids_seconds,max_iteration_seconds,num_points,"
                 "num_clusters,max_iterations,max_threads,omp_schedule,omp_chunk_size,"
                 "test_results\n");
}

/**
 * Print the results of the run with timing numbers in a single row to go in a csv file
 * @param out output file pointer
 * @param metrics metrics object
 */
void print_metrics(FILE *out, struct kmeans_metrics *metrics)
{
    char *test_results = "untested";
    switch (metrics->test_result) {
        case 1:
            test_results = "passed";
            break;
        case -1:
            test_results = "FAILED!";
            break;
    }
    fprintf(out, "%s,%d,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%s\n",
            metrics->label, metrics->used_iterations, metrics->total_seconds,
            metrics->assignment_seconds, metrics->centroids_seconds, metrics->max_iteration_seconds,
            metrics->num_points, metrics->num_clusters, metrics->max_iterations,
            metrics->omp_max_threads, metrics->omp_schedule_kind, metrics->omp_chunk_size,
            test_results);
}

/**
 * Read 2-dimensional points from the CSV file with headers.
 *
 * @param csv_file file pointer to the input file
 * @param dataset pre-allocated array of points into which to read the file
 * @param max_points max number of points to read
 * @param headers if not null, pre-allocated string array to hold the headers
 * @param dimensions number of headers
 *
 * @return number of actual points read from the file
 */
int read_csv(FILE* csv_file, struct point *dataset, int max_points, char *headers[], int *dimensions)
{
    char *line;
    *dimensions = csvheaders(csv_file, headers);
    int max_fields = *dimensions > 2 ? 3 : 2; // max is 2 unless there is a cluster in which case 3
    int count = 0;
    while (count < max_points && (line = csvgetline(csv_file)) != NULL) {
        int num_fields = csvnfield(); // fields on the line
#ifdef DEBUG
        if (num_fields > max_fields) {
            printf("Warning: more that %d fields on line. Ignoring after the first %d: %s", max_fields, max_fields, line);
        }
#endif
        if (num_fields < 2) {
            printf("Warning: found non-empty trailing line. Will stop reading points now: %s", line);
            break;
        }
        else {
            struct point new_point;
            new_point.cluster = -1; // -1 => no cluster yet assigned
            char *x_string = csvfield(0);
            char *y_string = csvfield(1);
            new_point.x = strtod(x_string, NULL);
            new_point.y = strtod(y_string, NULL);

            if (num_fields > 2 && *dimensions > 2) {
                char *cluster_string = csvfield(2);
                int cluster;
                char prefix[200];
                sscanf(cluster_string,"%[^0-9]%d", prefix, &cluster);
                new_point.cluster = cluster;
            }
            dataset[count] = new_point;
            count++;
        }
    }
    fclose(csv_file);
    return count;
}

/**
 *
 * Write a Comma-Separated-Values file with points and cluster assignments
 * to the specified file path.
 *
 * IF the file exists it is silently overwritten.
 *
 * @param csv_file_name absolute path to the file to be written
 * @param dataset array of all points with cluster information
 * @param num_points size of the array of points
 * @param headers optional headers to put at the top of the file
 * @param dimensions number of headers
*/
int read_csv_file(char* csv_file_name, struct point *dataset, int max_points, char *headers[], int *dimensions)
{
    FILE *csv_file = fopen(csv_file_name, "r");
    if (!csv_file) {
        fprintf(stderr, "Error: cannot read the input file at %s\n", csv_file_name);
        exit(1);
    }
    return read_csv(csv_file, dataset, max_points, headers, dimensions);
}

/**
 * Write a Comma-Separated-Values file with points and cluster assignments to a file pointer.
 *
 * @param csv_file pointer to the file to write to
 * @param dataset array of all points with cluster information
 * @param num_points size of the array of points
 * @param headers optional headers to put at the top of the file
 * @param dimensions number of headers
 */
void write_csv(FILE *csv_file, struct point *dataset, int num_points, char *headers[], int dimensions)
{
    if (headers != NULL) {
        print_headers(csv_file, headers, dimensions);
    }

    print_points(csv_file, dataset, num_points);
}

void write_csv_file(char *csv_file_name, struct point *dataset, int num_points, char *headers[], int dimensions) {
    FILE *csv_file = fopen(csv_file_name, "w");
    if (!csv_file) {
        fprintf(stderr, "Error: cannot write to the output file at %s\n", csv_file_name);
        exit(1);
    }

    write_csv(csv_file, dataset, num_points, headers, dimensions);
}

void write_metrics_file(char *metrics_file_name, struct kmeans_metrics *metrics) {
    char *mode = "a"; // default to append to the metrics file
    bool first_time = false;
    if (access(metrics_file_name, F_OK ) == -1 ) {
        // first time - lets change the mode to "w" and append
        fprintf(stdout, "Creating metrics file and adding headers: %s", metrics_file_name);
        first_time = true;
        mode = "w";
    }
    FILE *metrics_file = fopen(metrics_file_name, mode);
    if (first_time) {
        print_metrics_headers(metrics_file);
    }

    print_metrics(metrics_file, metrics);
}

/**
 * Compares the dataset against test file.
 *
 * If every point in the dataset has a matching point at the same position in the
 * test dataset from the test file, and the clusters match, then 1 is returned,
 * otherwise -1 is returned indicating a failure.
 *
 * Note that the test file may have more points than the dataset - trailing points are ignored
 * in this case - but if it has fewer points, this is considered a test failure.
 *
 * The method returns -1 after the first failure.
 *
 * @param config
 * @param dataset
 * @param num_points
 * @return 1 or -1 if the files match
 */
int test_results(char *test_file_name, struct point *dataset, int num_points)
{
    int result = 1;
    struct point *testset = malloc((num_points + 10) * sizeof(struct point));
    int test_dimensions;
    static char* test_headers[3];
    int num_test_points = read_csv_file(test_file_name, testset, num_points, test_headers, &test_dimensions);
    if (num_test_points < num_points) {
        if (!log_config->silent) {
        fprintf(stderr, "Test failed. The test dataset has only %d records, but needs at least %d",
                num_test_points, num_points);
        }
        result = 1;
    }
    else {
        for (int n = 0; n < num_points; ++n) {
            struct point *p = &dataset[n];
            struct point *test_p = &testset[n];
            if (test_p->x == p->x && test_p->y == p->y) {
                if (test_p->cluster != p->cluster) {
                    // points match but assigned to different clusters
                    if (!log_config->silent) {
                        fprintf(stderr, "Test failure at %d: (%s) result cluster: %d does not match test: %d\n",
                                n + 1, p_to_s(p), p->cluster, test_p->cluster);
                    }
                    result = -1;
                    break; // give up comparing
                }
#ifdef TRACE
                else {
                    fprintf(stdout, "Test success at %d: (%s) clusters match: %d\n",
                            n+1, p_to_s(p), p->cluster);

                }
#endif
            }
            else {
                // points themselves are different
                if (!log_config->silent) {
                fprintf(stderr, "Test failure at %d: %s does not match test point: %s\n",
                        n+1, p_to_s(p), p_to_s(test_p));

                }
                result = -1;
                break; // give up comparing
            }
        }
    }
    return result;
}

