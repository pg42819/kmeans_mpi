#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include <float.h>
#include "kmeans.h"
#include "kmeans_support.h"
#include "csvhelper.h"
#include "log.h"


/**
 * Allocate points in a pointset struct that already exists
 * Useful for creating static structs in the module or on a function stack
 * @param new_pointset pointer to existing pointset struct
 * @param num_points number of points to allocate
 */
void allocate_pointset_points(struct pointset *new_pointset, int num_points)
{
    new_pointset->x_coords = (double *)malloc(num_points * sizeof(double));
    new_pointset->y_coords = (double *)malloc(num_points * sizeof(double));
    new_pointset->cluster_ids = (int *)malloc(num_points * sizeof(int));

    if (new_pointset->x_coords == NULL || new_pointset->y_coords == NULL || new_pointset->cluster_ids == NULL) {
        FAIL("Failed to allocate coordinates for a pointset of size %d", num_points);
    }

    new_pointset->num_points = num_points;
}

/**
 * Allocate memory for a new point set of size.
 * IMPORTANT: caller responsible for freeing the pointset later
 * @param num_points size of the pointset to create
 * @return a pointer to a newly allocated pointset
 */
struct pointset *allocate_pointset(int num_points)
{
    struct pointset *new_pointset = (struct pointset *)malloc(sizeof(struct pointset));

    allocate_pointset_points(new_pointset, num_points);
    return new_pointset;
}


/**
 * Fail and exit if the index is outside the bounds of the pointset or the pointset is null
 * Note pointsets start at 0, so if index == pointset.num_points that counts as failure
 *
 * @param pointset pointset to check
 * @param index index to check
 */
void check_bounds(struct pointset *pointset, int index)
{
    if (pointset == NULL) {
        FAIL("Attempted to reference a NULL pointset");
    }
    if (index >= pointset->num_points) {
        FAIL("Attempted to reference a point outsize the pre-allocated size for the pointset: %d", index);
    }
}

/**
 * Assign a cluster to a point in a pointset
 * @param pointset to set in
 * @param index index of point to set
 * @param cluster_id cluster to set: use IGNORE_CLUSTER_ID (-2)to set the cluster_id but leave it as is
 */
void set_cluster(struct pointset *pointset, int index, int cluster_id)
{
    check_bounds(pointset, index);
    if (cluster_id != IGNORE_CLUSTER_ID) {
        // set cluster id too
        pointset->cluster_ids[index] = cluster_id;
    }
}

/**
 * Set a point in a pointset
 * @param pointset to set in
 * @param index index of point to set
 * @param x x coordinate to set
 * @param y y coordinate to set
 * @param cluster_id cluster to set: use IGNORE_CLUSTER_ID (-2)to set the cluster_id but leave it as is
 */
void set_point(struct pointset *pointset, int index, double x, double y, int cluster_id)
{
    check_bounds(pointset, index);
    pointset->x_coords[index] = x;
    pointset->y_coords[index] = y;
    set_cluster(pointset, index, cluster_id);
}

/**
 * Copy a point from one point set to another
 * @param source set to copy from
 * @param target set to copy to
 * @param index index of point to copy
 * @param include_cluster true to copy the cluster Id too
 */
void copy_point(struct pointset *source, struct pointset *target, int index, bool include_cluster)
{
    check_bounds(source, index);
    int cluster_id = include_cluster ? source->cluster_ids[index] : IGNORE_CLUSTER_ID;
    set_point(target, index, source->x_coords[index], source->y_coords[index], cluster_id);
}

/**
 * Copy a subset of points from one point set to another
 * @param source set to copy from
 * @param target set to copy to
 * @param start_index index of the first point to copy
 * @param size the number of points to copy
 * @param include_cluster true to copy the cluster Ids too
 */
void copy_points(struct pointset *source, struct pointset *target, int start_index, int size, bool include_cluster)
{
    check_bounds(source, start_index + size - 1);
    check_bounds(target, start_index + size - 1);
    for (int i = 0; i < size; ++i) {
        copy_point(source, target, start_index + i, include_cluster);
    }
}

/**
 * Returns true if the coordinates at the index of both points are equal
 */
bool same_point(struct pointset *pointset1, struct pointset *pointset2, int index)
{
    check_bounds(pointset1, index);
    check_bounds(pointset2, index);
    return (pointset1->x_coords[index] == pointset2->x_coords[index] &&
            pointset1->y_coords[index] == pointset2->y_coords[index]);
}

/**
 * Returns the eclidean distance between the points in the pointsets at the given index
 */
double point_distance(struct pointset *pointset1, int index1, struct pointset *pointset2, int index2)
{
    check_bounds(pointset1, index1);
    check_bounds(pointset2, index2);
    return euclidean_distance(pointset2->x_coords[index2], pointset2->y_coords[index2],
                              pointset1->x_coords[index1], pointset1->y_coords[index1]);
}

/**
 * Returns true if the cluster_ids at the index of both points are equal
 */
bool same_cluster(struct pointset *pointset1, struct pointset *pointset2, int index)
{
    check_bounds(pointset1, index);
    check_bounds(pointset2, index);
    return (pointset1->cluster_ids[index] == pointset2->cluster_ids[index]);
}

/**
 * Calculate the SQUARE of euclidean distance between two points.
 *
 * That is, the sum of the squares of the distances between coordinates
 *
 * Note we work with the pure sum of squares - so the _square_ of the distance,
 * since we really only need the _relative_ distances to assign_clusters a point to
 * the right cluster - and square-root is a slow function.
 * We have an option to enable the proper square if we need to
 *
 * The points are expected as pointers since the method does not change them and memory and time is
 * saved by not copying the structs unnecessarily.
 */
double euclidean_distance(double x2, double y2, double x1, double y1)
{
    double square_diff_x = (x2 - x1) * (x2 - x1);
    double square_diff_y = (y2 - y1) * (y2 - y1);
    double square_dist = square_diff_x + square_diff_y;
    // most k-means algorithms would stop here and return the square of the euclidean distance
    // because its faster, and we only need comparative values for clustering
    // Here we have the option of doing it either way - defaulting to the performant no-sqrt
    double dist = kmeans_config->proper_distance ? sqrt(square_dist) : square_dist;
    TRACE("Distance from (%.7f,%.7f) -> (%.7f,%.7f) = %f", x2, y2, x1, y1, dist);
    return dist;
}

/**
 * Convert a point to a string with a standard precision
 *
 * @param p point to print to string
 * @return allocated string holding point
 */
const char *p_to_s(struct pointset *dataset, int index)
{
    // TODO this eats a lot of memory when in a big loop - consider passing in a string to reuse
    // but for now we keep string printing OUT of timed sections so it won't have much affect
    char *result = (char *)malloc(50 * sizeof(char)); // big enough for 2 points
    sprintf(result, "%.7f,%.7f", dataset->x_coords[index], dataset->y_coords[index]);
    return result;
}

/**
 * Print dataset of points to a file pointer (may be stdout) including cluster assignment
 *
 * @param out file pointer for output
 * @param dataset array of points
 * @param num_points size of the array
 */
void print_points(FILE *out, struct pointset *dataset, const char *label) {
    if (label == NULL) {
        label = "";
    }
    for (int i = 0; i < dataset->num_points; ++i) {
        fprintf(out, "%s%s,cluster_%d\n", label, p_to_s(dataset, i), dataset->cluster_ids[i]);
    }
}

void debug_points(struct pointset *dataset, const char *label)
{
    if (IS_DEBUG) {
        print_points(stdout, dataset, label);
    }
}

/**
 * Print the set of centroids for the clusters to a file pointer (may be stdout)
 *
 * @param out file pointer for output
 * @param centroids array of centroid points
 * @param num_points number of centroids == number of clusters
 */
void print_centroids(FILE *out, struct pointset *centroids, char *label) {
    if (label == NULL) {
        label = "";
    }
    for (int i = 0; i < centroids->num_points; ++i) {
        fprintf(out, "%scentroid[%d] is at %s\n", label, i, p_to_s(centroids, i));
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
                 "num_clusters,max_iterations,num_processors,"
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
    fprintf(out, "%s,%d,%f,%f,%f,%f,%d,%d,%d,%d,%s\n",
            metrics->label, metrics->used_iterations, metrics->total_seconds,
            metrics->assignment_seconds, metrics->centroids_seconds, metrics->max_iteration_seconds,
            metrics->num_points, metrics->num_clusters, metrics->max_iterations,
            metrics->num_processors,
            test_results);
}

/**
 * Summarize the results of the run with timing numbers in a single row to go in a csv file
 * @param out output file pointer
 * @param metrics metrics object
 */
void summarize_metrics(FILE *out, struct kmeans_metrics *metrics)
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
    fprintf(out, "Run Label       : %s\n"
                 "Dataset size  N : %d\n"
                 "Num Clusters  K : %d\n"
                 "Total seconds   : %f\n"
                 "Iterations      : %d\n"
                 "Num Processors  : %d\n"
                 "Test            : %s\n",
            metrics->label, metrics->num_points, metrics->num_clusters, metrics->total_seconds,
            metrics->used_iterations, metrics->num_processors, test_results);
}

/**
 * Read 2-dimensional points from the CSV file with headers.
 *
 * @param csv_file file pointer to the input file
 * @param dataset pre-allocated dataset into which to read the file
 * @param max_points max number of points to read
 * @param headers if not null, pre-allocated string array to hold the headers
 * @param dimensions number of headers
 *
 * @return number of actual points read from the file
 */
int read_csv(FILE* csv_file, struct pointset *dataset, int max_points, char *headers[], int *dimensions)
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
            check_bounds(dataset, count);
            char *x_string = csvfield(0);
            char *y_string = csvfield(1);
            set_point(dataset, count, strtod(x_string, NULL), strtod(y_string, NULL), NO_CLUSTER_ID);

            if (num_fields > 2 && *dimensions > 2) {
                char *cluster_string = csvfield(2);
                int cluster;
                char prefix[200];
                sscanf(cluster_string,"%[^0-9]%d", prefix, &cluster);
                dataset->cluster_ids[count] = cluster;
            }
            count++;
        }
    }
    fclose(csv_file);

    // update the dataset length to match the count
    dataset->num_points = count;
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
 * @param dataset pointset of all points with cluster information
 * @param num_points size of the array of points
 * @param headers optional headers to put at the top of the file
 * @param dimensions number of headers
*/
int read_csv_file(char* csv_file_name, struct pointset *dataset, int max_points, char *headers[], int *dimensions)
{
    FILE *csv_file = fopen(csv_file_name, "r");
    if (!csv_file) {
        fprintf(stderr, "Error: cannot read the input file at %s\n", csv_file_name);
        exit(1);
    }
    return read_csv(csv_file, dataset, max_points, headers, dimensions);
}

/**
 * Write a Comma-Separated-Values file with points and cluster assignments to a pointset.
 *
 * @param csv_file pointer to the file to write to
 * @param dataset array of all points with cluster information
 * @param num_points size of the array of points
 * @param headers optional headers to put at the top of the file
 * @param dimensions number of headers
 */
void write_csv(FILE *csv_file, struct pointset *dataset, char *headers[], int dimensions)
{
    if (headers != NULL) {
        print_headers(csv_file, headers, dimensions);
    }

    print_points(csv_file, dataset, NULL);
}

void write_csv_file(char *csv_file_name, struct pointset *dataset, char *headers[], int dimensions) {
    FILE *csv_file = fopen(csv_file_name, "w");
    if (!csv_file) {
        FAIL("Cannot write to the output file at %s\n", csv_file_name);
    }

    write_csv(csv_file, dataset, headers, dimensions);
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
int test_results(char *test_file_name, struct pointset *dataset)
{
    int result = 1;
    int num_points = dataset->num_points;
    struct pointset *testset = allocate_pointset(num_points + 10);
    int test_dimensions;
    static char* test_headers[3];
    int num_test_points = read_csv_file(test_file_name, testset, num_points, test_headers, &test_dimensions);
    if (num_test_points < num_points) {
        WARN("Test failed. The test dataset has only %d records, but needs at least %d",
             num_test_points, num_points);
        result = 1;
    }
    else {
        for (int n = 0; n < num_points; ++n) {
            if (same_point(testset, dataset, n)) {
                if (!same_cluster(testset, dataset, n)) {
                    // points match but assigned to different clusters
                        WARN("Test failure at %d: (%s) result cluster: %d does not match test: %d\n",
                                n + 1, p_to_s(dataset, n), dataset->cluster_ids[n], testset->cluster_ids[n]);
                    result = -1;
                    break; // give up comparing
                }
                else {
                    TRACE("Test success at %d: (%s) clusters match: %d\n",
                          n+1, p_to_s(dataset, n), dataset->cluster_ids[n]);
                }
            }
            else {
                // points themselves are different
                WARN("Test failure at %d: %s does not match test point: %s\n",
                     n+1, p_to_s(dataset, n), p_to_s(testset, n));
                result = -1;
                break; // give up comparing
            }
        }
    }
    return result;
}
