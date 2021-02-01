#include <float.h>
#include "kmeans.h"
#include "kmeans_support.h"
#include "kmeans_impl.h"
#include "log.h"

static char* headers[3];
static int dimensions;
struct kmeans_config *kmeans_config;
enum log_level_t log_level;

int load_dataset(struct pointset *dataset)
{
    char *csv_file_name = valid_file('f', kmeans_config->in_file);
    int num_points = read_csv_file(csv_file_name, dataset, kmeans_config->max_points, headers, &dimensions);
    DEBUG("Loaded %d points from the dataset file at %s", num_points, csv_file_name);
    return num_points;
}

void main_loop(int max_iterations, struct kmeans_timing *timing)
{
    // we deliberately skip the centroid initialization phase in calculating the
    // total time as it is constant and never optimized
    start_main_timing(timing);
    int cluster_changes = MAX_POINTS; // start at a max then work down to zero chagnes
    int iterations = 0;

    while (!is_done(cluster_changes, iterations, max_iterations)) {
        DEBUG("Starting iteration %d. %d change in last iteration", iterations, cluster_changes);
        // K-Means Algo Step 2: assign_clusters every point to a cluster (closest centroid)

        start_iteration_timing(timing);

        TRACE("calling assign_clusters");
        cluster_changes = assign_clusters();
        TRACE("returned from assign_clusters");

        between_assignment_centroids(timing);

        // K-Means Algo Step 3: calculate new centroids: one at the center of each cluster
        TRACE("calling calculate_centroids");
        calculate_centroids();
        TRACE("returned from calculate_centroids");

        end_iteration_timing(timing);
        iterations++;
    }

    end_main_timing(timing, iterations);

    INFO("Ended after %d iterations with %d changed clusters\n", iterations, cluster_changes);
}

void main_finalize(struct pointset *dataset, struct kmeans_metrics *metrics, struct kmeans_timing *timing)
{
    // update timings on metrics
    metrics->assignment_seconds = timing->accumulated_assignment_seconds;
    metrics->centroids_seconds = timing->accumulated_centroids_seconds;
    metrics->max_iteration_seconds = timing->max_iteration_seconds;
    metrics->total_seconds = timing->elapsed_total_seconds;
    metrics->used_iterations = timing->used_iterations;

    // output file is not always written: sometimes we only run for metrics and compare with test data
    if (kmeans_config->out_file) {
        INFO("Writing output to %s\n", kmeans_config->out_file);
        write_csv_file(kmeans_config->out_file, dataset, headers, dimensions);
    }

    if (IS_DEBUG) {
        write_csv(stdout, dataset, headers, dimensions);
    }

    if (kmeans_config->test_file) {
        char *test_file_name = valid_file('t', kmeans_config->test_file);
        INFO("Comparing results against test file: %s\n", kmeans_config->test_file);
        metrics->test_result = test_results(test_file_name, dataset);
    }

    if (kmeans_config->metrics_file) {
        // metrics file may or may not already exist
        INFO("Reporting metrics to: %s\n", kmeans_config->metrics_file);
        write_metrics_file(kmeans_config->metrics_file, metrics);
    }

    if (IS_VERBOSE) {
        print_points(stdout, dataset, "Final ");
    }

    if (IS_INFO) {
        summarize_metrics(stdout, metrics);
        printf("\n");
    }

    if (IS_WARN) {
        print_metrics_headers(stdout);
        print_metrics(stdout, metrics);
    }
}

int main(int argc, char* argv [])
{
    kmeans_config = new_kmeans_config();
    parse_kmeans_cli(argc, argv, kmeans_config, &log_level);

    // set up a metrics struct to hold timing and other info for comparison
    struct kmeans_metrics *metrics = new_kmeans_metrics(kmeans_config);

    DEBUG("Initializing dataset");
    initialize(kmeans_config->max_points, metrics);

    // K-Means Lloyds alorithm  Step 1: initialize the centroids
    initialize_representatives(kmeans_config->num_clusters);

    // run the main loop
    struct kmeans_timing *timing = new_kmeans_timing();
    run(kmeans_config->max_iterations, timing);

    // finalize with the metrics
    finalize(metrics, timing);
    free(kmeans_config);
    return 0;
}

