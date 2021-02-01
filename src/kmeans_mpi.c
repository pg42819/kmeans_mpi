#include <float.h>
#include "kmeans.h"
#include "kmeans_support.h"
#include "log.h"
#ifdef __APPLE__
#include "/opt/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif
#include "mpi_log.h"
#include "kmeans.h"
#include "kmeans_support.h"
#include "log.h"

bool done = false;
int mpi_rank = 0;
int mpi_world_size = 0;
int num_points_node = 0; // number of points handled by this node
int num_points_total = 0;
bool is_root;
char node_label[20];
static char* headers[3];
static int dimensions;
struct kmeans_config *kmeans_config;
enum log_level_t log_level;

struct pointset main_dataset;
struct pointset node_dataset;
struct pointset node_centroids;

void mpi_log_centroids(int level, char *label)
{
    if (log_level < level) return;
    mpi_log(level, "Centroids: %s", label);
    char full_label[256];
    sprintf(full_label, "%s : %s", node_label, label);
    node_color();
    print_centroids(stdout, &node_centroids, node_label);
    reset_color();
}

void mpi_log_dataset(int level, struct pointset *pointset, char *label)
{
    if (log_level < level) return;
    mpi_log(level, "Dataset: %s", label);
    char full_label[256];
    sprintf(full_label, "%s%s ", node_label, label);
    node_color();
    print_points(stdout, pointset, full_label);
    reset_color();
}

int load_dataset(struct pointset *dataset)
{
    char *csv_file_name = valid_file('f', kmeans_config->in_file);
    int num_points = read_csv_file(csv_file_name, dataset, kmeans_config->max_points, headers, &dimensions);
    DEBUG("Loaded %d points from the dataset file at %s", num_points, csv_file_name);
    return num_points;
}

void initialize(int max_points, struct kmeans_metrics *metrics)
{
    // MPI PREP
    MPI_Status status;
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    is_root = mpi_rank == 0;
    if (is_root) {
        sprintf(node_label, "Root %d: ", mpi_rank);
    }
    else {
        sprintf(node_label, "Node %d: ", mpi_rank);
    }

    if (IS_DEBUG) {
        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        mpi_log(debug, "Processor %s, rank %d out of %d processors\n",
                processor_name, mpi_rank, mpi_world_size);
    }

    mpi_log(debug, "Initializing dataset");
    if (is_root) {
        metrics->num_processors=mpi_world_size;
        // for root we actually load the dataset, for others we just return the empty one
        allocate_pointset_points(&main_dataset, max_points);
        mpi_log(debug, "Allocated %d point space", max_points);
        num_points_total = load_dataset(&main_dataset);
        mpi_log(info, "Loaded main dataset with %d points (confirmation: %d)", num_points_total, main_dataset.num_points);

        // number of points managed by each subnode is the total number divided by processes
        // plus 1 in case of remainder (number of points is not is not an even multiple of processors)
        num_points_node = num_points_total / mpi_world_size;
        if (num_points_total % mpi_world_size > 0) {
            num_points_node += 1;
            mpi_log(debug, "Calculated subnode dataset size: %d / %d (+ 1?) = %d",
                    num_points_total, mpi_world_size, num_points_node);
        }
    }

    // broadcast from root to sub nodes, or receive to make the calculation from root
    MPI_Bcast(&num_points_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_log(debug, "Got %d as num_points_subnode after broadcast", num_points_node);

    // Create a subnode dataset on each subnode, independent of the main dataset
    // Note: the root node also has a node_dataset since scatter will assign_clusters IT a subset
    //       of the total dataset along with all the other subnodes
    allocate_pointset_points(&node_dataset, num_points_node);
    mpi_log(debug, "Allocated subnode dataset to %d points", num_points_node);
//    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Distribute dataset as subsets to other nodes nodes - including a subset to the root node
 */
void mpi_scatter_dataset()
{
    mpi_log(debug, "Starting scatter of %d points", num_points_node);
    MPI_Scatter(main_dataset.x_coords, num_points_node, MPI_DOUBLE,
                node_dataset.x_coords, num_points_node, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset.y_coords, num_points_node, MPI_DOUBLE,
                node_dataset.y_coords, num_points_node, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset.cluster_ids, num_points_node, MPI_INT,
                node_dataset.cluster_ids, num_points_node, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_log(debug, "Scattered/Received %d points to/from other nodes. First x_coord is %.2f",
            num_points_node, node_dataset.x_coords[0]);
    mpi_log_dataset(debug, &node_dataset, "After Scatter ");
}

/**
 * Gather back subsets of data to the root node from the various processes (including the root)
 */
void mpi_gather_dataset()
{
    mpi_log(debug, "Starting Gather of subset with %d points:", num_points_node);
    MPI_Gather(node_dataset.x_coords, num_points_node, MPI_DOUBLE,
               main_dataset.x_coords, num_points_node, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset.y_coords, num_points_node, MPI_DOUBLE,
               main_dataset.y_coords, num_points_node, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset.cluster_ids, num_points_node, MPI_INT,
               main_dataset.cluster_ids, num_points_node, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_log(debug, "Done Gathering");
    mpi_log_dataset(debug, &main_dataset, "After Gather");
}

/**
 * Broadcast the centroids values to all nodes
 */
void mpi_broadcast_centroids()
{
    mpi_log(debug, "Broadcasting centroids");
    MPI_Bcast(node_centroids.x_coords, node_centroids.num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(node_centroids.y_coords, node_centroids.num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&node_centroids.num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_log(debug, "DONE Broadcasting centroids");
    mpi_log_centroids(trace, "after broadcast");
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
    mpi_log(trace, "Starting assign_clusters with %d datapoints", node_dataset.num_points);
    mpi_scatter_dataset();

    mpi_log(trace, "Calling simple_assign_clusters with node dataset at %d of size %d", &node_dataset, node_dataset.num_points);
    int total_reassignments = 0;
    int node_reassignments = simple_assign_clusters(&node_dataset, &node_centroids);

    MPI_Reduce(&node_reassignments, &total_reassignments, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    mpi_log(trace, "Returned from simple_assign_clusters with %d node, %d total cluster reassignments",
            node_reassignments, total_reassignments);
    mpi_gather_dataset();
    mpi_log(trace, "Leaving assign_clusters with %d changes", total_reassignments);
    return total_reassignments;
}

/**
 * Calculates new centroids for the clusters of the given dataset by finding the
 * mean x and y coordinates of the current members of the cluster for each cluster.
 *
 * @param dataset set of all points with current cluster assignments
 * @param centroids array to hold the centroids - already allocated
 */
void calculate_centroids()
{
    mpi_log(trace, "Starting calculate_centroids");
    if (is_root) {
        // calculate on the root node only for now then broadcast
        mpi_log_centroids(trace, "pre-calc-centroids");
        mpi_log_dataset(trace, &main_dataset, "pre-calc-centroids");

        simple_calculate_centroids(&main_dataset, &node_centroids);
        mpi_log_centroids(trace, "post-calc-centroids");
    }

    mpi_broadcast_centroids();
    mpi_log(trace, "Leaving calculate_centroids");
}

void initialize_representatives(int num_clusters)
{
    // all nodes need a centroids point set
    allocate_pointset_points(&node_centroids, num_clusters);

    if (is_root) {
        mpi_log(debug, "Initialize centroids in root node (%d)", mpi_rank);
        initialize_centroids(&main_dataset, &node_centroids);
    }
    mpi_broadcast_centroids();

//    if (is_root) {
//        debug_setup(&main_dataset, &node_centroids);// print dbg info
//    }
}


bool is_done(int changes, int iterations, int max_iterations)
{
    // only root completes the loop
    if (is_root) {
        if (changes == 0 || iterations >= max_iterations) {
            mpi_log(info, "ROOT is done with %d changes after %d iterations", changes, iterations);
            done = true;
        }
    }
    mpi_log(debug, "Broadcasting done");
    MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    mpi_log(debug, "AFter broadcast done: %d", done);
    return done;
}

void main_loop(int max_iterations, struct kmeans_metrics *metrics) {

    // we deliberately skip the centroid initialization phase in calculating the
    // total time as it is constant and never optimized
    double start_time = omp_get_wtime();
    int cluster_changes = MAX_POINTS; // start at a max then work down to zero chagnes
    int iterations = 0;

    while (!is_done(cluster_changes, iterations, max_iterations)) {
        mpi_log(debug, "Starting iteration %d. %d change in last iteration", iterations, cluster_changes);
//        DEBUG("Starting iteration %d. %d change in last iteration", iterations, cluster_changes);
        // K-Means Algo Step 2: assign_clusters every point to a cluster (closest centroid)
        double start_iteration;
        double start_assignment;
        double assignment_seconds;
        double start_centroids;

        if (is_root) {
            start_iteration = omp_get_wtime();
            start_assignment = start_iteration;
        }
        mpi_log(debug, "calling assign_clusters");
        cluster_changes = assign_clusters();
        mpi_log(debug, "returned from assign_clusters");

        if (is_root) {
            assignment_seconds = omp_get_wtime() - start_assignment;
            metrics->assignment_seconds += assignment_seconds;
            start_centroids = omp_get_wtime();
        }

        // K-Means Algo Step 3: calculate new centroids: one at the center of each cluster
        mpi_log(debug, "calling calculate_centroids");
        calculate_centroids();
        mpi_log(debug, "returned from calculate_centroids");

//        MPI_Barrier(MPI_COMM_WORLD);
        if (is_root) {
            mpi_log(debug, "Calculating time and setting metrics on root");
            double centroids_seconds = omp_get_wtime() - start_centroids;
            metrics->centroids_seconds += centroids_seconds;

            // potentially costly calculation may skew stats, hence only in ifdef
            double iteration_seconds = omp_get_wtime() - start_iteration;
            if (iteration_seconds > metrics->max_iteration_seconds) {
                metrics->max_iteration_seconds = iteration_seconds;
            }
        }
        iterations++;
    }

    if (is_root) {
        metrics->total_seconds = omp_get_wtime() - start_time;
        metrics->used_iterations = iterations;
    }
    mpi_log(info, "Ended after %d iterations with %d changed clusters\n", iterations, cluster_changes);
}

void run(int max_iterations, struct kmeans_metrics *metrics)
{
    mpi_log(debug, "Running main loop");
    main_loop(max_iterations, metrics);
    mpi_log(debug, "Main loop completed");
}

void main_finalize(struct pointset *dataset, struct kmeans_metrics *metrics)
{
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
        print_points(stdout, &main_dataset, "Final ");
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

void finalize(struct kmeans_metrics *metrics)
{
//    MPI_Barrier(MPI_COMM_WORLD);
    mpi_log(debug, "Finalizing");
    if (is_root) {
        metrics->num_points = num_points_total;
        main_finalize(&main_dataset, metrics);
    }
    // else the subnodes do not run the main loop but all mpi nodes must finalize
    MPI_Finalize();
}

void debug_setup(struct pointset *dataset, struct pointset *centroids)
{
    if (IS_DEBUG) {
        printf("\nDatabase Setup:\n");
        print_headers(stdout, headers, dimensions);
        print_points(stdout, dataset, "Setup ");
        printf("\nCentroids Setup:\n");
        print_centroids(stdout, centroids, "Setup ");
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
    run(kmeans_config->max_iterations, metrics);

    // finalize with the metrics
    finalize(metrics);
    free(kmeans_config);
    return 0;
}


