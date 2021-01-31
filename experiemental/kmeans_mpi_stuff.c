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
int num_points_subnode = 0; // number of points handled by this node
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

struct pointset allocate_pointset2(int num_points)
{
    struct pointset new_pointset;// = (struct pointset )malloc(sizeof(struct pointset));
    new_pointset.x_coords = (double *)malloc(num_points * sizeof(double));
    new_pointset.y_coords = (double *)malloc(num_points * sizeof(double));
    new_pointset.cluster_ids = (int *)malloc(num_points * sizeof(int));
    new_pointset.num_points = num_points;
    return new_pointset;
}

int load_dataset(struct pointset *dataset) {
    char *csv_file_name = valid_file('f', kmeans_config->in_file);
    int num_points = read_csv_file(csv_file_name, dataset, kmeans_config->max_points, headers, &dimensions);
    DEBUG("Loaded %d points from the dataset file at %s", num_points, csv_file_name);
    return num_points;
}


void initialize(int max_points)
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
        // for root we actually load the dataset, for others we just return the empty one
        main_dataset = allocate_pointset2(max_points);
        mpi_log(debug, "Allocated %d point space", max_points);
        num_points_total = load_dataset(&main_dataset);
        mpi_log(debug, "Loaded main dataset with %d points (confirmation: %d)", num_points_total, main_dataset.num_points);

        // number of points managed by each subnode is the total number divided by processes
        // plus 1 in case of remainder (number of points is not is not an even multiple of processors)
        num_points_subnode = num_points_total / mpi_world_size;
        if (num_points_total % mpi_world_size > 0) {
            num_points_subnode += 1;
            mpi_log(debug, "Calculated subnode dataset size: %d / %d (+ 1?) = %d",
                    num_points_total, mpi_world_size, num_points_subnode);
        }
    }

    // broadcast from root to sub nodes, or receive to make the calculation from root
    MPI_Bcast(&num_points_subnode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_log(debug, "Got %d as num_points_subnode after broadcast", num_points_subnode);

    // Create a subnode dataset on each subnode, independent of the main dataset
    // Note: the root node also has a node_dataset since scatter will assign_clusters IT a subset
    //       of the total dataset along with all the other subnodes
    node_dataset = allocate_pointset2(num_points_subnode);
    mpi_log(debug, "Allocated subnode dataset to %d points", num_points_subnode);
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_scatter_dataset()
{
    // if root node, then the dataset is already populated
    /* Distribute the work among all nodes. The data points itself will stay constant and
        not change for the duration of the algorithm. */
    mpi_log(debug, "Starting scatter of %d points", num_points_subnode);
    MPI_Scatter(main_dataset.x_coords, num_points_subnode, MPI_DOUBLE,
                node_dataset.x_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset.y_coords, num_points_subnode, MPI_DOUBLE,
                node_dataset.y_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset.cluster_ids, num_points_subnode, MPI_INT,
               node_dataset.cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_log(debug, "Scattered/Received %d points to/from other nodes. First x_coord is %.2f",
          num_points_subnode, node_dataset.x_coords[0]);
    mpi_log_dataset(debug, &node_dataset, "After Scatter ");
}

void mpi_gather_dataset()
{
    mpi_log(debug, "Starting Gather of subset with %d points:", num_points_subnode);
//    dbg_points(&node_dataset, "PRE gather ");//_node_label);
//    fprintf(stderr, "%sGO!!!!!!!\n\n", node_label);
    MPI_Gather(node_dataset.x_coords, num_points_subnode, MPI_DOUBLE,
               main_dataset.x_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset.y_coords, num_points_subnode, MPI_DOUBLE,
               main_dataset.y_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset.cluster_ids, num_points_subnode, MPI_INT,
               main_dataset.cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);
//    fprintf(stderr, "%sDONE!!!!!!!\n\n", node_label);
    mpi_log(debug, "Done Gathering");
    mpi_log_dataset(debug, &main_dataset, "After Gather");
//    mpi_log_dataset()
//    mpi_log(debug, "Gathered/Sent %d points from other nodes. First x_coord is %.2f",
//            num_points_subnode, main_dataset.x_coords[0]);
}

void mpi_broadcast_centroids()
{
    mpi_log(debug, "Broadcasting centroids");
    MPI_Bcast(node_centroids.x_coords, node_centroids.num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(node_centroids.y_coords, node_centroids.num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&node_centroids.num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_log(debug, "DONE Broadcasting centroids");
    mpi_log_centroids(trace, "after broadcast");
}

int lowlevel_assign_clusters(int num_points, double *x_coords, double *y_coords,
                             int num_clusters, double *centroid_x_coords, double *centroid_y_coords,
                             int *cluster_ids)
{
//    TRACE("Starting low level assignment");
    mpi_log(trace, "Starting low level assignment");
    int cluster_changes = 0;

    for (int n = 0; n < num_points; ++n) {
        double point_x = x_coords[n];
        double point_y = y_coords[n];
        double min_distance = DBL_MAX; // init the min distance to a big number
        int closest_cluster = -1;
        for (int k = 0; k < num_clusters; ++k) {
            double centroid_x = centroid_x_coords[k];
            double centroid_y = centroid_y_coords[k];
            // calc the distance passing pointers to points since the distance does not modify them
            double distance_from_centroid = euclidean_distance(centroid_x, centroid_y, point_x, point_y);
            if (distance_from_centroid < min_distance) {
                min_distance = distance_from_centroid;
                closest_cluster = k;
//                TRACE("Closest cluster to point %d is %d", n, k);
                mpi_log(trace, "Closest cluster to point %d is %d", n, k);
            }
        }
        // if the point was not already in the closest cluster, move it there and count changes
        if (cluster_ids[n] != closest_cluster) {
            cluster_ids[n] = closest_cluster;
            cluster_changes++;
            mpi_log(trace, "Assigning (%.2f,%.2f) to cluster %d with centroid (%.2f,%.2f)",
                           point_x, point_y, closest_cluster,
                           centroid_x_coords[closest_cluster], centroid_y_coords[closest_cluster]);
//            TRACE("Assigning (%s) to cluster %d with centroid (%s) d = %f\n",
//                  p_to_s(dataset, n),  closest_cluster, p_to_s(centroids, closest_cluster), min_distance);
        }
    }
    TRACE("Leaving simple assignment with %d cluster changes", cluster_changes);
    return cluster_changes;
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

//  for (int i = 0; i < 20; ++i) {
//     fprintf(stderr, "waiting while you attach a debugger");
//      sleep(1);
//  }
//    MPI_Scatter(main_dataset.cluster_ids, num_points_subnode, MPI_INT,
//                node_dataset.cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);
//
    mpi_scatter_dataset();
    MPI_Barrier(MPI_COMM_WORLD);
    mpi_log(trace, "Calling simple_assign_clusters with node dataset at %d of size %d", &node_dataset, node_dataset.num_points);
//    int reassignments = simple_assign_clusters(&node_dataset, &node_centroids);
    int total_reassignments = 0;
//    int node_reassignments = lowlevel_assign_clusters(
//            node_dataset.num_points, node_dataset.x_coords, node_dataset.y_coords,
//            node_centroids.num_points, node_centroids.x_coords, node_centroids.y_coords,
//            node_dataset.cluster_ids);
    int node_reassignments = simple_assign_clusters(&node_dataset, &node_centroids);

    MPI_Reduce(&node_reassignments, &total_reassignments, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    mpi_log(trace, "Returned from simple_assign_clusters with %d node, %d total cluster reassignments",
            node_reassignments, total_reassignments);
//    MPI_Gather(node_dataset.cluster_ids, num_points_subnode, MPI_INT,
//               main_dataset.cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);
    mpi_gather_dataset();
    // TODO barrier probably not necessary due to gather - get rid of it when all else is working
    MPI_Barrier(MPI_COMM_WORLD);
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
    node_centroids = allocate_pointset(num_clusters);

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

        MPI_Barrier(MPI_COMM_WORLD);
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

    DEBUG("Initializing dataset");
    initialize(kmeans_config->max_points);

    // K-Means Lloyds alorithm  Step 1: initialize the centroids
    initialize_representatives(kmeans_config->num_clusters);

    // set up a metrics struct to hold timing and other info for comparison
    struct kmeans_metrics *metrics = new_kmeans_metrics(kmeans_config);

    // run the main loop
    run(kmeans_config->max_iterations, metrics);

    // finalize with the metrics
    finalize(metrics);
    free(kmeans_config);
    return 0;
}


