#include <float.h>
#include <math.h>
#include "kmeans.h"
#include "kmeans_extern.h"
#include "log.h"
#ifdef __APPLE__
#include "/opt/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif

extern void initialize_centroids(struct pointset* dataset, struct pointset *centroids);
extern int load_dataset(struct pointset* dataset);
extern void main_loop(int max_iterations, struct kmeans_metrics *metrics);
extern void main_finalize(struct pointset *dataset, struct kmeans_metrics *metrics);
extern void debug_setup(struct pointset *dataset, struct pointset *centroids);

int mpi_rank = 0;
int mpi_world_size = 0;
int num_points_subnode = 0; // number of points handled by this node
int num_points_total = 0;
bool is_root;
char node_label[20];
struct pointset *datapoints;
struct pointset *nodepoints;
struct pointset *node_centroids;

// TODO move simple_calculate to kmeans-simple-support
void simple_calculate_centroids(struct pointset* dataset, struct pointset *centroids)
{
    int num_points = dataset->num_points;
    int num_clusters = centroids->num_points;
    double sum_of_x_per_cluster[num_clusters];
    double sum_of_y_per_cluster[num_clusters];
    int num_points_in_cluster[num_clusters];
    for (int k = 0; k < num_clusters; ++k) {
        sum_of_x_per_cluster[k] = 0.0;
        sum_of_y_per_cluster[k] = 0.0;
        num_points_in_cluster[k] = 0;
    }

    // loop over all points in the database and sum up
    // the x coords of clusters to which each belongs
    for (int n = 0; n < num_points; ++n) {
        int k = dataset->cluster_ids[n];
        sum_of_x_per_cluster[k] += dataset->x_coords[n];
        sum_of_y_per_cluster[k] += dataset->y_coords[n];
        // count the points in the cluster to get a mean later
        num_points_in_cluster[k]++;
    }

    // the new centroids are at the mean x and y coords of the clusters
    for (int k = 0; k < num_clusters; ++k) {
        int cluster_size = num_points_in_cluster[k];
        VERBOSE("%sCluster %d has %d points", node_label, k, cluster_size);
        // ignore empty clusters (otherwise div by zero!)
        if (cluster_size > 0) {
            // mean x, mean y => new centroid
            double new_centroid_x = sum_of_x_per_cluster[k] / cluster_size;
            double new_centroid_y = sum_of_y_per_cluster[k] / cluster_size;
            set_point(centroids, k, new_centroid_x, new_centroid_y, IGNORE_CLUSTER_ID);
        }
    }
}

int mpi_scatter_dataset(struct pointset *main_dataset, struct pointset *node_dataset)
{
    // if root node, then the dataset is already populated
    /* Distribute the work among all nodes. The data points itself will stay constant and
        not change for the duration of the algorithm. */
    print_points(stderr, main_dataset, node_label);
    DEBUG("%sStarting scatter", node_label);
    MPI_Scatter(main_dataset->x_coords, num_points_subnode, MPI_DOUBLE,
                node_dataset->x_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset->y_coords, num_points_subnode, MPI_DOUBLE,
                node_dataset->y_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset->cluster_ids, num_points_subnode, MPI_INT,
               node_dataset->cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG("%sScattered/Received %d points to/from other nodes. First x_coord is %.2f",
          node_label, num_points_subnode, node_dataset->x_coords[0]);
}

int mpi_gather_dataset(struct pointset *main_dataset, struct pointset *node_dataset)
{
    // if root node, then the dataset is already populated
    /* Distribute the work among all nodes. The data points itself will stay constant and
        not change for the duration of the algorithm. */
    DEBUG("%sStarting Gather of subset with %d points:", node_label, num_points_subnode);
    print_points(stdout, node_dataset, node_label);
    fprintf(stderr, "GO!!!!!!!\n\n");
    MPI_Gather(node_dataset->x_coords, num_points_subnode, MPI_DOUBLE,
               main_dataset->x_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset->y_coords, num_points_subnode, MPI_DOUBLE,
               main_dataset->y_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset->cluster_ids, num_points_subnode, MPI_INT,
               main_dataset->cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);

    fprintf(stderr, "DONE!!!!!!!\n\n");
    DEBUG("%sGathered/Sent %d points from other nodes. First x_coord is %.2f",
          node_label, num_points_subnode, node_dataset->x_coords[0]);
}

void mpi_broadcast_centroids(struct pointset *centroids) {
    DEBUG("Node %d : Broadcasting centroids", mpi_rank);
    MPI_Bcast(centroids->x_coords, centroids->num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(centroids->y_coords, centroids->num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&centroids->num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG("Node %d : DONE Broadcasting centroids", mpi_rank);
    node_centroids = centroids;
    print_centroids(stdout, node_centroids, node_label);
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

    if (log_config->debug) {
        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        DEBUG("Processor %s, rank %d out of %d processors\n",
              processor_name, mpi_rank, mpi_world_size);
    }

    DEBUG("%sInitializing dataset", node_label);
    if (is_root) {
        // for root we actually load the dataset, for others we just return the empty one
        datapoints = allocate_pointset(max_points);
        num_points_total = load_dataset(datapoints);
        DEBUG("%sLoaded main dataset with %d points", node_label, num_points_total);

        // number of points managed by each subnode is the total number divided by processes
        // plus 1 in case of remainder (number of points is not is not an even multiple of processors)
        num_points_subnode = num_points_total / mpi_world_size;
        if (num_points_total % mpi_world_size > 0) {
            num_points_subnode += 1;
            DEBUG("%sCalculated subnode dataset size: %d / %d (+ 1?) = %d",
                  node_label, num_points_total, mpi_world_size, num_points_subnode);
        }
    }

    // broadcast from root to sub nodes, or receive to make the calculation from root
    MPI_Bcast(&num_points_subnode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG("%sGot %d as num_points_subnode after broadcast", node_label, num_points_subnode);

    // Create a subnode dataset on each subnode, independent of the main dataset
    // Note: the root node also has a node_dataset since scatter will assign IT a subset
    //       of the total dataset along with all the other subnodes
    nodepoints = allocate_pointset(num_points_subnode);
    DEBUG("%sAllocated subnode dataset to %d points", node_label, num_points_subnode);
}

/**
 * Assigns each point in the dataset to a cluster based on the distance from that cluster.
 *
 * The return value indicates how many points were assigned to a _different_ cluster
 * in this assignment process: this indicates how close the algorithm is to completion.
 * When the return value is zero, no points changed cluster so the clustering is complete.
 *
 * @param dataset set of all points with current cluster assignments
 * @param centroids set of current centroids
 * @return the number of points for which the cluster assignment was changed
 */
int simple_assign_clusters(struct pointset* dataset, struct pointset *centroids)
{
    DEBUG("%sStarting simple assignment", node_label);
    int cluster_changes = 0;

    int num_points = dataset->num_points;
    int num_clusters = centroids->num_points;
    for (int n = 0; n < num_points; ++n) {
        double min_distance = DBL_MAX; // init the min distance to a big number
        int closest_cluster = -1;
        for (int k = 0; k < num_clusters; ++k) {
            // calc the distance passing pointers to points since the distance does not modify them
            double distance_from_centroid = point_distance(dataset, n, centroids, k);
            if (distance_from_centroid < min_distance) {
                min_distance = distance_from_centroid;
                closest_cluster = k;
                DEBUG("%sClosest cluster to point %d is %d",node_label, n, k);
            }
        }
        // if the point was not already in the closest cluster, move it there and count changes
        if (dataset->cluster_ids[n] != closest_cluster) {
            dataset->cluster_ids[n] = closest_cluster;
            cluster_changes++;
            trace_assignment(dataset, n, centroids, closest_cluster, min_distance);
        }
    }
    DEBUG("%sLeaving simple assignment with %d cluster changes", node_label, cluster_changes);
    return cluster_changes;
}

/**
 * Assigns each point in the dataset to a cluster based on the distance from that cluster.
 *
 * The return value indicates how many points were assigned to a _different_ cluster
 * in this assignment process: this indicates how close the algorithm is to completion.
 * When the return value is zero, no points changed cluster so the clustering is complete.
 *
 * @param dataset set of all points with current cluster assignments
 * @param centroids set of current centroids
 * @return the number of points for which the cluster assignment was changed
 */
int assign_clusters(struct pointset* dataset, struct pointset *centroids)
{
    DEBUG("Node %d : Starting assign_clusters with %d datapoints", mpi_rank, dataset->num_points);
//    mpi_scatter_dataset(dataset);
    int reassignments = simple_assign_clusters(dataset, centroids);
//    mpi_gather_dataset(dataset);
    DEBUG("Node %d : Leaving assign_clusters after %d cluster reassignments", mpi_rank, reassignments);

    // TODO barrier probably not necessary due to gather - get rid of it when all else is working
    MPI_Barrier(MPI_COMM_WORLD);
    return reassignments;
}
int assign()
{
    DEBUG("%sStarting assign_clusters with %d datapoints", node_label, nodepoints->num_points);
    DEBUG("%sCalling scatter", node_label);
    for (int i = 0; i < 20; ++i) {
        sleep(1);
    }
    mpi_scatter_dataset(datapoints, nodepoints);
    DEBUG("%sReturned from scatter", node_label);
    int reassignments = simple_assign_clusters(nodepoints, node_centroids);
    mpi_gather_dataset(datapoints, nodepoints);
    MPI_Barrier(MPI_COMM_WORLD);
    DEBUG("%sLeaving assign_clusters after %d cluster reassignments", node_label, reassignments);
    if (log_config->debug) {
        print_points(stdout, datapoints, node_label);
    }

    // TODO barrier probably not necessary due to gather - get rid of it when all else is working
//    MPI_Barrier(MPI_COMM_WORLD);
    return reassignments;
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
    if (is_root) {
        // calculate on the root node only for now then broadcast
        if (log_config -> debug) {
            DEBUG("Root: starting calculate centroids starting with:")
            print_centroids(stdout, node_centroids, node_label);
            DEBUG("ROOT: and data:");
            print_points(stdout, datapoints, node_label);
        }

        simple_calculate_centroids(datapoints, node_centroids);
        // TODO change verbose to trace
        if (log_config->verbose) {
            printf("New centroids calculated:\n");
            print_centroids(stdout, node_centroids, node_label);
        }
    }
    mpi_broadcast_centroids(node_centroids);
}

void initialize_representatives(int num_clusters)
{
    // all nodes need a centroids point set
    node_centroids = allocate_pointset(num_clusters);

    if (is_root) {
        DEBUG("Initialize centroids in root node (%d)", mpi_rank);
        // TODO move initialize_centroids to support
        initialize_centroids(datapoints, node_centroids);
    }
    mpi_broadcast_centroids(node_centroids);

    if (is_root) {
        debug_setup(datapoints, node_centroids);// print debug info
    }
}

int populate_dataset(struct pointset *dataset)
{
    // if root node, then the dataset is already populated
    /* Distribute the work among all nodes. The data points itself will stay constant and
        not change for the duration of the algorithm. */
//    MPI_Scatter(dataset->x_coords, num_points_subnode, MPI_DOUBLE,
//                dataset->x_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Scatter(dataset->y_coords, num_points_subnode, MPI_DOUBLE,
//                dataset->y_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    DEBUG("Node %d : Scattered (received if %d>0) %d points to other nodes. First x_coord is %.2f",
//          mpi_rank, mpi_rank, num_points_subnode, dataset->x_coords[0]);
}

void run(int max_iterations, struct kmeans_metrics *metrics)
{
    if (is_root) {
        DEBUG("Root: Running main loop")
        main_loop(max_iterations, metrics);
        DEBUG("Root: Main loop completed")
    }
    // else the subnodes do not run the main loop
}

void finalize(struct kmeans_metrics *metrics)
{
    metrics->num_points = num_points_total;
    MPI_Barrier(MPI_COMM_WORLD);

    DEBUG("%sFinalizing", node_label);
    if (is_root) {
        main_finalize(datapoints, metrics);
    }
    // else the subnodes do not run the main loop but all mpi nodes must finalize
    MPI_Finalize();
}

