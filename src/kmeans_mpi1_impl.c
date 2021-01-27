#include <float.h>
#include <math.h>
#include "kmeans.h"
#include "kmeans_extern.h"
#include <mpi.h>

//int main( int argc, char *argv[])
//{
//    int rank, msg;
//    MPI_Status status;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
//    /* Process 0 sends and Process 1 receives */
//    if (rank == 0) {
//        msg = 123456;
//        MPI_Send( &msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
//    }
//    else if (rank == 1) {
//        MPI_Recv( &msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
//        printf( "Received %d\n", msg);
//    }
//
//    MPI_Finalize();
//    return 0;
//}

/**
 * Assigns each point in the dataset to a cluster based on the distance from that cluster.
 *
 * The return value indicates how many points were assigned to a _different_ cluster
 * in this assignment process: this indicates how close the algorithm is to completion.
 * When the return value is zero, no points changed cluster so the clustering is complete.
 *
 * @param dataset set of all points with current cluster assignments
 * @param num_points number of points in the dataset
 * @param centroids array that holds the current centroids
 * @param num_clusters number of clusters - hence size of the centroids array
 * @return the number of points for which the cluster assignment was changed
 */
int assign_clusters(struct point* dataset, int num_points, struct point *centroids, int num_clusters)
{
    DEBUG("\nStarting assignment phase:\n");
    int cluster_changes = 0;
    for (int n = 0; n < num_points; ++n) {
        double min_distance = DBL_MAX; // init the min distance to a big number
        int closest_cluster = -1;
        for (int k = 0; k < num_clusters; ++k) {
            // calc the distance passing pointers to points since the distance does not modify them
            double distance_from_centroid = euclidean_distance(&dataset[n], &centroids[k]);
            if (distance_from_centroid < min_distance) {
                min_distance = distance_from_centroid;
                closest_cluster = k;
            }
        }
        // if the point was not already in the closest cluster, move it there and count changes
        if (dataset[n].cluster != closest_cluster) {
            dataset[n].cluster = closest_cluster;
            cluster_changes++;
            debug_assignment(&dataset[n], closest_cluster, &centroids[closest_cluster], min_distance);
        }
    }
    return cluster_changes;
}

/**
 * Calculates new centroids for the clusters of the given dataset by finding the
 * mean x and y coordinates of the current members of the cluster for each cluster.
 *
 * The centroids are set in the array passed in, which is expected to be pre-allocated
 * and contain the previous centroids: these are overwritten by the new values.
 *
 * @param dataset set of all points with current cluster assigments
 * @param num_points number of points in the dataset
 * @param centroids array to hold the centroids - already allocated
 * @param num_clusters number of clusters - hence size of the centroids array
 */
void calculate_centroids(struct point* dataset, int num_points, struct point *centroids, int num_clusters)
{
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
        // use pointer to struct to avoid creating unnecessary copy in memory
        struct point *p = &dataset[n];
        int k = p->cluster;
        sum_of_x_per_cluster[k] += p->x;
        sum_of_y_per_cluster[k] += p->y;
        // count the points in the cluster to get a mean later
        num_points_in_cluster[k]++;
    }

    // the new centroids are at the mean x and y coords of the clusters
    for (int k = 0; k < num_clusters; ++k) {
        struct point new_centroid;
        // mean x, mean y => new centroid
        new_centroid.x = sum_of_x_per_cluster[k] / num_points_in_cluster[k];
        new_centroid.y = sum_of_y_per_cluster[k] / num_points_in_cluster[k];
        centroids[k] = new_centroid;
    }
}


