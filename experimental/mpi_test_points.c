#include <stdbool.h>
#include <stdlib.h>
#include "mpi_test.h"
#include "../src/kmeans.h"
#ifdef __APPLE__
#include "/opt/openmpi/include/mpi.h"
#else
#include <mpi.h>
#endif

#define NUM_POINTS 6

int mpi_rank = 0;
int mpi_world_size = 0;
int num_points_subnode = 0; // number of points handled by this node
int num_points_total = 0;
bool is_root;
char node_label[20];
struct pointset main_dataset;
struct pointset node_dataset;

struct pointset allocate_pointset(int num_points)
{
    struct pointset new_pointset;// = (struct pointset )malloc(sizeof(struct pointset));
    new_pointset.x_coords = (double *)malloc(num_points * sizeof(double));
    new_pointset.y_coords = (double *)malloc(num_points * sizeof(double));
    new_pointset.cluster_ids = (int *)malloc(num_points * sizeof(int));
    new_pointset.num_points = num_points;
    return new_pointset;
}

void print_points(char *label)
{
    printf("%s [%s] Main points:\n", node_label, label);
    for (int i = 0; i < num_points_total; ++i) {
        printf("    %s [%s] main [%d] %.1f\n", node_label, label, i, main_dataset.x_coords[i]);
    }
    printf("%s [%s] Node points:\n", node_label, label);
    for (int i = 0; i < num_points_subnode; ++i) {
        printf("    %s [%s] node [%d] %.1f\n", node_label, label, i, node_dataset.x_coords[i]);
    }
    printf("\n");
}


int mpi_scatter_dataset()
{
    // if root node, then the dataset is already populated
    /* Distribute the work among all nodes. The data points itself will stay constant and
        not change for the duration of the algorithm. */
//    print_points(stderr, main_dataset, node_label);
    DEBUG("%sStarting scatter", node_label);
//    mpi_scatter_dataset_simple(dataset.x_coords, nodeset.x_coords);
//
    MPI_Scatter(main_dataset.x_coords, num_points_subnode, MPI_DOUBLE,
                node_dataset.x_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset.y_coords, num_points_subnode, MPI_DOUBLE,
                node_dataset.y_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(main_dataset.cluster_ids, num_points_subnode, MPI_INT,
                node_dataset.cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);
    DEBUG("%sScattered/Received %d points to/from other nodes. First x_coord is %.2f",
          node_label, num_points_subnode, node_dataset.x_coords[0]);
}

int mpi_gather_dataset()
{
    DEBUG("%sStarting Gather of subset with %d points:", node_label, num_points_subnode);
    print_points("PRE Gather ");
    fprintf(stderr, "GO!!!!!!!\n\n");
//    mpi_gather_dataset_simple(dataset.x_coords, nodeset.x_coords);
    MPI_Gather(node_dataset.x_coords, num_points_subnode, MPI_DOUBLE,
               main_dataset.x_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset.y_coords, num_points_subnode, MPI_DOUBLE,
               main_dataset.y_coords, num_points_subnode, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(node_dataset.cluster_ids, num_points_subnode, MPI_INT,
               main_dataset.cluster_ids, num_points_subnode, MPI_INT, 0, MPI_COMM_WORLD);

    fprintf(stderr, "DONE!!!!!!!\n\n");
    DEBUG("%sGathered/Sent %d points from other nodes. First x_coord is %.2f",
          node_label, num_points_subnode, node_dataset.x_coords[0]);
}

int load_dataset()
{
    for (int i = 0; i < NUM_POINTS; ++i) {
        main_dataset.x_coords[i] = (double)2 * i;
        main_dataset.y_coords[i] = (double)3 * i;
        main_dataset.cluster_ids[i] = -1;
    }
    main_dataset.num_points = NUM_POINTS;
    return NUM_POINTS;
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

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    DEBUG("Processor %s, rank %d out of %d processors\n",
          processor_name, mpi_rank, mpi_world_size);

    DEBUG("%sInitializing dataset", node_label);

    if (is_root) {
        // for root we actually load the dataset, for others we just return the empty one
        main_dataset = allocate_pointset(max_points);
        num_points_total = load_dataset();
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
    // Note: the root node also has a node_dataset since scatter will assign_clusters IT a subset
    //       of the total dataset along with all the other subnodes
    node_dataset = allocate_pointset(num_points_subnode);
    DEBUG("%sAllocated subnode dataset to %d points", node_label, num_points_subnode);
}

int assign_clusters()
{
    DEBUG("%sStarting assign_clusters with %d datapoints", node_label, num_points_subnode);
//    for (int i = 0; i < 20; ++i) {
//        sleep(1);
//    }
    mpi_scatter_dataset();
    DEBUG("%sReturned from scatter", node_label);

    DEBUG("%sAdding 10 to node dataset points ", node_label);
    for (int i = 0; i < num_points_subnode; ++i) {
        node_dataset.x_coords[i] += 10.0f;
    }

    print_points("pre-gather");

    mpi_gather_dataset();

    print_points("post-gather");

//    MPI_Barrier(MPI_COMM_WORLD);
    DEBUG("%sLeaving assign_clusters", node_label);


    // TODO barrier probably not necessary due to gather - get rid of it when all else is working
//    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

int main()
{
    initialize(MAX_POINTS);
    print_points("initial");
    assign_clusters();

    bool passed = true;
    if (is_root) {
        printf("\nDONE in ROOT:\n");
//        print_points("FINAL");

        for (int i = 0; i < num_points_total; ++i) {
            double expected = i * 2.0f + 10.f;
            double actual = main_dataset.x_coords[i];
            if (actual != expected) {
                fprintf(stderr, "FAILURE: point[%d] expected %.2f got %.2f\n", i, expected, actual);
                passed = false;
            }
            else {
                printf("SUCCESS: point[%d] expected %.2f got %.2f\n", i, expected, actual);
            }
        }
        fprintf(stderr, (passed ? "\nPASSED!\n" : "\n!!!! failed !!!!\n"));
    }

    // else the subnodes do not run the main loop but all mpi nodes must finalize
    MPI_Finalize();

    return 0;
}



