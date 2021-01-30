#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <unistd.h>
#include <string.h>
#include "kmeans.h"
#include "log.h"

extern struct log_config *log_config;
extern struct kmeans_config *kmeans_config;

/**
 * Initialize a new config to hold the run configuration set from the command line
 */

struct kmeans_config *new_kmeans_config()
{
    struct kmeans_config *new_config = malloc(sizeof(struct kmeans_config));
    new_config->in_file = NULL;
    new_config->out_file = NULL;
    new_config->test_file = NULL;
    new_config->metrics_file = NULL;
    new_config->label = "no-label";
    new_config->max_points = MAX_POINTS;
    new_config->num_clusters = NUM_CLUSTERS;
    new_config->max_iterations = MAX_ITERATIONS;
    new_config->num_processors = 1;
    return new_config;
}
struct log_config *new_log_config()
{
    struct log_config *new_config = malloc(sizeof(struct log_config));
    new_config->silent = false;
    new_config->quiet = false;
    new_config->verbose = false;
    new_config->debug = false;
    return new_config;
}

/**
 * Initialize a new metrics to hold the run performance metrics and settings
 */
struct kmeans_metrics *new_kmeans_metrics(struct kmeans_config *config)
{
    struct kmeans_metrics *new_metrics = malloc(sizeof(struct kmeans_metrics));
    new_metrics->label = config->label;
    new_metrics->max_iterations = config->max_iterations;
    new_metrics->num_clusters = config->num_clusters;
    new_metrics->total_seconds = 0;
    new_metrics->test_result = 0; // zero = no test performed
    return new_metrics;
}

void log_usage()
{
    fprintf(stderr, "Output logging options:\n");
    fprintf(stderr, "    -q --quiet fewer output messages\n");
    fprintf(stderr, "    -z --silent no output messages only the result for metrics\n");
    fprintf(stderr, "    -v --verbose lots of output messages including full matrices for debugging\n");
    fprintf(stderr, "    -d --debug debug messages (includes verbose)\n");
    fprintf(stderr, "    -h print this help and exit\n");
    fprintf(stderr, "\n");
}

void kmeans_usage()
{
    fprintf(stderr, "Usage: kmeans_<program> [options]\n");
    fprintf(stderr, "Options include:\n");
    fprintf(stderr, "    -f INFILE.CSV to read data points from a file (REQUIRED)\n");
    fprintf(stderr, "    -k --clusters NUM number of clusters to create (default: %d)\n", NUM_CLUSTERS);
    fprintf(stderr, "    -n --max-points NUM maximum number of points to read from the input file (default: %d)\n", MAX_POINTS);
    fprintf(stderr, "    -i --iterations NUM maximum number of iterations to loop over (default: %d)\n", MAX_ITERATIONS);
    fprintf(stderr, "    -o OUTFILE.CSV to write the resulting clustered points to a file (default is none)\n");
    fprintf(stderr, "    -t TEST.CSV compare result with TEST.CSV\n");
    fprintf(stderr, "    -m METRICS.CSV append metrics to this CSV file (creates it if it does not exist)\n");
    fprintf(stderr, "    -e --proper-distance measure Euclidean proper distance (slow) (defaults to faster square of distance)\n");
    log_usage();
    fprintf(stderr, "\n");
    exit(1);
}

char* valid_file(char opt, char *filename)
{
    if (access(filename, F_OK ) == -1 ) {
        fprintf(stderr, "Error: The option '%c' expects the name of an existing file (cannot find %s)\n", opt, filename);
        kmeans_usage();
    }
    return filename;
}

int valid_count(char opt, char *arg)
{
    int value = atoi(arg);
    if (value <= 0) {
        fprintf(stderr, "Error: The option '%c' expects a counting number (got %s)\n", opt, arg);
        kmeans_usage();
    }
    return value;
}

void validate_config(struct kmeans_config *config)
{
    if (config->in_file == NULL || strlen(config->in_file) == 0) {
        fprintf(stderr, "ERROR: You must at least provide an input file with -f\n");
        kmeans_usage();
    }

    const char* distance_type = config->proper_distance ? "proper distance" : "relative distance (d^2)";
    const char* loop_order_names[] = {"ijk", "ikj", "jki"};
    if (!log_config->quiet) {
        printf("Config:\n");
        printf("Input file        : %-10s\n", config->in_file);
        printf("Output file       : %-10s\n", config->out_file);
        printf("Test file         : %-10s\n", config->test_file);
        printf("Metrics file      : %-10s\n", config->metrics_file);
        printf("Clusters (k)      : %-10d\n", config->num_clusters);
        printf("Max Iterations    : %-10d\n", config->max_iterations);
        printf("Max Points        : %-10d\n", config->max_points);
        printf("Distance measure  : %s\n", distance_type);
        printf("\n");
    }
}

/**
 * Parse command line args and construct a config object
 */
void parse_kmeans_cli(int argc, char *argv[], struct kmeans_config *kmeans_config, struct log_config *log_config)
{
    int opt;
    struct option long_options[] = {
            {"input", required_argument, NULL, 'f'},
            {"output", required_argument, NULL, 'o'},
            {"test", required_argument, NULL, 't'},
            {"metrics", required_argument, NULL, 'm'},
            {"label", required_argument, NULL, 'l'},
            {"clusters", required_argument, NULL, 'k'},
            {"iterations", required_argument, NULL, 'i'},
            {"max-points", required_argument, NULL, 'n'},
            {"proper-distance", required_argument, NULL, 'e'},
            {"help", required_argument, NULL, 'h'},
            // log options
            {"silent", no_argument, NULL, 'z' },
            {"verbose", no_argument, NULL, 'v' },
            {"debug", no_argument, NULL, 'd' },
            {"quiet", no_argument, NULL, 'q'},
            {NULL, 0, NULL, 0}
    };
    int option_index = 0;
    while((opt = getopt_long(argc, argv, "-o:f:i:k:n:l:t:m:hqdvze" , long_options, &option_index)) != -1)
    {
//        fprintf(stderr, "FOUND OPT: [%c]\n", opt);
        switch(opt) {
            case 0:
                /* If this option set a flag, do nothing else: the flag is set */
                if (long_options[option_index].flag != 0)
                    break;
                // unexpected for now but maybe useful later
                printf("Unexpected option %s\n", long_options[option_index].name);
                kmeans_usage();
            case 'h':
                kmeans_usage();
                break;
            case 'i':
                kmeans_config->max_iterations = valid_count(optopt, optarg);
                break;
            case 'n':
                kmeans_config->max_points = valid_count(optopt, optarg);
                break;
            case 'k':
                kmeans_config->num_clusters = valid_count(optopt, optarg);
                break;
            case 'f':
                kmeans_config->in_file = valid_file('f', optarg);
                break;
            case 'o':
                kmeans_config->out_file = optarg;
                break;
            case 't':
                kmeans_config->test_file = optarg;
                break;
            case 'm':
                kmeans_config->metrics_file = optarg;
                break;
            case 'e':
                kmeans_config->proper_distance = true;
                break;
            case 'l':
                kmeans_config->label = optarg;
                break;

            // log options
            case 'd':
                log_config->debug = true;
                break;
            case 'z':
                log_config->silent = true;
                break;
            case 'v':
                log_config->verbose = true;
                break;
            case 'q':
                log_config->quiet = true;
                break;
            case ':':
                fprintf(stderr, "ERROR: Option %c needs a value\n", optopt);
                kmeans_usage();
                break;
            case '?':
                fprintf(stderr, "ERROR: Unknown option: %c\n", optopt);
                kmeans_usage();
                break;
            default:
                fprintf(stderr, "ERROR: Should never get here. opt=[%c]", opt);
        }
    }

    validate_config(kmeans_config);
}