#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <unistd.h>
#include "log.h"

extern struct log_config *log_config;
/**
 * Initialize a new log_config to hold the run configuration set from the command line
 */
struct log_config *new_log_config()
{
    struct log_config *new_config = malloc(sizeof(struct log_config));
    new_config->silent = false;
    new_config->quiet = false;
    new_config->verbose = false;
    new_config->debug = false;
    return new_config;
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
    exit(1);
}

/**
 * Parse command line args and construct a config object
 * Caller must free at the end of use
 */
struct log_config *parse_log_cli(int argc, char *argv[], char *all_options)
{
    int opt;
    struct log_config *config = new_log_config();

    struct option long_options[] = {
            {"silent", no_argument, NULL, 'z' },
            {"verbose", no_argument, NULL, 'v' },
            {"debug", no_argument, NULL, 'd' },
            {"quiet", no_argument, NULL, 'q'},
            {NULL, 0, NULL, 0}
    };
    int option_index = 0;

    while((opt = getopt_long(argc, argv, all_options, long_options, &option_index)) != -1)
    {
        switch(opt) {
            case 0:
                /* If this option set a flag, do nothing else: the flag is set */
                if (long_options[option_index].flag != 0)
                    break;
                // unexpected for now but maybe useful later
                printf("Unexpected option %s\n", long_options[option_index].name);
                log_usage();
            case 'h':
                log_usage();
                break;
            case 'd':
                config->debug = true;
                break;
            case 'z':
                config->silent = true;
                break;
            case 'v':
                config->verbose = true;
                break;
            case 'q':
                config->quiet = true;
                break;
            case '?':
                // may be used in other parser
                break;
//            default:
                // ignore might be handled elsewhere
//                fprintf(stderr, "ERROR: Should never get here. opt=[%c]", opt);
        }
    }

    if (config->silent) {
        config->quiet = true; // silent implies quiet
    }
    if (config->debug) {
        config->verbose = true; // debug includes verbose messages
        config->silent = false; // just in case it was accidentally set on command line
        config->quiet = false;
    }

    return config;
}