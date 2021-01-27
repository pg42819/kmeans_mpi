#ifndef LOG_H
#define LOG_H
#include <stdbool.h>
#include <stdio.h>

extern struct log_config *log_config;
extern struct log_config *parse_log_cli(int argc, char *argv[], char *all_options);

struct log_config {
    bool silent;
    bool verbose;
    bool debug;
    bool quiet;
};

// debug and logging macros
#define DEBUG__INT(fmt, ...) if (log_config->debug) printf("DEBUG " fmt "%s", __VA_ARGS__);
#define DEBUG(...) DEBUG__INT(__VA_ARGS__, "\n")
//#define DEBUG(...) (void)0

#define INFO__INT(fmt, ...) if (log_config != NULL && !log_config->quiet) printf(fmt "%s", __VA_ARGS__)
#define INFO(...) INFO__INT(__VA_ARGS__, "\n")
//#define INFO(...) (void)0

#define VERBOSE__INT(fmt, ...) if (log_config != NULL && log_config->verbose) printf(fmt "%s", __VA_ARGS__)
#define VERBOSE(...) VERBOSE__INT(__VA_ARGS__, "\n")
//#define VERBOSE(...) (void)0

#define ERROR__INT(fmt, ...) fprintf(stderr, "ERROR: " fmt "%s", __VA_ARGS__)
#define ERROR(...) ERROR__INT(__VA_ARGS__, "\n")
//#define ERROR(...) (void)0

#endif
