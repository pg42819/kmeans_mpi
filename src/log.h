#ifndef LOG_H
#define LOG_H
#include <stdio.h>

enum log_level_t {
    error = 0,
    warn = 1,
    info = 2,
    verbose = 3,
    debug = 4,
    trace = 5
};


extern enum log_level_t log_level;

#define IS_ERROR (log_level >= error)
#define IS_WARN (log_level >= warn)
#define IS_INFO (log_level >= info)
#define IS_VERBOSE (log_level >= verbose)
#define IS_DEBUG (log_level >= debug)
#define IS_TRACE (log_level >= trace)

// debug and logging macros
#define ERROR__INT(fmt, ...) if (log_level >= error) fprintf(stderr, "ERROR: " fmt "%s", __VA_ARGS__)
#define ERROR(...) ERROR__INT(__VA_ARGS__, "\n")
//#define ERROR(...) (void)0

#define WARN__INT(fmt, ...) if (log_level >= warn) printf(fmt "%s", __VA_ARGS__)
#define WARN(...) WARN__INT(__VA_ARGS__, "\n")
//#define WARN(...) (void)0

#define INFO__INT(fmt, ...) if (log_level >= info) printf(fmt "%s", __VA_ARGS__)
#define INFO(...) INFO__INT(__VA_ARGS__, "\n")
//#define INFO(...) (void)0

#define VERBOSE__INT(fmt, ...) if (log_level >= verbose) printf(fmt "%s", __VA_ARGS__)
#define VERBOSE(...) VERBOSE__INT(__VA_ARGS__, "\n")
//#define VERBOSE(...) (void)0

#define DEBUG__INT(fmt, ...) if (log_level >= debug) printf("DEBUG " fmt "%s", __VA_ARGS__);
#define DEBUG(...) DEBUG__INT(__VA_ARGS__, "\n")
//#define DEBUG(...) (void)0

#define TRACE__INT(fmt, ...) if (log_level >= trace) printf(fmt "%s", __VA_ARGS__)
#define TRACE(...) TRACE__INT(__VA_ARGS__, "\n")
//#define TRACE(...) (void)0

#define FAIL__INT(fmt, ...) fprintf(stderr, "FATAL ERROR: " fmt "%s", __VA_ARGS__); exit(1);
#define FAIL(...) FAIL__INT(__VA_ARGS__, "\n")
//#define ERROR(...) (void)0

#endif
