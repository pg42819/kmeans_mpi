#ifndef MPI_LOG_H
#define MPI_LOG_H
#include <stdarg.h>
#include <stdio.h>
#include "log.h"

extern int mpi_rank;
extern enum log_level_t log_level;

const char * const log_level_string[6] = {
        [error] = "error",
        [warn] = "warn",
        [info]  = "info",
        [verbose]  = "verbose",
        [debug]  = "debug",
        [trace]  = "trace"
};

void node_color()
{
    int color = mpi_rank % 6 + 1;
    printf("\033[0;3%dm", color);
}

void reset_color()
{
    printf("\033[0m"); // reset terminal color
}

int mpi_log(int level, const char *fmt, ...)
{
    if (log_level < level) return 0;
    FILE *out = stdout;
    if (level == error) {
        out = stderr;
    }
    node_color();
    fprintf(out, "Node %d [%s] ", mpi_rank, log_level_string[level]);
    va_list args;
    va_start(args, fmt);
    int rc = vfprintf(out, fmt, args);
    va_end(args);
    fprintf(out, "\n");
    reset_color();
    return rc;
}

#endif