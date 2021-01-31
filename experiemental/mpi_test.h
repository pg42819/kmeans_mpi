#ifndef MPI_TEST_H
#define MPI_TEST_H
#include <stdio.h>
// debug and logging macros
#define DEBUG__INT(fmt, ...) printf("DEBUG " fmt "%s", __VA_ARGS__);
#define DEBUG(...) DEBUG__INT(__VA_ARGS__, "\n")

#endif
