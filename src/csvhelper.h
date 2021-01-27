/* Copyright (C) 1999 Lucent Technologies */
/* Excerpted from 'The Practice of Programming' */
/* by Brian W. Kernighan and Rob Pike */

/* csv.h: interface for csv library */
#ifndef PROJECT1_CSVHELPER_H
#define PROJECT1_CSVHELPER_H
extern char *csvgetline(FILE *f); /* read next input line */
extern char *csvfield(int n);	  /* return field n */
extern int csvnfield(void);		  /* return number of fields */
extern int csvheaders(FILE *f, char *headers[]);
extern int test(char *f, int max_lines);
#endif //PROJECT1_CSVHELPER_H
