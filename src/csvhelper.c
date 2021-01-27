
/* CSV Helper is Excerpted from chapter 5 of 'The Practice of Programming' */
/* by Brian W. Kernighan and Rob Pike
   Here is what that book has to say about this library:

   Assumptions:
   Fields are separated by commas.
   A field may be enclosed in double-quote characters "...".
   A quoted field may contain commas but not newlines.
   A quoted field may contain double-quote characters ", represented by "".
   Fields may be empty; "" and an empty string both represent an empty field.
   Leading and trailing white space is preserved.

   */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "csvhelper.h"

enum { NOMEM = -2 };          /* out of memory signal */

static char *line    = NULL;  /* input chars */
static char *sline   = NULL;  /* line copy used by split */
static int  maxline  = 0;     /* size of line[] and sline[] */
static char **field  = NULL;  /* field pointers */
static int  maxfield = 0;     /* size of field[] */
static int  nfield   = 0;     /* number of fields in field[] */

static char fieldsep[] = ","; /* field separator chars */

static char *advquoted(char *);
static int split(void);

/* endofline: check for and consume \r, \n, \r\n, or EOF */
static int endofline(FILE *fin, int c)
{
    int eol;

    eol = (c=='\r' || c=='\n');
    if (c == '\r') {
        c = getc(fin);
        if (c != '\n' && c != EOF)
            ungetc(c, fin);	/* read too far; put c back */
    }
    return eol;
}

/* reset: set variables back to starting values */
static void reset(void)
{
    free(line);	/* free(NULL) permitted by ANSI C */
    free(sline);
    free(field);
    line = NULL;
    sline = NULL;
    field = NULL;
    maxline = maxfield = nfield = 0;
}

/*
 * csvgetline:  get one line, grow as needed
 * sample input: "LU",86.25,"11/4/1998","2:19PM",+4.0625
    reads one line from open input file f;
    assumes that input lines are terminated by \r, \n, \r\n, or EOF.
    returns pointer to line, with terminator removed, or NULL if EOF occurred.
    line may be of arbitrary length
    returns NULL if memory limit exceeded.
    line must be treated as read-only storage
    caller must make a copy to preserve or change contents.
*/
char *csvgetline(FILE *fin)
{
    int i, c;
    char *newl, *news;

    if (line == NULL) {			/* allocate on first call */
        maxline = maxfield = 1;
        line = (char *) malloc(maxline);
        sline = (char *) malloc(maxline);
        field = (char **) malloc(maxfield*sizeof(field[0]));
        if (line == NULL || sline == NULL || field == NULL) {
            reset();
            return NULL;		/* out of memory */
        }
    }
    for (i=0; (c=getc(fin))!=EOF && !endofline(fin,c); i++) {
        if (i >= maxline-1) {	/* grow line */
            maxline *= 2;		/* double current size */
            newl = (char *) realloc(line, maxline);
            if (newl == NULL) {
                reset();
                return NULL;
            }
            line = newl;
            news = (char *) realloc(sline, maxline);
            if (news == NULL) {
                reset();
                return NULL;
            }
            sline = news;


        }
        line[i] = c;
    }
    line[i] = '\0';
    if (split() == NOMEM) {
        reset();
        return NULL;			/* out of memory */
    }
    return (c == EOF && i == 0) ? NULL : line;
}

/* split: split line into fields */
static int split(void)
{
    char *p, **newf;
    char *sepp; /* pointer to temporary separator character */
    int sepc;   /* temporary separator character */

    nfield = 0;
    if (line[0] == '\0')
        return 0;
    strcpy(sline, line);
    p = sline;

    do {
        if (nfield >= maxfield) {
            maxfield *= 2;			/* double current size */
            newf = (char **) realloc(field,
                                     maxfield * sizeof(field[0]));
            if (newf == NULL)
                return NOMEM;
            field = newf;
        }
        if (*p == '"')
            sepp = advquoted(++p);	/* skip initial quote */
        else
            sepp = p + strcspn(p, fieldsep);
        sepc = sepp[0];
        sepp[0] = '\0';				/* terminate field */
        field[nfield++] = p;
        p = sepp + 1;
    } while (sepc == ',');

    return nfield;
}

/* advquoted: quoted field; return pointer to next separator */
static char *advquoted(char *p)
{
    int i, j;

    for (i = j = 0; p[j] != '\0'; i++, j++) {
        if (p[j] == '"' && p[++j] != '"') {
            /* copy up to next separator or \0 */
            int k = strcspn(p+j, fieldsep);
            memmove(p+i, p+j, k);
            i += k;
            j += k;
            break;
        }
        p[i] = p[j];
    }
    p[i] = '\0';
    return p + j;
}

/**
 * Returns a pointer to n-th field from the last line read by csvgetline
 *
 * - fields are numbered from 0.
 * - returns NULL if n < 0 or beyond last field.
 * - fields are separated by commas.
 * - fields may be surrounded by "..."; such quotes are removed;
 *   within "...", "" is replaced by " and comma is not a separator.
 * - in unquoted fields, quotes are regular characters.
 * - there can be an arbitrary number of fields of any length;
 * - returns NULL if memory limit exceeded.
 * - field must be treated as read-only storage;
 * - caller must make a copy to preserve or change contents.
 * - behavior undefined if called before csvgetline is called.
 *
 * @return pointer to the nth field - readonly
 */
char *csvfield(int n)
{
    if (n < 0 || n >= nfield)
        return NULL;
    return field[n];
}

/**
 * Returns the number of fields in the current line
 *
 * @return number of fields
 */
int csvnfield(void)
{
    return nfield;
}

/**
 * Read the first line into the headers array and return the number of headers
 * The headers array is pre-allocated but the header strings are allocated here
 * Callers must free them if they are at risk of growing.
 *
 * @param csv_file csv file pointer
 * @param headers pre-allocated array of strings
 * @return number of headers
 */
int csvheaders(FILE *csv_file, char **headers) {
    if (headers != NULL) {
        if ((line = csvgetline(csv_file)) != NULL) {
            for (int i = 0; i < csvnfield(); i++) {
                char *field = csvfield(i);
                headers[i] = malloc(strlen(field) + 1);
                strcpy(headers[i], field);
            }
        }
    }
    return csvnfield();
}

int test(char *csv_file_name, int max_lines) {
    FILE *csv_file = fopen(csv_file_name, "r");
    if (!csv_file) {
        fprintf(stderr, "Error: cannot read the input file at %s\n", csv_file_name);
        exit(1);
    }

    char *headers[20];
    char *line;

    int num_headers = csvheaders(csv_file, headers);
    int i = 0;
    while ((line = csvgetline(csv_file)) != NULL) {
        printf("line = '%s'\n", line);
        for (int i = 0; i < csvnfield(); i++) {
            printf("field[%d] = `%s'\n", i, csvfield(i));
        }
    }

    printf("headers: %s", headers[0]);
    for (int i = 1; i < num_headers; ++i) {
        printf(",%s", headers[i]);
    }
    printf("\n");
    return i;
}
