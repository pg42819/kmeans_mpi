#!/usr/bin/env bash

# interactive, 1 processor, 1 hour
qsub -I -qmei -lnodes=1 -lwalltime=1:30:00
