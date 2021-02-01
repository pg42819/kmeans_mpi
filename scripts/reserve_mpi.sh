#!/usr/bin/env bash

# interactive, 1 processor, 1 hour
qsub -I -qmei -lnodes=4:ppn=32:r641 -lwalltime=1:30:00
