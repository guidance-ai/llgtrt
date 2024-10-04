#!/bin/bash

set -e

if [ -z "$LLGTRT_BIN" ] ; then
    LLGTRT_BIN=/usr/local/bin/llgtrt
fi

ENGINE="$1"
shift

if test -f "$ENGINE/rank0.engine" ; then
    :
else
    echo "Error: $ENGINE/rank0.engine not found - doesn't look like engine directory"
    exit 1
fi

TP=1
for n in $(seq 1 8) ; do
    if test -f "$ENGINE/rank$n.engine" ; then
        TP=$((n+1))
    else
        break
    fi
done

MPI=

if [ $TP -gt 1 ] ; then
    MPI="mpirun -n $TP --allow-run-as-root"
fi

set -x
$MPI $LLGTRT_BIN --engine $ENGINE "$@"
