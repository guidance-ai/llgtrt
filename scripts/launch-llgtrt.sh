#!/bin/bash

set -e

if [ -z "$LLGTRT_BIN" ] ; then
    LLGTRT_BIN=/usr/local/bin/llgtrt
fi

ENGINE="$1"
DRAFT_ENGINE="$2"
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

cmd="$LLGTRT_BIN --engine $ENGINE"

# assume draft model has single engine,
# TODO assume fit single gpu check somehow?
if [ $DRAFT_ENGINE ]; then
    if test -f "${DRAFT_ENGINE}/rank0.engine" ; then
        TP=$((n+1))  # TODO assuming single engine on single gpu
    else
        echo "Error: $DRAFT_ENGINE/rank0.engine not found - doesn't look like draft engine directory"
        exit 1
    fi
else
    cmd="$cmd --draft-engine $DRAFT_ENGINE"
fi

if [ $TP -gt 1 ] ; then
    MPI="mpirun -n $TP --allow-run-as-root"
    cmd="$MPI $cmd"
fi

cmd="$cmd "$@""  # TODO this syntax isn't right

set -x
echo $cmd
$cmd
