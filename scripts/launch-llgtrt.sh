#!/bin/bash

set -e

if [ -z "$LLGTRT_BIN" ] ; then
    LLGTRT_BIN=/usr/local/bin/llgtrt
fi

if [ -n "$1" ]; then
    ENGINE=$1
    shift
else
    echo "Error: Must provide an engine path"
    exit 1
fi

# Optional second positional arg = DRAFT_ENGINE
if [[ "$1" != "--"* && -n "$1" ]]; then
    DRAFT_ENGINE=$1
    shift
fi

if test -f "$ENGINE/rank0.engine" ; then
    :
else
    echo "Error: $ENGINE/rank0.engine not found - doesn't look like engine directory"
    exit 1
fi

# Determine TP size
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
# Add draft engine if provided
if [ -n "$DRAFT_ENGINE" ]; then
    if test -f "$DRAFT_ENGINE/rank0.engine"; then
        cmd="$cmd --draft-engine $DRAFT_ENGINE"
    else
        echo "Error: $DRAFT_ENGINE/rank0.engine not found - doesn't look like draft engine directory"
        exit 1
    fi
fi

if [ $TP -gt 1 ] ; then
    MPI="mpirun -n $TP --allow-run-as-root"
    cmd="$MPI $cmd"
fi

cmd="$cmd ${@}"  # Append any additional params

set -x
echo $cmd
$cmd
