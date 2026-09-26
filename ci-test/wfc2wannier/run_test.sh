#!/bin/bash
# Test of EDUS_wfc2wannier on synthetic QE/wannier90 data (see generate.py).
# Usage: run_test.sh <path to EDUS_wfc2wannier> [number of MPI ranks, default 1]
# The MPI launcher can be changed with the variables MPIEXEC and MPIEXEC_FLAGS (as in ../scripts/run.sh).
set -e
EXE=$(realpath "$1")
NP=${2:-1}
MPIEXEC=${MPIEXEC:-mpirun}
MPIEXEC_FLAGS=${MPIEXEC_FLAGS:-}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/wfc2wannier_XXXXXX")
trap 'rm -rf "$WORK"' EXIT

python3 "$HERE/generate.py" "$WORK"
status=0
for case in plain dis gamma spinor; do
    echo "=== case $case ==="
    cd "$WORK/$case"
    if [ "$NP" -gt 1 ]; then
        $MPIEXEC $MPIEXEC_FLAGS -n "$NP" "$EXE" input.json > output.txt
    else
        "$EXE" input.json > output.txt
    fi
    python3 "$HERE/check.py" "$WORK/$case" || { status=1; cat output.txt; }
done
exit $status
