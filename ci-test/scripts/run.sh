#!/bin/bash
# Usage: run.sh <test name> [number of MPI ranks, default 1]
#
# Runs EDUS on ci-test/inputs/<test name>.json in CTEST/<test name> and compares
# every file present in ci-test/outputs/<test name> with the one produced by the run.
# The MPI launcher can be changed with the variables MPIEXEC and MPIEXEC_FLAGS.

SEEDNAME=$1
NP=${2:-1}
MPIEXEC=${MPIEXEC:-mpirun}
MPIEXEC_FLAGS=${MPIEXEC_FLAGS:-}

# ─────────────────────────────────────────────
# Locate project root from script location
# ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BUILD_DIR="${PROJECT_ROOT}/build"
INPUT="${PROJECT_ROOT}/ci-test/inputs/${SEEDNAME}.json"
REF_DIR="${PROJECT_ROOT}/ci-test/outputs/${SEEDNAME}"
OUT_DIR="${PROJECT_ROOT}/CTEST/${SEEDNAME}_np${NP}"

# ─────────────────────────────────────────────
# The ab-initio models are in the submodule tb_models/ab_initio (gcistaro/EDUS-models)
# ─────────────────────────────────────────────
if grep -q "tb_models/ab_initio" "${INPUT}" && [ ! -f "${PROJECT_ROOT}/tb_models/ab_initio/README.md" ]; then
    echo "[FAIL] models not found, download them with: git submodule update --init tb_models/ab_initio"
    exit 1
fi

# ─────────────────────────────────────────────
# Clean output directory
# ─────────────────────────────────────────────
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

# ─────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────
${MPIEXEC} ${MPIEXEC_FLAGS} -np ${NP} "${BUILD_DIR}/EDUS" "${INPUT}" || { echo "[FAIL] EDUS run"; exit 1; }

# ─────────────────────────────────────────────
# Compare outputs (numerical regression): every file in the reference directory
# ─────────────────────────────────────────────
num_failures=0
for ref in "${REF_DIR}"/*; do
    file=$(basename "${ref}")
    tolerance=""
    case "${file}" in
        absorbance.txt)
            # obtained from Output/Velocity.txt with the postprocessing script
            python3 "${PROJECT_ROOT}/Postproces/Absorbance.py" "--smearing=0.6" > absorbance.log 2>&1 \
                || echo "[ERROR] Absorbance.py failed, see ${OUT_DIR}/absorbance.log"
            out="${file}"
            ;;
        BANDSTRUCTURE.txt)
            # energies and orbital weights |U|^2, written with 6 significant digits: the weights
            # depend on the diagonalization (LAPACK) in the last digits, especially at degenerate k points
            out="${file}"
            tolerance="--rtol 1e-5 --atol 1e-5"
            ;;
        pdos.txt|wannier_tb.dat)
            # written in the working directory, with a limited number of digits
            out="${file}"
            tolerance="--rtol 1e-5 --atol 1e-8"
            ;;
        DM.txt)
            # density matrix at equilibrium: each rank writes its R points in Output/DM<rank>.txt,
            # their concatenation in rank order does not depend on the number of ranks.
            # Written with 6 significant digits
            cat $(ls Output/DM[0-9]*.txt | sort -V) > DM.txt
            out="${file}"
            tolerance="--rtol 1e-5 --atol 1e-8"
            ;;
        *)
            out="Output/${file}"
            ;;
    esac
    python3 "${PROJECT_ROOT}/ci-test/compare.py" "${ref}" "${out}" "${file}" ${tolerance} \
        || num_failures=$((num_failures+1))
done

# ─────────────────────────────────────────────
# Energy balance: E(t) - E(0) = work of the field (no reference needed)
# ─────────────────────────────────────────────
if ! grep -q '"decay"' "${INPUT}"; then
    python3 "${PROJECT_ROOT}/ci-test/check_energy.py" Output/Energy.txt || num_failures=$((num_failures+1))
else
    # with decay the energy is not conserved: analytic check of the relaxation after the pulses
    python3 "${PROJECT_ROOT}/ci-test/check_decay.py" "${INPUT}" Output || num_failures=$((num_failures+1))
fi

if [ ${num_failures} -ne 0 ]; then
    echo "${num_failures} comparison(s) failed for ${SEEDNAME}"
    exit 1
fi
echo "All comparisons passed for ${SEEDNAME}"
