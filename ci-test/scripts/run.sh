#!/bin/bash
set -e

SEEDNAME=$1
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
OUT_DIR="${PROJECT_ROOT}/CTEST/${SEEDNAME}"

# ─────────────────────────────────────────────
# Clean output directory
# ─────────────────────────────────────────────
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

# ─────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────
"${BUILD_DIR}/EDUS" "${INPUT}"

# ─────────────────────────────────────────────
# Compare outputs (numerical regression)
# ─────────────────────────────────────────────
python3 "${PROJECT_ROOT}/ci-test/compare.py" \
    Output/Laser.txt "${REF_DIR}/Laser.txt"

python3 "${PROJECT_ROOT}/ci-test/compare.py" \
    Output/Laser_A.txt "${REF_DIR}/Laser_A.txt"

python3 "${PROJECT_ROOT}/ci-test/compare.py" \
    Output/DM0.txt "${REF_DIR}/DM0.txt"

python3 "${PROJECT_ROOT}/ci-test/compare.py" \
    Output/Population.txt "${REF_DIR}/Population.txt"

python3 "${PROJECT_ROOT}/ci-test/compare.py" \
    Output/Velocity.txt "${REF_DIR}/Velocity.txt"


python3 "${PROJECT_ROOT}/ci-test/compare.py" \
    Output/Population_wannier.txt "${REF_DIR}/Population_wannier.txt"

python3 "${PROJECT_ROOT}/Postproces/Absorbance.py" \
         "--smearing=0.6"

python3 "${PROJECT_ROOT}/ci-test/compare.py" \
    absorbance.txt "${REF_DIR}/absorbance.txt"
