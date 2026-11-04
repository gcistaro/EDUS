# local potential at q0 = b1/8 with Hartree and SEX: continuity (O(q^2) in tight binding, see check_continuity_q.py)
python3 "${PROJECT_ROOT}/ci-test/check_continuity_q.py" "$1" Output --tolerance 3e-2
