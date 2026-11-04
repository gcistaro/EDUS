# local potential at q0 = b1/8 in IPA: linear response against the exact solution, and continuity
python3 "${PROJECT_ROOT}/ci-test/check_local_potential.py" "$1" Output --tolerance 1e-3 || exit 1
python3 "${PROJECT_ROOT}/ci-test/check_continuity_q.py" "$1" Output --tolerance 3e-2
