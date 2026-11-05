# phonons at q0 = b1/3 in the primitive cell (qgrid [3,1,1]) against the supercell 3 x 1 x 1 at Gamma
# (tb_models/hBN_synthetic_epw_q_sc3, utility/make_supercell_epw.py): same system, the run is repeated in supercell/
mkdir -p supercell
( cd supercell && ${MPIEXEC:-mpirun} ${MPIEXEC_FLAGS:-} -np 2 "${BUILD_DIR:-${PROJECT_ROOT}/build}/EDUS" \
      "${PROJECT_ROOT}/ci-test/inputs/hBN_phonons_q_supercell.json" > out.txt 2>&1 ) || { echo "[FAIL] EDUS run of the supercell"; exit 1; }
python3 "${PROJECT_ROOT}/ci-test/check_supercell_q.py" Output supercell/Output 3 --tolerance 1e-6
