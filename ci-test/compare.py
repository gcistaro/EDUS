import numpy as np
import sys
import json
from pathlib import Path


# --------------------------------------------------
# Load numeric file safely
# --------------------------------------------------
def load(path):
    try:
        return np.loadtxt(path)
    except Exception as e:
        print(f"[ERROR] Cannot load {path}: {e}")
        sys.exit(1)

# --------------------------------------------------
# Compare two arrays
# --------------------------------------------------
def compare(name, ref_file, out_file, rtol=1e-10, atol=1e-12):
    ref = load(ref_file)
    out = load(out_file)

    if ref.shape != out.shape:
        print(f"[FAIL] {name}")
        print(f"Shape mismatch: ref={ref.shape}, out={out.shape}")
        return 1

    diff = np.abs(ref - out)

    ok = np.allclose(ref, out, rtol=rtol, atol=atol)

    if not ok:
        print(f"[FAIL] {name}")
        print(f"Max abs error: {np.max(diff)}")
        print(f"Max rel error: {np.max(diff / (np.abs(ref) + 1e-15))}")
        return 1

    print(f"[OK] {name}")
    return 0

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: compare.py <ref_file> <out_file> [name]")
        sys.exit(1)

    ref_file = sys.argv[1]
    out_file = sys.argv[2]

    # test name = filename if not provided
    name = sys.argv[3] if len(sys.argv) > 3 else Path(ref_file).name

    # get tolerances per file if defined

    sys.exit(compare(name, ref_file, out_file))
