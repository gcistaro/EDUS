import numpy as np
import sys
import argparse
from pathlib import Path


# --------------------------------------------------
# Load numeric file safely
# --------------------------------------------------
def load(path):
    """
    Returns all the numbers of the file, in order, as a 1D array.
    - complex numbers written as (re,im) are read as two numbers
    - lines that are not fully numeric (comments, headers, dates) are skipped
    """
    try:
        values = []
        with open(path) as f:
            for line in f:
                tokens = line.replace("(", " ").replace(")", " ").replace(",", " ").split()
                try:
                    values.extend(float(token) for token in tokens)
                except ValueError:
                    continue
        return np.array(values)
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

    ok = np.allclose(out, ref, rtol=rtol, atol=atol)

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

    parser = argparse.ArgumentParser(description="Compare numerically an output file with a reference")
    parser.add_argument("ref_file")
    parser.add_argument("out_file")
    parser.add_argument("name", nargs="?", default=None, help="name printed in the report (default: file name)")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()

    name = args.name if args.name else Path(args.ref_file).name

    sys.exit(compare(name, args.ref_file, args.out_file, rtol=args.rtol, atol=args.atol))
