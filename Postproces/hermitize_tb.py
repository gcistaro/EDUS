"""
Makes a wannier90 seedname_tb.dat file exactly hermitian.

The Hamiltonian and the position operator between Wannier functions must satisfy
    H_mn(R) = conj(H_nm(-R)),    r_mn(R) = conj(r_nm(-R)),
EDUS checks it (up to 1e-13) before diagonalizing H(k). Files written with truncations or
few digits can violate it: here each pair (R, -R) is replaced by its hermitian part
    H_mn(R) <- [H_mn(R) + conj(H_nm(-R))]/2
and the same for r. The maximum correction is printed. The values are written with 8 decimals, as
wannier90 does: H(R) and H(-R) remain exact conjugates after the rounding.

Usage: python3 hermitize_tb.py input_tb.dat output_tb.dat
"""
import sys
import numpy as np


def read_tb(path):
    with open(path) as f:
        lines = f.read().split("\n")
    header = lines[:4]                    # comment + lattice vectors
    nw = int(lines[4])
    nR = int(lines[5])
    i = 6
    deg = []
    while len(deg) < nR:
        deg += [int(x) for x in lines[i].split()]
        i += 1

    def read_blocks(ncol):
        nonlocal i
        Rs, blocks = [], []
        for _ in range(nR):
            while not lines[i].strip():
                i += 1
            Rs.append(tuple(int(x) for x in lines[i].split()))
            i += 1
            data = np.array([lines[i + j].split() for j in range(nw * nw)], dtype=float)
            i += nw * nw
            M = np.zeros((ncol, nw, nw), dtype=complex)
            m = data[:, 0].astype(int) - 1
            n = data[:, 1].astype(int) - 1
            for c in range(ncol):
                M[c, m, n] = data[:, 2 + 2 * c] + 1j * data[:, 3 + 2 * c]
            blocks.append(M)
        return Rs, blocks

    RH, H = read_blocks(1)
    Rr, r = read_blocks(3)
    assert RH == Rr, "different R vectors for H and r"
    return header, nw, deg, RH, H, r


def hermitize(Rs, blocks, name):
    index = {R: iR for iR, R in enumerate(Rs)}
    out = []
    max_correction = 0.
    for iR, R in enumerate(Rs):
        minusR = tuple(-x for x in R)
        if minusR not in index:
            raise RuntimeError(f"{name}: R = {R} has no -R in the file")
        partner = blocks[index[minusR]]
        herm = 0.5 * (blocks[iR] + np.conj(np.transpose(partner, (0, 2, 1))))
        max_correction = max(max_correction, np.abs(herm - blocks[iR]).max())
        out.append(herm)
    print(f"{name}: max correction = {max_correction:.3e}")
    return out


def write_tb(path, header, nw, deg, Rs, H, r):
    with open(path, "w") as f:
        for line in header:
            f.write(line + "\n")
        f.write(f"{nw:12d}\n{len(Rs):12d}\n")
        for i in range(0, len(deg), 15):
            f.write("".join(f"{d:5d}" for d in deg[i:i + 15]) + "\n")
        for ncol, blocks in ((1, H), (3, r)):
            for R, M in zip(Rs, blocks):
                f.write("\n" + "".join(f"{x:5d}" for x in R) + "\n")
                for n in range(nw):
                    for m in range(nw):
                        values = "".join(f"{M[c, m, n].real:16.8E}{M[c, m, n].imag:16.8E}" for c in range(ncol))
                        f.write(f"{m + 1:5d}{n + 1:5d}{values}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    header, nw, deg, Rs, H, r = read_tb(sys.argv[1])
    index = {R: iR for iR, R in enumerate(Rs)}
    for iR, R in enumerate(Rs):
        if deg[iR] != deg[index[tuple(-x for x in R)]]:
            raise RuntimeError(f"degeneracy of R = {R} differs from the one of -R")
    H = hermitize(Rs, H, "H (eV)")
    r = hermitize(Rs, r, "r (Angstrom)")
    write_tb(sys.argv[2], header, nw, deg, Rs, H, r)
    print(f"written {sys.argv[2]}")
