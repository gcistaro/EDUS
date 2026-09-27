#!/usr/bin/env python3
"""Checks the output of EDUS_wfc2wannier on the synthetic cases produced by generate.py.
Usage: check.py <case directory>   (exit code 1 on failure)"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import gauss_coef  # noqa: E402


def read_record(f):
    (n,) = struct.unpack("<i", f.read(4))
    data = f.read(n)
    (m,) = struct.unpack("<i", f.read(4))
    assert n == m
    return data


def read_wfc(fname):
    with open(fname, "rb") as f:
        ik, x, y, z, ispin, gamma_only, scalef = struct.unpack("<i3diid", read_record(f))
        ngw, igwx, npol, nbnd = struct.unpack("<4i", read_record(f))
        B = np.frombuffer(read_record(f), "<f8").reshape(3, 3)
        M = np.frombuffer(read_record(f), "<i4").reshape(igwx, 3)
        evc = np.array([np.frombuffer(read_record(f), "<c16") for _ in range(nbnd)])
    return dict(ik=ik, xk=np.array([x, y, z]), npol=npol, nbnd=nbnd, B=B, M=M, evc=evc, gamma_only=gamma_only)


def main(d):
    ref = json.load(open(f"{d}/reference.json"))
    kpts = np.array(ref["kpts"])
    taus = ref["taus"]
    chi = None if ref["spinors"] is None else [np.array([c[0] + 1j * c[1] for c in sp]) for sp in ref["spinors"]]
    ok = True

    # 1) Bloch functions in the Wannier gauge == analytic Bloch sums of the Gaussians
    err = 0.0
    for ik, kc in enumerate(kpts):
        wf = read_wfc(f"{d}/out/wfc{ik + 1}.dat")
        assert wf["nbnd"] == len(taus) and not wf["gamma_only"]
        err = max(err, np.max(np.abs(wf["xk"] - kc @ wf["B"])))
        for n, t in enumerate(taus):
            c = gauss_coef(kc, wf["M"], t, ref["sigma"])
            if chi is not None:
                c = np.concatenate([chi[n][0] * c, chi[n][1] * c])
            err = max(err, np.max(np.abs(wf["evc"][n] - c)))
    print(f"  Bloch functions: max error {err:.2e}")
    ok &= err < 1.0e-9

    # 2) real-space Wannier functions: norm, centre, spread
    data = np.loadtxt(f"{d}/out/wannier_centres.txt", ndmin=2)
    for row in data:
        n = int(row[0]) - 1
        e_norm = abs(row[1] - 1.0)
        e_c = np.max(np.abs(row[2:5] - np.array(ref["centres_ang"][n])))
        e_s = abs(row[5] - ref["spread_ang2"])
        print(f"  WF {n + 1}: |norm-1| = {e_norm:.1e}  |centre error| = {e_c:.1e} Ang  |spread error| = {e_s:.1e} Ang^2")
        ok &= e_norm < 1.0e-6 and e_c < 1.0e-6 and e_s < 1.0e-5

    # 3) xsf files exist and are complete
    for n in range(len(taus)):
        txt = open(f"{d}/out/wannier_{n + 1:05d}.xsf").read()
        ok &= "END_BLOCK_DATAGRID_3D" in txt
    return ok


if __name__ == "__main__":
    good = main(sys.argv[1])
    print("  ->", "PASSED" if good else "FAILED")
    sys.exit(0 if good else 1)
