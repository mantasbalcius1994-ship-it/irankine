"""Trimble TTM („TNL TIN DTM FILE“) paviršiai - skaitymas.

Struktūra (little-endian):
  antraštė: versija (2 B), parašas (20 B), pavadinimas (32 B), vienetai ir kt. (3 B),
            6 x double: N poslinkis, E poslinkis, min N, min E, max N, max E,
            dydžiai (B): koordinatės, aukščio, viršūnės nr., trikampio nr.,
            viršūnių (pradžia int, kiekis int, įrašo dydis short), trikampių (tas pats),
            kraštų sąrašas (tas pats), pradžios taškai (tas pats)
  viršūnė:  N, E (koord. dydis, santykinės su poslinkiais), Z (aukščio dydis)
  trikampis: 3 viršūnių nr. (1-based), 3 kaimynų nr.
"""
from __future__ import annotations

import struct

import numpy as np

SIGNATURE = b"TNL TIN DTM FILE"


def _ints(buf, size):
    """Little-endian sveikieji (2, 3 arba 4 baitai) -> int64 masyvas."""
    b = np.frombuffer(buf, np.uint8).reshape(-1, size).astype(np.int64)
    out = np.zeros(len(b), np.int64)
    for k in range(size):
        out |= b[:, k] << (8 * k)
    return out


def read(path) -> dict:
    """-> {'name', 'pts' (n,3) X=E Y=N Z, 'tris' (m,3) 0-based, 'units'}"""
    data = open(path, "rb").read()
    if data[2:2 + len(SIGNATURE)] != SIGNATURE:
        raise ValueError(f"{path}: ne TTM failas (nerastas parašas „TNL TIN DTM FILE“)")
    name = data[22:54].split(b"\0", 1)[0].decode("latin-1").strip()
    units = data[54]
    off_n, off_e = struct.unpack_from("<2d", data, 57)
    p = 57 + 48
    csz, vsz, vnsz, tnsz = data[p:p + 4]
    p += 4
    sv, nv, rv, st, nt, rt = struct.unpack_from("<iihiih", data, p)
    if csz not in (4, 8) or vsz not in (4, 8) or vnsz not in (2, 3, 4):
        raise ValueError(f"{path}: nepalaikomi TTM laukų dydžiai ({csz}, {vsz}, {vnsz})")
    # viršūnės
    cf, vf = ("<f4" if csz == 4 else "<f8"), ("<f4" if vsz == 4 else "<f8")
    rec = np.frombuffer(data, np.uint8, nv * rv, sv).reshape(nv, rv)
    n = np.frombuffer(rec[:, :csz].tobytes(), cf).astype(float) + off_n
    e = np.frombuffer(rec[:, csz:2 * csz].tobytes(), cf).astype(float) + off_e
    z = np.frombuffer(rec[:, 2 * csz:2 * csz + vsz].tobytes(), vf).astype(float)
    # trikampiai
    trec = np.frombuffer(data, np.uint8, nt * rt, st).reshape(nt, rt)
    tris = np.column_stack([_ints(trec[:, k * vnsz:(k + 1) * vnsz].tobytes(), vnsz) for k in range(3)]) - 1
    ok = (tris >= 0).all(1) & (tris < nv).all(1)
    return {"name": name or "TTM", "pts": np.column_stack([e, n, z]), "tris": tris[ok], "units": units}
