# Reference colours for check.cpp: 300 random unit quaternions (w x y z) and the
# orix IPFColorKeyTSL colour of the CONJUGATE quaternion (orix applies R d,
# SPPARKS needs R^T d) for sample direction z and x, per crystal.
# run (any env with orix): python -I gen_refs.py   -> ipf_ref_<crystal>_<z|x>.csv
# The cubic files ipf_ref_z.csv / ipf_ref_x.csv predate this script (orix 0.13).
import numpy as np
from orix.quaternion import Orientation, symmetry as S
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d

groups = {"hcp": S.D6, "trigonal": S.D3, "tetragonal": S.D4, "orthorhombic": S.D2}
q = np.random.default_rng(7).normal(size=(300, 4))
q /= np.linalg.norm(q, axis=1)[:, None]
conj = q * [1, -1, -1, -1]
for name, g in groups.items():
    for tag, d in (("z", Vector3d.zvector()), ("x", Vector3d.xvector())):
        rgb = IPFColorKeyTSL(g, direction=d).orientation2color(Orientation(conj, symmetry=g))
        np.savetxt(f"ipf_ref_{name}_{tag}.csv", np.hstack((q, rgb)), fmt="%.10f")
