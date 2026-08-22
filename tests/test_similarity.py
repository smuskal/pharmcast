import numpy as np

from pharmcast import bits as B
from pharmcast import pharmsim, pharmtan, pharmtan_matrix


def _fp(seed, density=0.05):
    rng = np.random.default_rng(seed)
    return B.pack(rng.random(B.N_PHARM) < density)


def test_identity_is_one():
    assert pharmtan(_fp(0), _fp(0)) == 1.0


def test_disjoint_is_zero():
    a = B.pack(np.array([True] + [False] * (B.N_PHARM - 1)))
    b = B.pack(np.array([False, True] + [False] * (B.N_PHARM - 2)))
    assert pharmtan(a, b) == 0.0


def test_pharmsim_decomposition_sums_to_the_union():
    a, b = _fp(1), _fp(2)
    r = pharmsim(a, b)
    assert r["shared"] + r["only_a"] + r["only_b"] == r["union"]
    assert r["shared"] + r["only_a"] == r["bits_a"]
    assert r["shared"] + r["only_b"] == r["bits_b"]


def test_pharmsim_agrees_with_pharmtan():
    a, b = _fp(3), _fp(4)
    assert abs(pharmsim(a, b)["tanimoto"] - pharmtan(a, b)) < 1e-12


def test_matrix_agrees_with_the_pairwise_routine():
    fps = [_fp(i) for i in range(5)]
    m = pharmtan_matrix(fps)
    assert m.shape == (5, 5)
    assert np.allclose(np.diag(m), 1.0)
    for i in range(5):
        for j in range(5):
            assert abs(m[i, j] - pharmtan(fps[i], fps[j])) < 1e-9


def test_matrix_against_separate_targets():
    q, t = [_fp(10), _fp(11)], [_fp(20), _fp(21), _fp(22)]
    m = pharmtan_matrix(q, t)
    assert m.shape == (2, 3)
    assert abs(m[1, 2] - pharmtan(q[1], t[2])) < 1e-9
