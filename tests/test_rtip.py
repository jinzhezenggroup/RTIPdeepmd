import numpy as np
import pytest

from rtipdeepmd.rtip import rti_dist, rti_dists_vecs, rti_pot_force
from rtipdeepmd.system import rand_rot


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_rti_dist_is_roto_translation_invariant(rng):
    c1 = rng.normal(size=(7, 3))
    c2 = c1 @ rand_rot(rng) + rng.normal(size=3)
    assert rti_dist(c1, c2) == pytest.approx(0.0, abs=1e-6)


def test_rti_dist_equals_rmsd_times_sqrt_n(rng):
    c1 = rng.normal(size=(6, 3))
    c2 = c1 + 0.05 * rng.normal(size=(6, 3))
    # Kabsch RMSD after optimal superposition
    a = c1 - c1.mean(0)
    b = c2 - c2.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    expected = np.sqrt(np.sum((b - a @ rot) ** 2))
    assert rti_dist(c1, c2) == pytest.approx(expected, rel=1e-8)
    dists, vecs = rti_dists_vecs(c1, c2)
    assert np.all(np.diff(dists) >= 0)
    assert np.linalg.norm(vecs[0]) == pytest.approx(dists[0], rel=1e-8)


@pytest.mark.parametrize("a", [-0.3, 0.2])
def test_rti_pot_force_is_negative_gradient(rng, a):
    c1 = rng.normal(size=(5, 3))
    c2 = c1 + 0.4 * rng.normal(size=(5, 3))
    sigma = 0.8
    e0, force = rti_pot_force(c1, c2, a, sigma)
    h = 1e-6
    grad = np.zeros_like(c2)
    for i in range(c2.shape[0]):
        for k in range(3):
            cp, cm = c2.copy(), c2.copy()
            cp[i, k] += h
            cm[i, k] -= h
            grad[i, k] = (rti_pot_force(c1, cp, a, sigma)[0] - rti_pot_force(c1, cm, a, sigma)[0]) / (2 * h)
    np.testing.assert_allclose(force, -grad, atol=1e-7)
    assert np.sum(force, axis=0) == pytest.approx(np.zeros(3), abs=1e-10)


def test_negative_height_attracts(rng):
    c1 = rng.normal(size=(6, 3))
    c2 = c1 + 0.3 * rng.normal(size=(6, 3))
    d0 = rti_dist(c1, c2)
    _, force = rti_pot_force(c1, c2, -0.1, d0)
    assert rti_dist(c1, c2 + 1e-3 * force) < d0
    _, force = rti_pot_force(c1, c2, +0.1, d0)
    assert rti_dist(c1, c2 + 1e-3 * force) > d0


def test_coincident_structures_give_height_and_zero_force(rng):
    c = rng.normal(size=(4, 3))
    e, f = rti_pot_force(c, c.copy(), -0.2, 0.5)
    assert e == pytest.approx(-0.2)
    assert np.all(f == 0.0)
