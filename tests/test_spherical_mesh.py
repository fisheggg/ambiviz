"""Tests for the equirectangular sampling grid."""

import numpy as np
import pytest

from ambiviz.ambisonics.spherical_maps import spherical_mesh


@pytest.mark.parametrize("n_phi, n_nu", [(4, 2), (90, 45), (36, 18)])
def test_mesh_uses_cell_centres(n_phi, n_nu):
    phi_mesh, nu_mesh = spherical_mesh(n_phi=n_phi, n_nu=n_nu)
    phi, nu = phi_mesh[0], nu_mesh[:, 0]
    assert phi_mesh.shape == nu_mesh.shape == (n_nu, n_phi)
    # Decreasing azimuth (pi on the left), increasing elevation, half a cell from each edge.
    np.testing.assert_allclose(phi[0], np.pi - np.pi / n_phi)
    np.testing.assert_allclose(phi[-1], -np.pi + np.pi / n_phi)
    np.testing.assert_allclose(nu[0], -np.pi / 2 + np.pi / (2 * n_nu))
    np.testing.assert_allclose(np.diff(phi), -2 * np.pi / n_phi)
    np.testing.assert_allclose(np.diff(nu), np.pi / n_nu)
    np.testing.assert_allclose(nu, -nu[::-1], atol=1e-12)


def test_mesh_has_no_duplicate_directions():
    phi_mesh, nu_mesh = spherical_mesh(n_phi=36, n_nu=19)
    xyz = np.stack(
        [
            np.cos(nu_mesh) * np.cos(phi_mesh),
            np.cos(nu_mesh) * np.sin(phi_mesh),
            np.sin(nu_mesh),
        ],
        axis=-1,
    ).reshape(-1, 3)
    assert len(np.unique(xyz.round(9), axis=0)) == len(xyz)


def test_angular_res_matches_counts():
    for actual, expected in zip(
        spherical_mesh(angular_res=4), spherical_mesh(n_phi=90, n_nu=45)
    ):
        np.testing.assert_allclose(actual, expected)


def test_mesh_requires_size():
    with pytest.raises(ValueError):
        spherical_mesh(n_phi=10)
