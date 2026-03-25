"""Unit and integration tests for ambiviz.anglegram."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ambiviz.anglegram import anglegram, compute_anglegram_from_aem

# Real test files (used by integration tests)
TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"
REAL_AUDIO_PATH = TEST_FILES_DIR / "test_anglegram.wav"
TEST_OUTPUT_DIR = Path(__file__).parent.parent / "test_output"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_aem_data(n_frames=10, height=4, width=8):
    """Return fake (aem, time_stamp, phi_mesh, nu_mesh) as compute_aem would."""
    rng = np.random.default_rng(0)
    aem = rng.random((n_frames, height, width)).astype(np.float32)
    time_stamp = np.linspace(0, 1, n_frames)
    phi_mesh, nu_mesh = np.meshgrid(
        np.linspace(-np.pi, np.pi, width),
        np.linspace(-np.pi / 2, np.pi / 2, height),
    )
    return aem, time_stamp, phi_mesh, nu_mesh


def _make_plot_return():
    n = 5
    return (MagicMock(), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))


# ---------------------------------------------------------------------------
# compute_anglegram_from_aem unit tests
# ---------------------------------------------------------------------------


class TestComputeAnglegramFromAem:
    def test_output_shapes(self):
        aem, time_stamp, phi_mesh, nu_mesh = _make_aem_data(n_frames=10)
        ts, phi, nu, energy = compute_anglegram_from_aem(
            time_stamp=time_stamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=aem
        )
        assert ts.shape == (10,)
        assert phi.shape == (10,)
        assert nu.shape == (10,)
        assert energy.shape == (10,)

    def test_energy_is_db(self):
        """Energy record should be in dB: max value must be 0.0."""
        aem, time_stamp, phi_mesh, nu_mesh = _make_aem_data(n_frames=8)
        _, _, _, energy = compute_anglegram_from_aem(
            time_stamp=time_stamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=aem
        )
        assert np.max(energy) == pytest.approx(0.0, abs=1e-5)

    def test_energy_nonpositive(self):
        """All dB values should be <= 0."""
        aem, time_stamp, phi_mesh, nu_mesh = _make_aem_data(n_frames=8)
        _, _, _, energy = compute_anglegram_from_aem(
            time_stamp=time_stamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=aem
        )
        assert np.all(energy <= 0.0)

    def test_picks_argmax_coordinates(self):
        """phi/nu should come from the grid point with the highest AEM value per frame."""
        n_frames, height, width = 3, 4, 8
        aem = np.zeros((n_frames, height, width), dtype=np.float32)
        phi_mesh, nu_mesh = np.meshgrid(
            np.linspace(-np.pi, np.pi, width),
            np.linspace(-np.pi / 2, np.pi / 2, height),
        )
        time_stamp = np.arange(n_frames, dtype=float)

        # Place the maximum at known positions
        expected_phi = []
        expected_nu = []
        for i in range(n_frames):
            row, col = i % height, (i * 2) % width
            aem[i, row, col] = 1.0
            expected_phi.append(phi_mesh[row, col])
            expected_nu.append(nu_mesh[row, col])

        _, phi, nu, _ = compute_anglegram_from_aem(
            time_stamp=time_stamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=aem
        )
        np.testing.assert_allclose(phi, expected_phi)
        np.testing.assert_allclose(nu, expected_nu)

    def test_requires_consistent_args(self):
        """Passing aem_path together with other args should raise AssertionError."""
        aem, time_stamp, phi_mesh, nu_mesh = _make_aem_data()
        with pytest.raises(AssertionError):
            compute_anglegram_from_aem(
                aem_path="dummy.npz",
                time_stamp=time_stamp,
                phi_mesh=phi_mesh,
                nu_mesh=nu_mesh,
                aem=aem,
            )

    def test_requires_all_args_when_no_path(self):
        """Omitting any required arg when aem_path is None should raise AssertionError."""
        aem, time_stamp, phi_mesh, nu_mesh = _make_aem_data()
        with pytest.raises(AssertionError):
            compute_anglegram_from_aem(
                time_stamp=time_stamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh
            )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_anglegram_integration():
    """Integration test for anglegram function."""
    try:
        anglegram(
            audio_path=REAL_AUDIO_PATH,
            save_path=TEST_OUTPUT_DIR / "test_anglegram.png",
        )
    except Exception as e:
        pytest.fail(f"Integration test failed with exception: {e}")
