"""Unit and integration tests for ambiviz.directogram."""

import logging
from pathlib import Path

import pytest

from ambiviz.directogram import directogram

# Real test files (used by integration tests)
TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"
REAL_AUDIO_PATH = TEST_FILES_DIR / "test_directogram.wav"
TEST_OUTPUT_DIR = Path(__file__).parent.parent / "test_output"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_directogram_integration():
    """Integration test for directogram function."""
    try:
        directogram(
            audio_path=REAL_AUDIO_PATH,
            save_path=TEST_OUTPUT_DIR / "test_directogram.png",
        )
    except Exception as e:
        pytest.fail(f"Integration test failed with exception: {e}")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_nu", [3, 4])
def test_directogram_uses_horizon_rows(n_nu):
    import numpy as np

    from ambiviz.ambisonics.spherical_maps import spherical_mesh
    from ambiviz.directogram import compute_directogram_from_aem

    phi_mesh, nu_mesh = spherical_mesh(n_phi=8, n_nu=n_nu)
    aem = np.zeros((2, n_nu, 8))
    aem[:, 0, 1] = 5.0  # loud but far below the horizon: ignored
    distance = np.abs(nu_mesh[:, 0])
    rows = np.flatnonzero(np.isclose(distance, distance.min()))
    # Even n_nu has two horizon rows; only their average makes column 3 the peak.
    aem[0, rows[0], 3] = 2.0
    aem[0, rows[-1], 5] = 1.5
    aem[1, rows, 6] = 1.0

    angles, energy = compute_directogram_from_aem(aem, None, phi_mesh, nu_mesh)
    np.testing.assert_allclose(angles, np.degrees(phi_mesh[0]))
    np.testing.assert_allclose(energy, [0, 0, 0, 2.0 / len(rows), 0, 0, 1.0, 0])


def test_plot_directogram_saves_its_own_figure(tmp_path):
    import matplotlib.image
    import matplotlib.pyplot as plt
    import numpy as np

    from ambiviz.directogram import plot_directogram

    fig = plt.figure(figsize=(2, 2), dpi=50)
    ax = fig.add_subplot(111, projection="polar")
    other = plt.figure(figsize=(6, 4), dpi=50)  # becomes the current figure
    try:
        plot_directogram(
            np.arange(0, 360, 45), np.ones(8), ax=ax, save_path=tmp_path / "d.png"
        )
        assert matplotlib.image.imread(tmp_path / "d.png").shape[:2] == (100, 100)
    finally:
        plt.close(fig)
        plt.close(other)
