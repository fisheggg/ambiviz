"""Unit and integration tests for ambiviz.anglegram."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, QuadMesh
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
    time_stamp = np.arange(n_frames, dtype=float) / 20
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
# anglegram mode tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [None, "argmax"])
@pytest.mark.parametrize("db_threshold", [-60, -0.25])
def test_anglegram_argmax_mode(mode, db_threshold):
    data = _make_aem_data()
    aem, timestamp, phi_mesh, nu_mesh = data
    _, phi, nu, energy = compute_anglegram_from_aem(
        time_stamp=timestamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=aem
    )
    with patch("ambiviz.anglegram.compute_aem", return_value=data):
        axes = anglegram(
            "audio.wav",
            db_threshold=db_threshold,
            **({} if mode is None else {"mode": mode}),
        )

    try:
        keep = energy > db_threshold
        for axis, angles in zip(axes, (phi, nu)):
            scatter = axis.collections[0]
            assert isinstance(scatter, PathCollection)
            np.testing.assert_allclose(
                scatter.get_offsets(), np.column_stack((timestamp[keep], angles[keep]))
            )
            np.testing.assert_allclose(scatter.get_array(), energy[keep])
    finally:
        plt.close(axes[0].figure)


@pytest.mark.parametrize("reverse_azimuth", [False, True])
@pytest.mark.parametrize("db_threshold", [-30, -3])
def test_anglegram_full_mode(reverse_azimuth, db_threshold):
    aem = (
        np.array(
            [
                [[1.0, 0.9, 0.0], [0.2, 0.8, 0.95]],
                [[0.4, 0.05, 0.0], [0.1, 0.5, 0.0]],
            ]
        )
        * 0.2
    )
    timestamp = np.array([1.5, 1.75])
    azimuth = np.linspace(-np.pi, np.pi, 3)
    if reverse_azimuth:
        azimuth = azimuth[::-1]
    phi_mesh, nu_mesh = np.meshgrid(azimuth, [-np.pi / 2, np.pi / 2])
    with patch(
        "ambiviz.anglegram.compute_aem",
        return_value=(aem, timestamp, phi_mesh, nu_mesh),
    ):
        axes = anglegram(
            "audio.wav",
            mode="full",
            fps=4,
            db_threshold=db_threshold,
            db_max=0,
            title="Full map",
        )

    try:
        for axis, angles, projection_axis in zip(
            axes, (phi_mesh[0], nu_mesh[:, 0]), (1, 2)
        ):
            heatmap = axis.collections[0]
            assert isinstance(heatmap, QuadMesh)
            expected = 20 * np.log10(
                np.maximum(aem.mean(axis=projection_axis).T, 1e-5) / aem.max()
            )
            actual = heatmap.get_array().reshape(expected.shape)
            np.testing.assert_array_equal(actual.mask, expected <= db_threshold)
            np.testing.assert_allclose(
                actual.compressed(), expected[expected > db_threshold]
            )
            assert heatmap.get_clim() == (db_threshold, 0)

            coordinates = heatmap.get_coordinates()
            centers = (coordinates[:-1, :-1] + coordinates[1:, 1:]) / 2
            np.testing.assert_allclose(
                centers[:, :, 0], np.broadcast_to(timestamp, expected.shape)
            )
            np.testing.assert_allclose(
                centers[:, :, 1], np.broadcast_to(angles[:, None], expected.shape)
            )
            peak_angles = angles[[0, 1]]
            peak_energy = 20 * np.log10([1.0, 0.5])
            keep = peak_energy > db_threshold
            scatter = axis.collections[1]
            assert isinstance(scatter, PathCollection)
            np.testing.assert_allclose(
                scatter.get_offsets(),
                np.column_stack((timestamp[keep], peak_angles[keep])),
            )
            np.testing.assert_allclose(scatter.get_array(), peak_energy[keep])
            assert scatter.get_clim() == heatmap.get_clim()
            assert scatter.get_zorder() > heatmap.get_zorder()
            assert axis.get_xlabel() == "Time (s)"
        assert axes[0].get_ylabel() == "Azimuth (rad)"
        np.testing.assert_allclose(
            axes[0].get_yticks(), np.array([-1, -0.5, 0, 0.5, 1]) * np.pi
        )
        assert [label.get_text() for label in axes[0].get_yticklabels()] == [
            r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"
        ]
        assert axes[1].get_ylabel() == "Elevation (rad)"
        assert axes[0].figure._suptitle.get_text() == "Full map"
        colorbar = heatmap.colorbar
        assert colorbar is not None
        assert scatter.get_alpha() < 1
        axes[0].figure.canvas.draw()
        np.testing.assert_allclose(colorbar.solids.get_facecolors()[:, 3], 1)
    finally:
        plt.close(axes[0].figure)


def test_anglegram_invalid_mode():
    with patch("ambiviz.anglegram.compute_aem") as compute:
        with pytest.raises(ValueError, match="mode must be 'argmax' or 'full'"):
            anglegram("audio.wav", mode="invalid")
        compute.assert_not_called()


def test_anglegram_full_single_frame():
    with patch(
        "ambiviz.anglegram.compute_aem", return_value=_make_aem_data(n_frames=1)
    ):
        axes = anglegram("audio.wav", mode="full", fps=4)
    try:
        for axis in axes:
            coordinates = axis.collections[0].get_coordinates()
            np.testing.assert_allclose(coordinates[0, :, 0], [-0.125, 0.125])
    finally:
        plt.close(axes[0].figure)


def test_anglegram_full_mode_save(tmp_path):
    save_path = tmp_path / "full.png"
    with patch("ambiviz.anglegram.compute_aem", return_value=_make_aem_data()):
        try:
            result = anglegram("audio.wav", mode="full", save_path=save_path)
            assert result is None
            assert save_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            plt.close(plt.gcf())


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
