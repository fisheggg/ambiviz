"""Numerical tests for spherical-harmonic decoding and AEM computation."""

import numpy as np
import pytest
import torch
from librosa.util import frame

from ambiviz.ambisonics.common import spherical_harmonics, spherical_harmonics_matrix
from ambiviz.ambisonics.position import Position
from ambiviz.ambisonics.spherical_maps import AEMGenerator


@pytest.mark.parametrize("ordering", ["ACN", "FURSE_MALHAM", "SID"])
@pytest.mark.parametrize("normalization", ["SN3D", "N3D", "MAX_N"])
def test_spherical_harmonics_matrix_matches_per_position(ordering, normalization):
    positions = [
        Position(phi, nu, 1.0, "polar")
        for phi, nu in [(0, 0), (np.pi / 3, np.pi / 8), (-2.5, -1.2), (np.pi, np.pi / 2)]
    ]
    expected = np.stack(
        [spherical_harmonics(p, 3, ordering, normalization) for p in positions]
    )
    np.testing.assert_allclose(
        spherical_harmonics_matrix(positions, 3, ordering, normalization), expected
    )


@pytest.mark.parametrize("ambi_order", [1, 3])
@pytest.mark.parametrize("batch_size", [None, 7])
def test_compute_matches_decoded_rms(ambi_order, batch_size):
    audio = np.random.default_rng(0).standard_normal(
        (1000, (ambi_order + 1) ** 2), dtype=np.float32
    )
    generator = AEMGenerator(
        128, 64, ambi_order=ambi_order, n_phi=12, n_nu=7, batch_size=batch_size
    )

    frames = frame(audio, frame_length=128, hop_length=64, axis=0)
    decoded = generator.decoder.decode(torch.from_numpy(frames.copy())).numpy()
    expected = np.sqrt((decoded**2).mean(axis=1)).reshape(-1, 7, 12)

    result = generator.compute(audio)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "phi, nu", [(0.0, 0.0), (np.pi / 2, 0.0), (-2.0, 0.5), (1.0, -0.9)]
)
def test_plane_wave_peaks_at_source(phi, nu):
    source = Position(phi, nu, 1.0, "polar")
    signal = np.random.default_rng(1).standard_normal((512, 1))
    encoded = (signal * spherical_harmonics(source, 3)).astype(np.float32)
    generator = AEMGenerator(128, 128, ambi_order=3, n_phi=72, n_nu=37)

    aem = generator.compute(encoded)
    for energy in aem:
        peak = np.unravel_index(energy.argmax(), energy.shape)
        assert abs(generator.phi_mesh[peak] - phi) < 0.1
        assert abs(generator.nu_mesh[peak] - nu) < 0.1
