"""Shared fixtures for ambiviz tests."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Synthetic data dimensions — small enough to be fast, large enough to exercise PIL resize
N_VIDEO_FRAMES = 4
VIDEO_H = 8
VIDEO_W = 16
AEM_N_NU = 9    # elevation bins
AEM_N_PHI = 18  # azimuth bins
FPS = 2
SR = 48000


@pytest.fixture
def fake_video():
    """A (N, H, W, 3) uint8 video array and fps."""
    rng = np.random.default_rng(0)
    video = rng.integers(100, 200, (N_VIDEO_FRAMES, VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    return video, FPS


@pytest.fixture
def fake_aem():
    """AEM of shape (N//2, n_nu, n_phi) — repeat(2) inside visualize_aem gives N frames."""
    rng = np.random.default_rng(1)
    aem = rng.random((N_VIDEO_FRAMES // 2, AEM_N_NU, AEM_N_PHI), dtype=np.float32) * 0.8 + 0.1
    timestamps = np.arange(N_VIDEO_FRAMES // 2, dtype=np.float32) / FPS
    phi_mesh = np.zeros((AEM_N_NU, AEM_N_PHI), dtype=np.float32)
    nu_mesh = np.zeros((AEM_N_NU, AEM_N_PHI), dtype=np.float32)
    return aem, timestamps, phi_mesh, nu_mesh


@pytest.fixture
def patched_env(fake_video, fake_aem, tmp_path):
    """Patches every external I/O dependency of visualize_aem and yields useful handles."""
    video_array, fps = fake_video

    mock_mg_instance = MagicMock()
    mock_mg_instance.numpy.return_value = (video_array, fps)

    mock_writer = MagicMock()

    with (
        patch("ambiviz.plot.musicalgestures.Mg360Video", return_value=mock_mg_instance),
        patch("ambiviz.plot.librosa.get_samplerate", return_value=SR),
        patch("ambiviz.plot.audio_to_aem", return_value=fake_aem),
        patch("ambiviz.plot.cv2.VideoWriter_fourcc", return_value=0),
        patch("ambiviz.plot.cv2.VideoWriter", return_value=mock_writer),
        patch("ambiviz.plot.subprocess.call") as p_subprocess,
        patch("ambiviz.plot.os.remove") as p_remove,
    ):
        yield {
            "tmp_path": tmp_path,
            "mock_writer": mock_writer,
            "p_subprocess": p_subprocess,
            "p_remove": p_remove,
            "video_array": video_array,
            "fps": fps,
        }
