"""Unit and integration tests for ambiviz.aem."""

import importlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ambiviz.aem import aem


aem_module = importlib.import_module("ambiviz.aem")


@pytest.fixture
def render(monkeypatch, tmp_path):
    """Exercise the public renderer, replacing only audio/video I/O."""
    writer = MagicMock()
    monkeypatch.setattr(aem_module.cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(aem_module.librosa, "get_samplerate", lambda _: 48000)
    mux = MagicMock()
    remove = MagicMock()
    monkeypatch.setattr(aem_module.subprocess, "run", mux)
    monkeypatch.setattr(aem_module.os, "remove", remove)

    def run(maps, timestamps=None, video=None, **kwargs):
        maps = np.asarray(maps, dtype=float)
        if timestamps is None:
            timestamps = np.arange(len(maps)) / 2
        if video is None:
            video = np.full((8, *maps.shape[1:], 3), 240, dtype=np.uint8)
        source = MagicMock()
        source.numpy.return_value = (video, 4)
        monkeypatch.setattr(
            aem_module.musicalgestures, "Mg360Video", lambda *a, **k: source
        )
        compute = MagicMock(return_value=(maps, np.asarray(timestamps), None, None))
        monkeypatch.setattr(aem_module, "compute_aem", compute)
        writer.reset_mock()
        with np.errstate(divide="raise", invalid="raise"):
            aem("audio.wav", ["video.mp4"], tmp_path / "render", duration=2, **kwargs)
        frames = np.stack([call.args[0] for call in writer.write.call_args_list])
        return frames, compute

    run.writer = writer
    run.mux = mux
    run.remove = remove
    return run


@pytest.mark.parametrize("normalization_mode", ["global", "local"])
@pytest.mark.parametrize("level", [0, 0.5, -100])
def test_constant_maps_are_transparent(render, normalization_mode, level):
    frames, _ = render(np.full((2, 2, 3), level), normalization_mode=normalization_mode)
    np.testing.assert_array_equal(frames, 240)


def test_colormap_blends_in_bgr_and_preserves_background(render, tmp_path):
    frames, _ = render([[[0, 0, 0], [0, 1, 0]]])
    # Positive elevation is at the top; YlOrRd's peak is RGB (128, 0, 38).
    np.testing.assert_array_equal(frames[:, 0, 1], [[139, 120, 184]] * 8)
    np.testing.assert_array_equal(frames[:, 1], 240)
    np.testing.assert_array_equal(frames[:, 0, 0], 240)
    assert (tmp_path / "render").is_dir()
    assert not (tmp_path / "render" / "audio.mp4").is_dir()
    assert render.mux.call_args.kwargs == {"check": True}
    render.remove.assert_called_once()


def test_zero_opacity_preserves_video(render):
    video = np.random.default_rng(1).integers(0, 256, (8, 2, 3, 3), dtype=np.uint8)
    frames, _ = render([[[0, 0, 0], [0, 1, 0]]], video=video, aem_mask_alpha=0)
    np.testing.assert_array_equal(frames, video)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_opacity(render, alpha):
    with pytest.raises(ValueError, match="aem_mask_alpha"):
        render(np.zeros((1, 2, 3)), aem_mask_alpha=alpha)
    render.writer.write.assert_not_called()


def test_interpolates_at_window_centers_and_covers_last_frame(render):
    frames, compute = render(
        [[[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 0]]],
        timestamps=[0, 1],
    )
    assert compute.call_args.kwargs["audio_frame_length"] == 48000
    # Centers are 0.5 and 1.5 seconds. At 1 second both lobes are equal.
    np.testing.assert_array_equal(frames[4, 0, 0], frames[4, 0, 1])
    assert frames[4, 0, 0, 1] < 240
    assert frames[3, 0, 1, 1] < frames[4, 0, 1, 1] < frames[5, 0, 1, 1]
    np.testing.assert_array_equal(frames[0], frames[2])
    np.testing.assert_array_equal(frames[-1], frames[-2])
    assert frames[-1, 0, 0, 1] < 240


def test_global_retains_level_changes_local_emphasizes_direction(render):
    maps = [[[0, 0, 0], [0, 0.5, 0]], [[0, 0, 0], [0, 1, 0]]]
    global_frames, _ = render(maps, normalization_mode="global")
    local_frames, _ = render(maps)
    assert global_frames[0, 0, 1, 1] > global_frames[-1, 0, 1, 1]
    np.testing.assert_array_equal(local_frames[0], local_frames[-1])


@pytest.mark.parametrize("normalization_mode", ["global", "local"])
def test_silent_db_frame_stays_transparent(render, normalization_mode):
    frames, compute = render(
        [
            [[-100, -100, -100], [-100, -100, -100]],
            [[-100, -100, -100], [-100, -20, -100]],
        ],
        normalization_mode=normalization_mode,
        to_db=True,
    )
    np.testing.assert_array_equal(frames[0], 240)
    assert frames[-1, 0, 1, 1] < 240
    assert compute.call_args.kwargs["to_db"] is True


def test_resize_produces_smooth_bounded_overlay(render):
    video = np.full((8, 8, 12, 3), 240, dtype=np.uint8)
    frames, _ = render([[[0, 0, 0], [0, 1, 0]]], video=video, aem_mask_alpha=1)
    assert frames.shape == video.shape
    assert frames.dtype == np.uint8
    assert len(np.unique(frames[0, 0, :, 1])) > 2
    np.testing.assert_array_equal(frames[0, -1], 240)


def test_failed_mux_keeps_temporary_video(render):
    render.mux.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
    with pytest.raises(subprocess.CalledProcessError):
        render(np.zeros((1, 2, 3)))
    render.remove.assert_not_called()


def test_failed_writer_is_reported(render):
    render.writer.isOpened.return_value = False
    with pytest.raises(RuntimeError, match="video writer"):
        render(np.zeros((1, 2, 3)))
    render.mux.assert_not_called()


# Real test files (used by integration tests)
TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"
REAL_AUDIO_PATH = TEST_FILES_DIR / "test_aem_3oa.wav"
REAL_VIDEO_PATH = TEST_FILES_DIR / "test_aem_video.mp4"


@pytest.mark.integration
def test_aem_integration(tmp_path):
    """Decode, render and mux a short excerpt of the real 360 recording."""
    output = tmp_path / "overlay.mp4"
    aem(
        audio_path=REAL_AUDIO_PATH,
        video_paths=[str(REAL_VIDEO_PATH)],
        save_path=output,
        duration=2,
        audio_offset=23.3 + 270,
        video_offset=32.5 + 270,
        ambi_order=3,
        to_db=True,
        normalization_mode="local",
    )
    capture = aem_module.cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert capture.get(aem_module.cv2.CAP_PROP_FRAME_COUNT) == 50
        capture.set(aem_module.cv2.CAP_PROP_POS_FRAMES, 49)
        ok, frame = capture.read()
        assert ok and frame.shape == (704, 1408, 3)
    finally:
        capture.release()
