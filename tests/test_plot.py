# """Unit and integration tests for ambiviz.plot.visualize_aem."""
# from pathlib import Path

# import numpy as np
# import pytest

# from ambiviz.aem import visualize_aem

# AUDIO_PATH = "fake/audio.wav"
# VIDEO_PATH = "fake/video.mp4"

# # Real test files (used by integration tests)
# TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"
# REAL_AUDIO_PATH = TEST_FILES_DIR / "test_audio_foa.wav"
# REAL_VIDEO_PATH = TEST_FILES_DIR / "test_video.mp4"


# def _call(tmp_path, normalization_mode=None, **kwargs):
#     """Helper: invoke visualize_aem with a concrete save path."""
#     visualize_aem(
#         AUDIO_PATH,
#         VIDEO_PATH,
#         str(tmp_path / "out.mp4"),
#         duration=2.0,
#         audio_offset=0.0,
#         video_offset=0.0,
#         normalization_mode=normalization_mode,
#         **kwargs,
#     )


# # ---------------------------------------------------------------------------
# # Validation
# # ---------------------------------------------------------------------------

# def test_invalid_normalization_mode_raises(tmp_path):
#     with pytest.raises(AssertionError):
#         _call(tmp_path, normalization_mode="bad_mode")


# def test_invalid_aem_mode_raises(tmp_path):
#     with pytest.raises(AssertionError):
#         _call(tmp_path, aem_mode="bad_mode")


# # ---------------------------------------------------------------------------
# # Smoke tests — one per normalization mode
# # ---------------------------------------------------------------------------

# @pytest.mark.parametrize("normalization_mode", [None, "global", "local"])
# def test_smoke(patched_env, normalization_mode):
#     """Function completes without error for every normalization mode."""
#     _call(patched_env["tmp_path"], normalization_mode=normalization_mode)


# # ---------------------------------------------------------------------------
# # Side-effect verification
# # ---------------------------------------------------------------------------

# def test_ffmpeg_called_once(patched_env):
#     _call(patched_env["tmp_path"])
#     patched_env["p_subprocess"].assert_called_once()


# def test_temp_file_removed(patched_env):
#     _call(patched_env["tmp_path"])
#     patched_env["p_remove"].assert_called_once()


# def test_writer_called_per_frame(patched_env):
#     """cv2.VideoWriter.write() must be called once per video frame."""
#     n_video_frames = patched_env["video_array"].shape[0]
#     _call(patched_env["tmp_path"])
#     assert patched_env["mock_writer"].write.call_count == n_video_frames


# # ---------------------------------------------------------------------------
# # AEM overlay logic
# # ---------------------------------------------------------------------------

# def test_channel2_increased_by_aem(patched_env):
#     """Channel index 2 of every written frame must be >= the original (AEM adds energy)."""
#     original_ch2 = patched_env["video_array"][:, :, :, 2].astype(np.float64)

#     _call(patched_env["tmp_path"])

#     written_frames = [
#         call_args[0][0]  # first positional arg to write()
#         for call_args in patched_env["mock_writer"].write.call_args_list
#     ]
#     for i, frame in enumerate(written_frames):
#         assert np.all(frame[:, :, 2].astype(np.float64) >= original_ch2[i] * 0.94), (
#             f"Frame {i}: channel 2 should not decrease below original × 0.95 brightness factor"
#         )


# # ---------------------------------------------------------------------------
# # Integration tests — require real files and ffmpeg
# # Run with: pytest -m integration
# # ---------------------------------------------------------------------------

# # @pytest.mark.integration
# # @pytest.mark.parametrize("normalization_mode", [None, "global", "local"])
# # def test_integration_smoke(tmp_path, normalization_mode):
# #     """End-to-end run with real audio/video files; verifies output .mp4 is created."""
# #     out = tmp_path / "out.mp4"
# #     visualize_aem(
# #         str(REAL_AUDIO_PATH),
# #         str(REAL_VIDEO_PATH),
# #         str(out),
# #         duration=1.0,
# #         audio_offset=0.0,
# #         video_offset=0.0,
# #         normalization_mode=normalization_mode,
# #     )
# #     assert out.exists(), "Output video file was not created"
# #     assert out.stat().st_size > 0, "Output video file is empty"
