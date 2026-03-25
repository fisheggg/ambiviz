"""Unit and integration tests for ambiviz.aem."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ambiviz.aem import aem


# Real test files (used by integration tests)
TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"
REAL_AUDIO_PATH = TEST_FILES_DIR / "test_aem_3oa.wav"
REAL_VIDEO_PATH = TEST_FILES_DIR / "test_aem_video.mp4"
TEST_OUTPUT_DIR = Path(__file__).parent.parent / "test_output"


@pytest.mark.integration
def test_anglegram_integration():
    """Integration test for aem"""

    try:
        aem(
            audio_path=REAL_AUDIO_PATH,
            video_paths=[str(REAL_VIDEO_PATH)],
            save_path=str(TEST_OUTPUT_DIR / "test_aem_output.mp4"),
            duration=30,
            audio_offset=23.3 + 270,
            video_offset=32.5 + 270,
            aem_mask_alpha=0.5,
            ambi_order=3,
            to_db=True,
            normalization_mode="local",
            verbose=True,
        )
    except Exception as e:
        pytest.fail(f"Integration test for aem failed with exception: {e}")
