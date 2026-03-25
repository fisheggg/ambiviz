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
