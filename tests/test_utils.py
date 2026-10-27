from pathlib import Path

import pytest

from ambiviz.utils import read_exchange_format_from_metadata


@pytest.mark.integration
def test_read_exchange_format():
    audio_path = (
        Path(__file__).parent.parent
        / "test_files"
        / "428548__clucs__1st-order-ambisonics-clock-test-voice-white-noise.wav"
    )
    assert read_exchange_format_from_metadata(audio_path) == (None, None)
