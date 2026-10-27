"""Exchange-format resolution and decoder configuration for both AEM modes."""

import importlib
import warnings
from unittest.mock import patch

import pytest
import torch

import ambiviz
from ambiviz.ambisonics import audio_to_aem
from ambiviz.ambisonics.decoder import AmbiDecoder


@pytest.mark.parametrize("mode", ["aem", "melaem"])
@pytest.mark.parametrize(
    "arguments, metadata, expected, should_warn",
    [
        (
            {"channel_order": "SID", "normalization": "N3D"},
            (None, None), ("SID", "N3D"), False,
        ),
        ({}, ("FURSE_MALHAM", "MAX_N"), ("FURSE_MALHAM", "MAX_N"), False),
        (
            {"channel_order": "FURSE_MALHAM"},
            ("ACN", None), ("FURSE_MALHAM", "SN3D"), True,
        ),
        (
            {"normalization": "N3D"},
            (None, "SN3D"), ("ACN", "N3D"), True,
        ),
        ({}, (None, None), ("ACN", "SN3D"), True),
    ],
)
def test_compute_aem_exchange_format(mode, arguments, metadata, expected, should_warn):
    audio = torch.ones((4, 128))
    with (
        patch.object(audio_to_aem.librosa, "load", return_value=(audio.numpy(), 48000)),
        patch.object(audio_to_aem.torchaudio, "load", return_value=(audio, 48000)),
        patch.object(
            audio_to_aem, "read_exchange_format_from_metadata", return_value=metadata
        ) as read_metadata,
        patch(
            "ambiviz.ambisonics.spherical_maps.AmbiDecoder", wraps=AmbiDecoder
        ) as decoder,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("ignore")
        importlib.reload(ambiviz)
        audio_to_aem.compute_aem(
            "audio.wav",
            mode=mode,
            fps=1500,
            audio_frame_length=64,
            aem_width=2,
            aem_height=2,
            batch_size=2,
            n_mels=2,
            **arguments,
        )
        warnings.warn("Unrelated warning", UserWarning)

    resolved_format = decoder.call_args.args[1]
    assert (resolved_format.ordering, resolved_format.normalization) == expected
    if len(arguments) == 2:
        read_metadata.assert_not_called()
    else:
        read_metadata.assert_called_once_with("audio.wav")
    assert len(caught) == int(should_warn)
    if should_warn:
        assert "Defaulting missing channel_order to ACN" in str(caught[0].message)
