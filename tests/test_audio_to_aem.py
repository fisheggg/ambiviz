"""Exchange-format resolution and decoder configuration for both AEM modes."""

import importlib
import warnings
from unittest.mock import patch

import numpy as np
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
            (None, None),
            ("SID", "N3D"),
            False,
        ),
        ({}, ("FURSE_MALHAM", "MAX_N"), ("FURSE_MALHAM", "MAX_N"), False),
        (
            {"channel_order": "FURSE_MALHAM"},
            ("ACN", None),
            ("FURSE_MALHAM", "SN3D"),
            True,
        ),
        (
            {"normalization": "N3D"},
            (None, "SN3D"),
            ("ACN", "N3D"),
            True,
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


@pytest.mark.parametrize(
    "ambi_order, channel_order, normalization",
    [(1, "FURSE_MALHAM", "MAX_N"), (3, "FURSE_MALHAM", "MAX_N"), (2, "SID", "N3D")],
)
def test_exchange_formats_match_acn_sn3d(ambi_order, channel_order, normalization):
    from ambiviz.ambisonics.common import spherical_harmonics
    from ambiviz.ambisonics.position import Position
    from ambiviz.ambisonics.spherical_maps import AEMGenerator

    source = Position(np.pi / 3, np.pi / 8, 1.0, "polar")
    signal = np.random.default_rng(0).standard_normal((256, 1))

    def render(order, norm):
        generator = AEMGenerator(
            64,
            64,
            ambi_order=ambi_order,
            n_phi=36,
            n_nu=19,
            batch_size=2,
            channel_order=order,
            normalization=norm,
        )
        encoded = signal * spherical_harmonics(source, ambi_order, order, norm)
        return generator.compute(encoded.astype(np.float32)), generator

    reference, generator = render("ACN", "SN3D")
    converted, _ = render(channel_order, normalization)
    np.testing.assert_allclose(converted, reference, rtol=1e-4, atol=1e-6)
    peak = np.unravel_index(reference[0].argmax(), reference[0].shape)
    assert abs(generator.phi_mesh[peak] - source.phi) < 0.2
    assert abs(generator.nu_mesh[peak] - source.nu) < 0.2
