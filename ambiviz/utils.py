import os
from uuid import UUID
from xml.etree import ElementTree

import numpy as np


def read_exchange_format_from_metadata(
    audio_path: str | os.PathLike,
) -> tuple[str | None, str | None]:
    """Read Ambisonic channel ordering and normalization from WAV metadata.

    Supports ASWG iXML fields and legacy WAVE-EX B-format GUIDs, regardless
    of the filename extension. Audio samples are skipped rather than loaded.

    Returns:
        (channel_order, normalization), using Ambiviz's format names:
        ``ACN`` or ``FURSE_MALHAM`` and ``SN3D``, ``N3D`` or ``MAX_N``.
        Each missing or unrecognized value is None; unsupported containers
        return (None, None). No format is inferred from channel count.

    Raises:
        OSError: If the file cannot be opened or read.
        ValueError: If WAV chunks or iXML are malformed, or iXML conflicts
            with a legacy B-format GUID.
    """
    channel_order = normalization = None
    is_b_format = False
    with open(audio_path, "rb") as audio_file:
        header = audio_file.read(12)
        # shortcut: only RIFF/WAVE containers, add RF64/CAF/MP4 parsing when needed.
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None, None

        riff_end = int.from_bytes(header[4:8], "little") + 8
        if riff_end < 12 or riff_end > audio_file.seek(0, os.SEEK_END):
            raise ValueError("Invalid or truncated WAV container.")
        audio_file.seek(12)

        while audio_file.tell() < riff_end:
            if riff_end - audio_file.tell() < 8:
                raise ValueError("Truncated WAV chunk header.")
            chunk_header = audio_file.read(8)
            chunk_id = chunk_header[:4]
            chunk_size = int.from_bytes(chunk_header[4:], "little")
            next_chunk = audio_file.tell() + chunk_size + (chunk_size % 2)
            if next_chunk > riff_end:
                raise ValueError("Truncated WAV chunk.")

            if chunk_id == b"fmt ":
                fmt = audio_file.read(min(chunk_size, 40))
                if (
                    len(fmt) >= 40
                    and fmt[:2] == b"\xfe\xff"
                    and int.from_bytes(fmt[16:18], "little") >= 22
                    and str(UUID(bytes_le=fmt[24:40]))
                    in (
                        "00000001-0721-11d3-8644-c8c1ca000000",
                        "00000003-0721-11d3-8644-c8c1ca000000",
                    )
                ):
                    is_b_format = True
            elif chunk_id == b"iXML":
                try:
                    # Some exporters fill reserved iXML space with 0xFF bytes.
                    root = ElementTree.fromstring(
                        audio_file.read(chunk_size).rstrip(b"\0\xff")
                    )
                except ElementTree.ParseError as exc:
                    raise ValueError("Malformed WAV iXML metadata.") from exc
                for group in root.iter():
                    if group.tag.rsplit("}", 1)[-1].upper() != "ASWG":
                        continue
                    for field in group:
                        name = field.tag.rsplit("}", 1)[-1].replace("_", "").lower()
                        value = (field.text or "").strip().replace("_", "").upper()
                        if name == "ambisonicchnorder":
                            channel_order = {
                                "ACN": "ACN",
                                "FUMA": "FURSE_MALHAM",
                                "FURSEMALHAM": "FURSE_MALHAM",
                            }.get(value)
                        elif name == "ambisonicnorm":
                            normalization = {
                                "SN3D": "SN3D",
                                "N3D": "N3D",
                                "MAXN": "MAX_N",
                            }.get(value)
            audio_file.seek(next_chunk)

    if is_b_format:
        if channel_order not in (None, "FURSE_MALHAM") or normalization not in (
            None,
            "MAX_N",
        ):
            raise ValueError("iXML metadata conflicts with the WAV B-format GUID.")
        return "FURSE_MALHAM", "MAX_N"
    return channel_order, normalization


def check_ambisonics_order(audio: np.ndarray) -> int:
    """
    Check the order of ambisonics.
    Raises error if the number of channels does not correspond to a valid ambisonics order.

    Parameters:
        audio: (n_channels, n_samples)

    Returns:
        order: int, the order of ambisonics
    """
    assert len(audio.shape) == 2, "Audio should be 2D (n_channels, n_samples)"
    n_channels = audio.shape[0]

    order = int(np.sqrt(n_channels) - 1)
    if (order + 1) ** 2 != n_channels:
        raise ValueError(
            f"Number of channels {n_channels} does not correspond to a valid ambisonics order."
        )

    return order
