import logging
import os
import warnings
from typing import Optional, Union
from pathlib import Path

import torchaudio
import librosa
import numpy as np

from ..utils import read_exchange_format_from_metadata
from .common import DEFAULT_NORMALIZATION, DEFAULT_ORDERING
from .spherical_maps import AEMGenerator, MelAEMGenerator

logger = logging.getLogger(__name__)


def compute_aem(
    audio_path: Union[str, os.PathLike, Path],
    save_dir: Optional[Union[str, os.PathLike, Path]] = None,
    duration: Optional[float] = None,
    offset: Optional[float] = None,
    fps: int = 20,
    audio_frame_length: int = 4800,
    audio_hop_length: Optional[int] = None,
    aem_width: int = 90,
    aem_height: int = 45,
    gpu: bool = False,
    batch_size: int = 10,
    mode: str = "aem",
    n_mels: int = 16,
    verbose: bool = False,
    to_db: bool = False,
    channel_order: Optional[str] = None,
    normalization: Optional[str] = None,
    **kwargs,
):
    """
    compute audio energy map for an audio file.

    Args:
        audio_path: pathlike, path to the audio directory
        save_dir: pathliuk, path to the save directory
        duration: float, duration of the audio in seconds
        fps: int, frames per second
        audio_frame_length: int, frame length for audio in samples
        aem_width: int, width of the AEM
        aem_height: int, height of the AEM
        save_path: str, path to save the figure
        gpu: bool, use GPU for AEM computation
        batch_size: int, batch size for AEM computation
        mode: str, "aem" or "melaem"
        channel_order: channel ordering; None reads metadata, then defaults to ACN
        normalization: normalization; None reads metadata, then defaults to SN3D

    """
    assert mode in ["aem", "melaem"]

    if channel_order is None or normalization is None:
        metadata_order, metadata_normalization = read_exchange_format_from_metadata(
            audio_path
        )
        if channel_order is None:
            channel_order = metadata_order
        if normalization is None:
            normalization = metadata_normalization

        if channel_order is None or normalization is None:
            warnings.warn(
                "Could not determine the full Ambisonic exchange format from input "
                "arguments or file metadata. Defaulting missing channel_order "
                "to ACN and missing normalization to SN3D.",
                stacklevel=2,
            )
            if channel_order is None:
                channel_order = DEFAULT_ORDERING
            if normalization is None:
                normalization = DEFAULT_NORMALIZATION
    logger.debug(
        f"Using channel order: {channel_order}, normalization: {normalization} for AEM computation."
    )
    if channel_order != "ACN" or normalization != "SN3D":
        raise NotImplementedError(
            f"Only ACN channel order and SN3D normalization are currently supported for AEM computation, but got {channel_order}/{normalization}."
        )

    # Determine hop length
    _, sr = librosa.load(audio_path, mono=False, sr=None, duration=1)
    if sr % fps != 0:
        raise ValueError(f"Sample rate ({sr}) must be divisible by fps ({fps}).")
    if audio_hop_length is None:
        audio_hop_length = sr // fps
    if audio_hop_length > audio_frame_length:
        raise Warning(
            f"Audio hop length ({audio_hop_length}) is larger than audio frame length ({audio_frame_length}), information might be lost."
        )

    if gpu and verbose:
        print("=> Using GPU for AEM computation.")

    if mode == "aem":
        aemg = AEMGenerator(
            audio_frame_length,
            audio_hop_length,
            n_phi=aem_width,
            n_nu=aem_height,
            gpu=gpu,
            batch_size=batch_size,
            show_progress_bar=verbose,
            to_db=to_db,
            channel_order=channel_order,
            normalization=normalization,
            **kwargs,
        )
    elif mode == "melaem":
        aemg = MelAEMGenerator(
            frame_length=audio_frame_length,
            hop_length=audio_frame_length,
            n_phi=aem_width,
            n_nu=aem_height,
            device="cuda" if gpu else "cpu",
            batch_size=batch_size,
            n_mels=n_mels,
            sample_rate=sr,
            show_progress=verbose,
            to_db=to_db,
            channel_order=channel_order,
            normalization=normalization,
            **kwargs,
        )

    y, sr_ = torchaudio.load(
        audio_path,
        frame_offset=int(sr * offset) if offset else 0,
        num_frames=int(sr * duration) if duration else -1,
    )

    if sr_ != sr:
        raise ValueError(f"Audio files have different sample rates: {sr_} and {sr}.")

    # Compute AEM first.
    # AEM shape:
    # (n_frames, n_phi, n_nu) in aem mode
    # (n_frames, n_phi, n_nu, n_mels) in melaem mode
    # Its first dimension is the actual number of analysis frames.
    aem = aemg.compute(y.T.numpy())

    # Generate one timestamp for each AEM frame.
    time_stamp = np.arange(aem.shape[0], dtype=float) * audio_hop_length / sr_

    # save aem
    if save_dir is not None:
        aem_name = os.path.basename(audio_path).replace(f".{audio_format}", "_aem.npz")
        phi_mesh = aemg.phi_mesh
        nu_mesh = aemg.nu_mesh
        np.savez(
            os.path.join(save_dir, aem_name),
            aem=aem,
            time_stamp=time_stamp,
            phi_mesh=phi_mesh,
            nu_mesh=nu_mesh,
        )
        if verbose:
            print(f"=> AEM file saved to: {os.path.join(save_dir, aem_name)}")
    else:
        return aem, time_stamp, aemg.phi_mesh, aemg.nu_mesh
