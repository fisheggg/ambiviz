import os
from typing import Optional, Union
from pathlib import Path

import torchaudio
import librosa
import numpy as np

from .spherical_maps import AEMGenerator, MelAEMGenerator


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

    """
    assert mode in ["aem", "melaem"]

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
            **kwargs,
        )

    y, sr_ = torchaudio.load(
        audio_path,
        frame_offset=int(sr * offset) if offset else 0,
        num_frames=int(sr * duration) if duration else -1,
    )

    if sr_ != sr:
        raise ValueError(f"Audio files have different sample rates: {sr_} and {sr}.")

    # compute the timestamp of each AEM frame
    time_stamp = np.arange(0, y.shape[1] - audio_frame_length, audio_hop_length) / sr_

    # compute aem
    # aem shape:
    # (n_frames, n_phi, n_nu) in aem mode
    # (n_frames, n_phi, n_nu, n_mels) in melaem mode
    aem = aemg.compute(y.T)
    # assert aem.shape[0] == len(time_stamp)

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
