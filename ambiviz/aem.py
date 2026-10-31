from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List

import cv2
import librosa
import musicalgestures
import numpy as np
import tqdm
from matplotlib import colormaps

from .ambisonics.audio_to_aem import compute_aem

logger = logging.getLogger(__name__)


def aem(
    audio_path: str,
    video_paths: List[str],
    save_path: str | Path,
    duration: float,
    audio_offset: float = 0.0,
    video_offset: float = 0.0,
    normalization_mode: str = "local",
    use_gpu: bool = False,
    aem_mode: str = "aem",
    aem_mask_alpha: float = 0.5,
    verbose: bool = False,
    to_db: bool = False,
    **kwargs,
):
    """
    Overlay a yellow-orange-red audio energy map on an equirectangular video.

    Parameters
    ----------
    normalization_mode: "global" or "local", default="local"
        Global uses the clip-wide min/max, preserving relative levels over time.
        Local uses each analysis frame's min/max to emphasize direction.
        Constant normalization ranges produce a transparent map.
    aem_mask_alpha: float in [0, 1]
        Maximum overlay opacity. Zero leaves the video unchanged before encoding.
    to_db: bool
        Convert RMS to dB before normalization. This reveals quieter directions.

    Notes
    -----
    Like SpatialAudioGen, the display maps normalized energy ``x`` to
    ``clip(2*x - 0.7, 0, 1)`` for both color and opacity. Maps are interpolated
    between analysis-window centers; the first/last map is held at clip edges.
    Additional keyword arguments are passed to ``compute_aem`` (e.g. ambi_order).
    """
    assert normalization_mode in ["global", "local"]
    assert aem_mode in ["aem"]
    if not 0 <= aem_mask_alpha <= 1:
        raise ValueError("aem_mask_alpha must be between 0 and 1.")

    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Load video
    logger.info("Loading video...")
    video, fps = musicalgestures.Mg360Video(
        video_paths,
        projection="equirectangular",
        starttime=video_offset,
        endtime=video_offset + duration,
    ).numpy()
    video = np.copy(video)  # make writable
    logger.info(f"Video shape: {video.shape}")
    logger.info(f"Video FPS: {fps}")
    logger.info(
        f"Video stats: max={np.max(video)}, min={np.min(video)}, mean={np.mean(video)}"
    )
    logger.info(f"Video dtype: {video.dtype}")

    video_frame_shape = video.shape[1:3]

    audio_sr = librosa.get_samplerate(audio_path)
    audio_frame_length = int(audio_sr * 4 / fps)
    aem, timestamp, phi_mesh, nu_mesh = compute_aem(
        audio_path,
        mode=aem_mode,
        fps=fps,
        audio_frame_length=audio_frame_length,
        audio_hop_length=int(audio_sr * 2 / fps),
        gpu=use_gpu,
        verbose=verbose,
        duration=duration,
        offset=audio_offset,
        to_db=to_db,
        **kwargs,
    )
    # cv2 counts video frame from top left,
    # aem counts from bottom left, so flip the aem vertically
    aem = np.flip(aem, axis=1)
    axes = None if normalization_mode == "global" else (1, 2)
    minimum = aem.min(axis=axes, keepdims=True)
    span = aem.max(axis=axes, keepdims=True) - minimum
    aem = np.divide(aem - minimum, span, out=np.zeros_like(aem), where=span > 0)
    logger.info(f"AEM shape: {aem.shape}")
    logger.info(f"AEM stats: max={np.max(aem)}, min={np.min(aem)}, mean={np.mean(aem)}")

    # compute_aem timestamps mark window starts, but RMS represents the center.
    centers = timestamp + audio_frame_length / (2 * audio_sr)
    positions = np.interp(np.arange(len(video)) / fps, centers, np.arange(len(aem)))
    # Musicalgestures and OpenCV both use BGR frames.
    colors = colormaps["YlOrRd"](np.linspace(0, 1, 256))[:, 2::-1] * 255
    frames = range(len(video))
    if verbose:
        frames = tqdm.tqdm(frames, desc="=> rendering AEM mask to video")
    for i in frames:
        left = int(positions[i])
        right = min(left + 1, len(aem) - 1)
        fraction = positions[i] - left
        energy = (1 - fraction) * aem[left] + fraction * aem[right]
        strength = cv2.resize(
            np.clip(2 * energy - 0.7, 0, 1).astype(np.float32),
            video_frame_shape[::-1],
            interpolation=cv2.INTER_LINEAR,
        )
        opacity = aem_mask_alpha * strength[..., None]
        heatmap = colors[(strength * 255).astype(np.uint8)]
        video[i] = np.clip((1 - opacity) * video[i] + opacity * heatmap, 0, 255)

    # save video
    save_path = Path(save_path).resolve().absolute()
    audio_path = Path(audio_path).resolve().absolute()
    if not save_path.suffix:  # add a filename if a dir is given
        save_path = save_path / f"{Path(audio_path).stem}.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path_tmp = save_path.with_stem(save_path.stem + "_tmp")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(save_path_tmp), fourcc, fps, video_frame_shape[::-1])
    if not out.isOpened():
        raise RuntimeError(f"Could not open video writer for {save_path_tmp}")
    range_ = range(video.shape[0])
    if verbose:
        range_ = tqdm.tqdm(range_, desc="=> writing video...")
    for frame_idx in range_:
        out.write(video[frame_idx])
    out.release()

    # merge audio and video, only take the first channel from ambisonics
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(save_path_tmp),
            "-ss",
            str(audio_offset),
            "-i",
            str(audio_path),
            "-filter_complex",
            "[1:a:0]pan=mono|c0=c0[a_out]",
            "-map",
            "0:v:0",
            "-map",
            "[a_out]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(save_path),
        ],
        check=True,
    )
    os.remove(save_path_tmp)

    if verbose:
        print(f"=> Video saved to {save_path}")

