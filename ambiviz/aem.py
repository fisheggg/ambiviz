from __future__ import annotations

import glob
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import librosa
import musicalgestures
import numpy as np
import PIL
import tqdm

from .ambisonics.audio_to_aem import compute_aem

logger = logging.getLogger(__name__)


def aem(
    audio_path: str,
    video_paths: List[str],
    save_path: str | Path,
    duration: float,
    audio_offset: float = 0.0,
    video_offset: float = 0.0,
    normalization_mode: str = "global",
    use_gpu: bool = False,
    aem_mode: str = "aem",
    aem_mask_alpha: float = 0.5,
    verbose: bool = False,
    to_db: bool = False,
    **kwargs,
):
    """
    Takes an FOA audio and a 360 video, plots the audio energy map on the video.

    Parameters
    ----------
    normalization_mode: "global" or "local"
        if "global", normalize the AEM according to the min/max over the entire video.
        if "local", normalize the AEM of each frame by itself
    """
    assert normalization_mode in ["global", "local"]
    assert aem_mode in ["aem"]

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
    video = video * 0.95  # reduce brightness

    audio_sr = librosa.get_samplerate(audio_path)
    aem, timestamp, phi_mesh, nu_mesh = compute_aem(
        audio_path,
        mode=aem_mode,
        fps=fps,
        audio_frame_length=int(audio_sr * 4 / fps),
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
    aem = np.repeat(
        aem, 2, axis=0
    )  # when hoplength reduced to half fps, repeat each AEM frame once to match the length

    if normalization_mode == "global":
        # global normalize
        aem = (aem - np.min(aem)) / (np.max(aem) - np.min(aem))
        # aem_frame_gain = np.mean(aem, axis=(1, 2))
        # aem_frame_gain = (aem_frame_gain - np.min(aem_frame_gain)) / (
        #     np.max(aem_frame_gain) - np.min(aem_frame_gain)
        # )
    logger.info(f"AEM shape: {aem.shape}")
    logger.info(f"AEM stats: max={np.max(aem)}, min={np.min(aem)}, mean={np.mean(aem)}")

    # add aem colormask to video
    frames = range(min(video.shape[0], aem.shape[0]))
    if verbose:
        frames = tqdm.tqdm(frames, desc="=> rendering AEM mask to video")
    for i in frames:
        # aem_frame = aem[i]
        # aem_frame = aem_frame_gain[i] * (aem[i] - np.min(aem[i])) / (np.max(aem[i]) - np.min(aem[i])) # gain+local normalization
        if normalization_mode == "local":
            # local normalize
            aem_frame = (aem[i] - np.min(aem[i])) / (np.max(aem[i]) - np.min(aem[i]))
        elif normalization_mode == "global":
            # make local changes more drastic, but keeps the local max of the frame
            aem_frame = aem[i]
            aem_frame_max = np.max(aem_frame)
            aem_frame = aem_frame**8
            aem_frame = aem_frame / np.max(aem_frame) * aem_frame_max

        aem_frame = (aem_mask_alpha * aem_frame * 255).astype(video.dtype)
        # aem_frame = (aem_mask_alpha * aem[i] * 255).astype(video.dtype)
        aem_frame_resized = PIL.Image.fromarray(aem_frame).resize(
            (video_frame_shape[1], video_frame_shape[0])
        )
        # video[i, :, :, 2] = (video[i, :, :, 2] * (1 - aem_mask_alpha) + np.asarray(aem_frame_resized)).astype(video.dtype)
        video[i, :, :, 2] = (video[i, :, :, 2] + np.asarray(aem_frame_resized)).astype(
            video.dtype
        )
    video = np.clip(video, 0, 255)

    # save video
    save_path = Path(save_path).resolve().absolute()
    audio_path = Path(audio_path).resolve().absolute()
    if not save_path.suffix:  # add a filename if a dir is given
        save_path = save_path / f"{Path(audio_path).stem}.mp4"
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path_tmp = save_path.with_stem(save_path.stem + "_tmp")
    video = video.astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(save_path_tmp), fourcc, fps, video_frame_shape[::-1])
    range_ = range(video.shape[0])
    if verbose:
        range_ = tqdm.tqdm(range_, desc="=> writing video...")
    for frame_idx in range_:
        out.write(video[frame_idx])
    out.release()

    # merge audio and video, only take the first channel from ambisonics
    cmd = f"ffmpeg -hide_banner -y -i '{save_path_tmp}' -ss {audio_offset} -i '{audio_path}' -filter_complex \"[1:a:0]pan=mono|c0=c0[a_out]\" -map 0:v:0 -map \"[a_out]\" -c:v copy -c:a aac -shortest '{save_path}'"
    # cmd = [
    #     "ffmpeg",
    #     "-hide_banner",
    #     "-y",
    #     "-i",
    #     f"{save_path_tmp}",
    #     "-ss",
    #     f"{audio_offset}",
    #     "-i",
    #     f"{audio_path}",
    #     "-filter_complex",
    #     "[1:a:0]pan=mono|c0=c0[a_out]",
    #     "-map",
    #     "[a_out]",
    #     "-c:v",
    #     "copy",
    #     "-c:a",
    #     "aac",
    #     "-shortest",
    #     f"{save_path}",
    # ]
    subprocess.call(cmd, shell=True)
    # subprocess.check_call(cmd)
    os.remove(save_path_tmp)

    if verbose:
        print(f"=> Video saved to {save_path}")


if __name__ == "__main__":
    audio_path = "../test_files/test_aem_3oa.wav"
    video_path = "../test_files/test_aem_video.mp4"

    aem(
        audio_path,
        [video_path],
        "../test_output",
        duration=30,
        audio_offset=23.3 + 270,
        video_offset=32.5 + 270,
        verbose=True,
        aem_mask_alpha=0.5,
        ambi_order=3,
        to_db=True,
    )
