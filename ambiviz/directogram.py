import logging
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from .ambisonics.audio_to_aem import compute_aem

logger = logging.getLogger(__name__)


def plot_directogram(
    angles,
    energy,
    title=None,
    figsize=(8, 8),
    dpi=300,
    ax=None,
    save_path=None,
):
    with plt.style.context(["science", "no-latex"]):
        if ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="polar")
            ax.set_title(title if title is not None else "Directogram")
        ax.bar(
            x=angles / 180 * np.pi,
            height=energy,
            width=2 * np.pi / len(angles),
            color="tab:blue",
            alpha=0.7,
            label="Summed directional energy",
        )
        ax.set_theta_zero_location("N")

    if save_path is not None:
        plt.savefig(save_path)
        logger.info(f"Figure saved to {save_path}")

    return ax


def compute_directogram_from_aem(aem, time_stamp, phi_mesh, nu_mesh):
    """
    Calculate the directogram from the AEM.

    Args:
        aem: np.ndarray, shape (n_frames, n_nu, n_phi)
        time_stamp: np.ndarray, shape (n_frames,)
        phi_mesh: np.ndarray, azimuth. shape: (n_nu, n_phi)
        nu_mesh: np.ndarray, elevation. shape: (n_nu, n_phi)

    Returns:
        angles: np.ndarray, in degrees, shape (n_phi,)
        energy: np.ndarray, shape (n_phi,)
    """
    horizontal_nu_idx = nu_mesh.shape[0] // 2
    horizontal_aem = aem[:, horizontal_nu_idx, :]
    angles = phi_mesh[0, :] / np.pi * 180
    energy = np.zeros_like(angles)
    for i in range(horizontal_aem.shape[0]):
        argmax_idx = np.argmax(horizontal_aem[i])
        energy[argmax_idx] += horizontal_aem[i, argmax_idx]

    return angles, energy


def directogram(
    audio_path: Union[str, os.PathLike, Path],
    save_path: Optional[Union[str, os.PathLike, Path]] = None,
    ax=None,
    fps: int = 20,
    audio_frame_length: int = 4800,
    aem_width: int = 90,
    aem_height: int = 45,
    batch_size: int = 5,
    gpu: bool = False,
):
    aem, time_stamp, phi_mesh, nu_mesh = compute_aem(
        audio_path=audio_path,
        fps=fps,
        audio_frame_length=audio_frame_length,
        aem_width=aem_width,
        aem_height=aem_height,
        gpu=gpu,
        batch_size=batch_size,
    )

    logger.debug(f"Computing anglegram. AEM shape: {aem.shape}")
    angles, energy = compute_directogram_from_aem(aem, time_stamp, phi_mesh, nu_mesh)

    logger.debug(f"Plotting directogram. Save path: {save_path}")
    ax = plot_directogram(angles, energy, ax=ax, save_path=save_path)

    return ax, angles, energy
