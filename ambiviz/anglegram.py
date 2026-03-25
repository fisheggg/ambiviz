import datetime
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from .ambisonics.audio_to_aem import compute_aem

logger = logging.getLogger(__name__)


def compute_anglegram_from_aem(
    aem_path: Optional[Union[str, os.PathLike, Path]] = None,
    time_stamp=None,
    phi_mesh=None,
    nu_mesh=None,
    aem=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if aem_path is not None:
        assert (
            time_stamp is None and phi_mesh is None and nu_mesh is None and aem is None
        ), "If aem_path is provided, other parameters should be None."
    else:
        assert (
            time_stamp is not None
            and phi_mesh is not None
            and nu_mesh is not None
            and aem is not None
        ), "If aem_path is not provided, other parameters should not be None."
    if aem_path is not None:
        data = np.load(aem_path)
        time_stamp, phi_mesh, nu_mesh, aem = (
            data["time_stamp"],
            data["phi_mesh"],
            data["nu_mesh"],
            data["aem"],
        )
    phi_record, nu_record, energy_record = [], [], []

    aem_argmax = [np.unravel_index(np.argmax(r), r.shape) for r in aem]
    for i, point in enumerate(aem_argmax):
        phi_record.append(phi_mesh[point[0], point[1]])
        nu_record.append(nu_mesh[point[0], point[1]])
        energy_record.append(aem[i, point[0], point[1]])
    phi_record = np.array(phi_record)
    nu_record = np.array(nu_record)
    energy_record = np.array(energy_record)
    energy_record = librosa.amplitude_to_db(energy_record, ref=np.max)

    return time_stamp, phi_record, nu_record, energy_record


# def get_anglegram_mean_rms(audio_path) -> Tuple[float, np.ndarray]:
#     """
#     Calculate average rms of the audio file, made for noise estimation..
#     """
#     time_stamp, phi_mesh, nu_mesh, aem = compute_aem(
#         audio_path=audio_path,
#         fps=20,
#         audio_frame_length=4800,
#         aem_width=360,  # angular_res = 1
#         aem_height=180,
#         gpu=True,
#         batch_size=5,
#     )
#     phi_record, nu_record, energy_record = [], [], []

#     aem_argmax = [np.unravel_index(np.argmax(r), r.shape) for r in aem]
#     for i, point in enumerate(aem_argmax):
#         phi_record.append(phi_mesh[point[0], point[1]])
#         nu_record.append(nu_mesh[point[0], point[1]])
#         energy_record.append(aem[i, point[0], point[1]])
#     phi_record = np.array(phi_record)
#     nu_record = np.array(nu_record)
#     energy_record = np.array(energy_record)

#     return np.mean(energy_record), energy_record


def plot_anglegram(
    time_record,
    phi_record,
    nu_record,
    rms_record,
    db_threshold=-60,
    db_max=-20,
    title: Optional[str] = None,
    save_path: Optional[Union[str, os.PathLike[str], Path]] = None,
):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        if title:
            fig.suptitle(title, fontsize=16)

        sc = ax[0].scatter(
            time_record[rms_record > db_threshold],
            phi_record[rms_record > db_threshold],
            c=rms_record[rms_record > db_threshold],
            label="Azimuth",
            alpha=0.9,
            vmin=db_threshold,
            vmax=db_max,
        )
        # find the closest multiple of 2pi to azimuth_min
        azimuth_min_pi = np.floor(np.min(phi_record) / np.pi) * np.pi
        azimuth_max_pi = np.ceil(np.max(phi_record) / np.pi) * np.pi
        ax[0].set_ylim(azimuth_min_pi - 0.1, azimuth_max_pi + 0.1)
        yticks = np.arange(azimuth_min_pi, azimuth_max_pi + 0.1, np.pi)
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels([f"{int(ytick / np.pi)}$\pi$" for ytick in yticks])

        # plt.colorbar(sc, ax=ax[0], label="RMS (dB)")
        ax[0].set(xlabel="Time (s)", ylabel="Azimuth (rad)")
        # horizontal lines on pi and 2pi
        for ytick in yticks:
            if np.abs(ytick % (2 * np.pi)) < 1e-1:
                ax[0].axhline(y=ytick, linestyle="-", color="black", alpha=0.5)
            else:
                ax[0].axhline(y=ytick, linestyle="--", alpha=0.3)

        ax[0].grid(axis="x")

        sc = ax[1].scatter(
            time_record[rms_record > db_threshold],
            nu_record[rms_record > db_threshold],
            c=rms_record[rms_record > db_threshold],
            label="Elevation",
            alpha=0.9,
            vmin=db_threshold,
            vmax=db_max,
        )
        # ax[1].plot(time_record, nu_record, label='Elevation')
        ax[1].set(
            xlabel="Time (s)",
            ylabel="Elevation (rad)",
            ylim=(-0.5 * np.pi - 0.1, 0.5 * np.pi + 0.1),
        )
        ax[1].grid(axis="x")
        yticks = np.array([-0.5 * np.pi, -0.25 * np.pi, 0, 0.25 * np.pi, 0.5 * np.pi])
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(
            [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"]
        )
        for ytick in yticks:
            if ytick == 0:
                ax[1].axhline(y=ytick, linestyle="-", color="black", alpha=0.5)
            else:
                ax[1].axhline(y=ytick, linestyle="--", alpha=0.3)
        # plt.colorbar(sc, ax=ax[1], label="RMS (dB)")
        fig.colorbar(sc, ax=ax.ravel().tolist(), label="RMS (dB)")

    if save_path is not None:
        plt.savefig(save_path)
        logger.info(f"Figure saved to {save_path}")

    return (
        ax,
        time_record[rms_record > db_threshold],
        phi_record[rms_record > db_threshold],
        nu_record[rms_record > db_threshold],
        rms_record[rms_record > db_threshold],
    )


def plot_anglemap(aem_path: str, save_path: Optional[str] = None, **kwargs):
    data = np.load(aem_path)
    time_stamp, _, _, aem = (
        data["time_stamp"],
        data["phi_mesh"],
        data["nu_mesh"],
        data["aem"],
    )

    logger.debug(f"aem shape: {aem.shape}")  # (1093, 180, 360)

    phi_img = np.mean(aem, axis=1)  # (1093, 360)
    phi_img = np.flip(phi_img, axis=1).T  # (360, 1093), top left pi
    logger.debug(f"phi_img shape: {phi_img.shape}")

    nu_img = np.mean(aem, axis=2)  # (1093, 180)
    nu_img = np.flip(nu_img, axis=1).T  # (180, 1093), top left pi/2
    logger.debug(f"nu_img shape: {nu_img.shape}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"{os.path.basename(aem_path)}\nmin: {aem.mean():.2f}, max: {aem.max():.2f}",
        fontsize=16,
    )
    ax[0].imshow(phi_img, cmap="hot", aspect="auto", **kwargs)
    ax[0].set_yticks([0, 89, 179, 269, 359])
    ax[0].set_yticklabels([r"$\pi$", r"$\pi/2$", r"$0$", r"$-\pi/2$", r"$-\pi$"])
    ax[0].set_xticks(np.arange(0, len(time_stamp), 200))
    ax[0].set_xticklabels(np.arange(0, len(time_stamp), 200) / 20)

    ax[0].set_ylabel("Azimuth (rad)")

    ax[1].imshow(nu_img, cmap="hot", aspect="auto", **kwargs)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Elevation (rad)")
    ax[1].set_yticks([0, 44, 89, 134, 179])
    ax[1].set_yticklabels([r"$\pi/2$", r"$\pi/4$", r"$0$", r"$-\pi/4$", r"$-\pi/2$"])

    if save_path is not None:
        plt.savefig(save_path)
        logger.info(f"=> Figure saved to {save_path}")
    else:
        plt.show()


def frame_to_seconds(frame_idx, fps=30):
    return frame_idx / fps


def seconds_to_timestring(seconds):
    return str(datetime.timedelta(seconds=seconds))


def frame_to_timestring(frame_idx, fps=30):
    return seconds_to_timestring(frame_to_seconds(frame_idx, fps))


def anglegram(
    audio_path: Union[str, os.PathLike[str], Path],
    save_path: Optional[Union[str, os.PathLike[str], Path]] = None,
    fps: int = 20,
    audio_frame_length: int = 4800,
    aem_width: int = 90,
    aem_height: int = 45,
    gpu: bool = False,
    db_threshold: int = -60,
    db_max: int = -20,
    title: Optional[str] = None,
    batch_size: int = 5,
):
    logger.debug(f"Computing AEM. Audio path: {audio_path}")
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
    timestamp, phi_record, nu_record, rms_record = compute_anglegram_from_aem(
        time_stamp=time_stamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=aem
    )
    logger.debug(
        f"rms record max: {rms_record.max()}, min: {rms_record.min()}, mean: {rms_record.mean()}"
    )

    logger.debug(f"Plotting anglegram. Save path: {save_path}")
    if title is None:
        title = os.path.basename(audio_path)

    ax, *_ = plot_anglegram(
        timestamp,
        phi_record,
        nu_record,
        rms_record,
        db_threshold=db_threshold,
        db_max=db_max,
        title=title,
        save_path=save_path,
    )

    if not save_path:
        return ax
