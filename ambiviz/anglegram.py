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
        # phi_mesh and nu_mesh shape: (n_nu, n_phi)
        # aem shape: (n_frames, n_nu, n_phi)
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


def plot_anglegram(
    time_record,
    phi_record,
    nu_record,
    rms_record,
    db_threshold=-60,
    db_max=-20,
    title: Optional[str] = None,
    save_path: Optional[Union[str, os.PathLike[str], Path]] = None,
    plot_mode: str = "argmax",
    fps: int = 20,
):
    """Plot peak directions, optionally over full RMS maps.

    plot_mode: "argmax" (default) or "full".
        In full mode, ``rms_record`` is a linear RMS array of shape
        (n_frames, n_elevations, n_azimuths). Each heatmap averages over the
        other angle before converting to dB relative to the full map's maximum.
        The strongest direction per frame is overlaid as scatter points.
        ``fps`` sets the time-bin width in full mode, including single-frame maps.
        Full mode returns the input records without threshold filtering.
    """
    if plot_mode not in ("argmax", "full"):
        raise ValueError("plot_mode must be 'argmax' or 'full'")

    peak_phi, peak_nu, peak_rms = phi_record, nu_record, rms_record
    if plot_mode == "full":
        phi_mesh, nu_mesh = np.meshgrid(phi_record, nu_record)
        _, peak_phi, peak_nu, peak_rms = compute_anglegram_from_aem(
            time_stamp=time_record, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=rms_record
        )
    keep = peak_rms > db_threshold

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        if title:
            fig.suptitle(title, fontsize=16)

        for axis, angles, peak_angles, label, projection_axis in zip(
            ax,
            (phi_record, nu_record),
            (peak_phi, peak_nu),
            ("Azimuth", "Elevation"),
            (1, 2),
        ):
            if plot_mode == "full":
                # Rows near the poles cover less solid angle, so weight them by cos(nu).
                weights = np.cos(nu_record) if projection_axis == 1 else None
                energy = librosa.amplitude_to_db(
                    np.average(rms_record, axis=projection_axis, weights=weights),
                    ref=np.max(rms_record),
                )
                time_edges = np.append(
                    time_record - 0.5 / fps, time_record[-1] + 0.5 / fps
                )
                angle_step = angles[1] - angles[0] if len(angles) > 1 else np.pi
                angle_edges = np.append(
                    angles - angle_step / 2, angles[-1] + angle_step / 2
                )
                heatmap = axis.pcolormesh(
                    time_edges,
                    angle_edges,
                    np.ma.masked_less_equal(energy.T, db_threshold),
                    shading="flat",
                    vmin=db_threshold,
                    vmax=db_max,
                )
            sc = axis.scatter(
                time_record[keep],
                peak_angles[keep],
                c=peak_rms[keep],
                label=label,
                alpha=0.6 if plot_mode == "full" else 0.9,
                vmin=db_threshold,
                vmax=db_max,
                **(
                    {"s": 9, "linewidths": 0.4, "edgecolors": "white", "zorder": 3}
                    if plot_mode == "full"
                    else {}
                ),
            )
        # find the closest multiple of 2pi to azimuth_min
        azimuth_min_pi = np.floor(np.min(phi_record) / np.pi) * np.pi
        azimuth_max_pi = np.ceil(np.max(phi_record) / np.pi) * np.pi
        ax[0].set_ylim(azimuth_min_pi - 0.1, azimuth_max_pi + 0.1)
        yticks = np.arange(azimuth_min_pi, azimuth_max_pi + 0.1, np.pi / 2)
        ax[0].set_yticks(yticks)
        ytick_labels = []
        for ytick in yticks:
            multiple = int(round(2 * ytick / np.pi))
            numerator = multiple if multiple % 2 else multiple // 2
            coefficient = {-1: "-", 1: ""}.get(numerator, str(numerator))
            denominator = "/2" if multiple % 2 else ""
            ytick_labels.append(
                rf"${coefficient}\pi{denominator}$" if numerator else r"$0$"
            )
        ax[0].set_yticklabels(ytick_labels)

        ax[0].set(xlabel="Time (s)", ylabel="Azimuth (rad)")
        # horizontal lines on pi and 2pi
        for ytick in yticks:
            if np.abs(ytick % (2 * np.pi)) < 1e-1:
                ax[0].axhline(y=ytick, linestyle="-", color="black", alpha=0.5)
            else:
                ax[0].axhline(y=ytick, linestyle="--", alpha=0.3)

        ax[0].grid(axis="x")

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
        fig.colorbar(
            heatmap if plot_mode == "full" else sc,
            ax=ax.ravel().tolist(),
            label="RMS (dB)",
        )

    if save_path is not None:
        fig.savefig(save_path)
        logger.info(f"Figure saved to {save_path}")

    if plot_mode == "full":
        return ax, time_record, phi_record, nu_record, rms_record

    return (
        ax,
        time_record[rms_record > db_threshold],
        phi_record[rms_record > db_threshold],
        nu_record[rms_record > db_threshold],
        rms_record[rms_record > db_threshold],
    )


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
    mode: str = "argmax",
):
    """Plot an audio file's directions over time.

    ``mode="argmax"`` (default) plots the strongest direction per frame.
    ``mode="full"`` plots azimuth and elevation heatmaps, averaging linear
    RMS over the other angle, with argmax scatter points overlaid. Both
    heatmaps and scatter points use dB relative to the maximum RMS in the
    full AEM, with ``db_threshold`` and ``db_max`` setting the color scale.
    Values at or below ``db_threshold`` are hidden.

    Returns the two Matplotlib axes when ``save_path`` is not provided.
    """
    if mode not in ("argmax", "full"):
        raise ValueError("mode must be 'argmax' or 'full'")

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
    if mode == "argmax":
        timestamp, phi_record, nu_record, rms_record = compute_anglegram_from_aem(
            time_stamp=time_stamp, phi_mesh=phi_mesh, nu_mesh=nu_mesh, aem=aem
        )
    else:
        timestamp, phi_record, nu_record, rms_record = (
            time_stamp,
            phi_mesh[0, :],
            nu_mesh[:, 0],
            aem,
        )
    logger.debug(
        f"rms record max: {rms_record.max()}, min: {rms_record.min()}, mean: {rms_record.mean()}"
    )

    logger.debug(f"Plotting anglegram. Save path: {save_path}")
    if title is None:
        title = os.path.basename(audio_path)
        title = f"Anglegram of {title}"

    ax, *_ = plot_anglegram(
        timestamp,
        phi_record,
        nu_record,
        rms_record,
        db_threshold=db_threshold,
        db_max=db_max,
        title=title,
        save_path=save_path,
        plot_mode=mode,
        fps=fps,
    )

    if not save_path:
        return ax
