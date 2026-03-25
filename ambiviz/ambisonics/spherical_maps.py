import logging
from typing import Optional

import numpy as np
import torch
from librosa import amplitude_to_db
from librosa.util import frame
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from .common import AmbiFormat
from .decoder import AmbiDecoder
from .position import Position

logger = logging.getLogger(__name__)


def spherical_mesh(angular_res=None, n_phi=None, n_nu=None):
    """
    phi(azimuth): pi (left) to -pi (right), counter-clockwise growing
    nu(elevation): -pi/2 (bottom) to pi/2 (top)
    """
    if angular_res is None and n_phi is None and n_nu is None:
        raise ValueError("Either angular_res or n_phi and n_nu must be provided")
    elif angular_res is not None:
        phi_rg = np.flip(np.arange(-180.0, 180.0, angular_res) / 180.0 * np.pi)
        nu_rg = np.arange(-90.0, 90.0, angular_res) / 180.0 * np.pi
    elif n_phi is not None and n_nu is not None:
        phi_rg = np.flip(np.linspace(-np.pi, np.pi, n_phi))
        nu_rg = np.linspace(-np.pi / 2, np.pi / 2, n_nu)
    elif n_phi is None or n_nu is None:
        raise ValueError("Both n_phi and n_nu must be provided")

    phi_mesh, nu_mesh = np.meshgrid(phi_rg, nu_rg)
    return phi_mesh, nu_mesh


class AEMGenerator(object):
    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        ambi_order: int = 1,
        angular_res: Optional[float] = None,
        n_phi: Optional[int] = None,
        n_nu: Optional[int] = None,
        batch_size: Optional[int] = None,
        gpu: bool = False,
        show_progress_bar: bool = False,
        to_db: bool = False,
    ):
        """
        generates audio energy maps from ambisonics audio

        Args:
            frame_length: int, frame length in samples
            hop_length: int, hop length in samples
            ambi_order: int, ambisonics order
            window: int, window size in samples
            angular_res: float, angular resolution in degrees. either angular_res or n_phi and n_nu must be provided
            n_phi: int, number of phi bins
            n_nu: int, number of nu bins
            gpu: bool, use GPU for decoding
            batch_size: int, batch size for decoding
            show_progress: bool, show progress bar
        """
        self.angular_res = angular_res
        self.phi_mesh, self.nu_mesh = spherical_mesh(angular_res, n_phi, n_nu)
        self.frame_shape = (self.phi_mesh.shape[0], self.nu_mesh.shape[1])
        self.n_speakers = len(self.phi_mesh.flatten())
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.gpu = gpu
        self.batch_size = batch_size
        self.show_progress = show_progress_bar
        self.to_db = to_db
        mesh_p = [
            Position(phi, nu, 1.0, "polar")
            for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))
        ]

        # Setup decoder
        self.decoder = AmbiDecoder(
            mesh_p, AmbiFormat(ambi_order), method="projection", gpu=gpu
        )

    def compute(self, audio: np.ndarray):
        """
        audio: (n_samples, n_harmonics)
        """
        assert len(audio.shape) == 2

        # frame the audio into ((n_batchs), n_frames, self.frame_length, n_harmonics)
        audio = np.copy(audio)
        data = frame(
            audio, frame_length=self.frame_length, hop_length=self.hop_length, axis=0
        )
        n_frames = data.shape[0]
        if self.batch_size is not None:
            # pad the data to make it divisible by batch_size
            n_pad = self.batch_size - (data.shape[0] % self.batch_size)
            data = np.pad(data, ((0, n_pad), (0, 0), (0, 0)), mode="constant")
            data = frame(
                data, frame_length=self.batch_size, hop_length=self.batch_size, axis=0
            )

        # decoded final shape: (n_frames, n_speakers)
        if self.batch_size is not None:
            # decoded init shape: ((n_batchs), n_frames, frame_length, n_speakers)
            decoded = np.zeros((data.shape[0], self.batch_size, self.n_speakers))
            range_ = range(data.shape[0])
            if self.show_progress:
                range_ = tqdm(range_, desc="Decoding batches")

            # print(f"=> memory before decoding: {torch.cuda.memory_summary()}")
            # actual decoding and rms computation
            for i in range_:
                data_in = torch.from_numpy(data[i, :, :, :])
                if self.gpu:
                    data_in = data_in.cuda()
                decoded[i, :, :] = np.sqrt(
                    np.mean(self.decoder.decode(data_in).cpu().numpy() ** 2, axis=1)
                )
                del data_in

            # reshape to (n_frames, n_phi, n_nu)
            decoded = decoded.reshape(-1, *self.frame_shape)[:n_frames]

        else:
            decoded = self.decoder.decode(data)
            # compute rms
            decoded = np.sqrt(decoded.mean(decoded**2, axis=1))
            # reshape to (n_frames, n_phi, n_nu)
            decoded = decoded.reshape(decoded.shape[0], *self.frame_shape)

        logger.debug(f"Decoded shape: {decoded.shape}")
        if self.to_db:
            return amplitude_to_db(decoded)
        else:
            return decoded


class MelAEMGenerator(object):
    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        n_mels: int = 128,
        ambi_order: int = 1,
        angular_res: float = None,
        n_phi: int = None,
        n_nu: int = None,
        batch_size: int = None,
        device: str = "cuda",
        show_progress: bool = False,
        to_db: bool = False,
        **mel_kwargs,
    ):
        """
        generates audio energy maps from ambisonics audio

        Args:
            frame_length: int, frame length in samples
            hop_length: int, hop length in samples
            ambi_order: int, ambisonics order
            window: int, window size in samples
            angular_res: float, angular resolution in degrees. either angular_res or n_phi and n_nu must be provided
            n_phi: int, number of phi bins
            n_nu: int, number of nu bins
            gpu: bool, use GPU for decoding
            batch_size: int, batch size for decoding
            show_progress: bool, show progress bar
        """
        self.angular_res = angular_res
        self.phi_mesh, self.nu_mesh = spherical_mesh(angular_res, n_phi, n_nu)
        self.n_speakers = len(self.phi_mesh.flatten())
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.n_mels = n_mels
        self.frame_shape = (self.phi_mesh.shape[0], self.nu_mesh.shape[1], self.n_mels)
        self.to_db = to_db

        mesh_p = [
            Position(phi, nu, 1.0, "polar")
            for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))
        ]

        # Setup decoder
        self.decoder = AmbiDecoder(
            mesh_p,
            AmbiFormat(ambi_order),
            method="projection",
            gpu=(True if device == "cuda" else False),
        )
        self.mel_transform = MelSpectrogram(
            n_mels=self.n_mels,
            n_fft=self.frame_length,
            hop_length=self.frame_length,
            win_length=self.frame_length,
            center=False,
            **mel_kwargs,
        ).to(self.device)

    def compute(self, audio: np.ndarray):
        """
        audio: (n_samples, n_harmonics)
        """
        assert len(audio.shape) == 2

        # frame the audio into ((n_batchs), n_frames, self.frame_length, n_harmonics)
        data = frame(
            audio, frame_length=self.frame_length, hop_length=self.hop_length, axis=0
        )
        n_frames = data.shape[0]
        # data = torch.from_numpy(data)
        if self.batch_size is not None:
            # pad the data to make it divisible by batch_size
            n_pad = self.batch_size - (data.shape[0] % self.batch_size)
            data = np.pad(data, ((0, n_pad), (0, 0), (0, 0)), mode="constant")
            data = frame(
                data, frame_length=self.batch_size, hop_length=self.batch_size, axis=0
            )
        # print(f"Data shape: {data.shape}")

        # decoded final shape: (n_frames, n_speakers)
        if self.batch_size is not None:
            # decoded init shape: ((n_batchs), n_frames, n_speakers), n_speakers=n_phi*n_nu
            decoded_mel = torch.zeros(
                (data.shape[0], self.batch_size, self.n_speakers, self.n_mels)
            )

            range_ = range(data.shape[0])
            if self.show_progress:
                range_ = tqdm(range_, desc="Decoding batches")

            # actual decoding and mel computation
            for i in range_:
                data_in = torch.from_numpy(data[i, :, :, :]).to(self.device)
                # data_in shape: (n_batch, frame_length, n_harmonics)
                decoded_audio = self.decoder.decode(data_in).permute(0, 2, 1)
                # decoded_audio shape: (n_batch, n_speakers, frame_length)
                del data_in

                # compute mel spectrogram
                mel_emb = self.mel_transform(decoded_audio).squeeze(-1)
                # print(f"Mel spec shape: {mel_emb.shape}")
                decoded_mel[i] = mel_emb.cpu()
                # mel_spec shape: (n_batch, n_speakers, n_mels)
                del decoded_audio

            decoded_mel = decoded_mel.reshape(-1, *self.frame_shape)[:n_frames]

        # else:
        #     decoded = self.decoder.decode(data)
        #     # compute rms
        #     decoded = np.sqrt(decoded.mean(decoded**2, axis=1))
        #     # reshape to (n_frames, n_phi, n_nu)
        #     decoded = decoded.reshape(decoded.shape[0], *self.frame_shape)

        # print(f"Decoded shape: {decoded.shape}")
        # print(f"RMS shape: {rms.shape}")
        if self.to_db:
            return amplitude_to_db(decoded_mel)
        else:
            return decoded_mel
