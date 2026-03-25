import numpy as np
import torch

from .common import AmbiFormat, spherical_harmonics_matrix
from .position import Position

DECODING_METHODS = ["projection", "pseudoinv"]
DEFAULT_DECODING = "projection"


class AmbiDecoder(object):
    def __init__(
        self,
        speakers_pos: "list[Position]",
        ambi_format: AmbiFormat,
        method: str = DEFAULT_DECODING,
        gpu: bool = False,
    ):
        """
        Ambisonics decoder.

        Args:
            speakers_pos: list of Position objects, the positions of the speakers in the format (phi, nu)
            ambi_format: AmbiFormat object, the format of the ambisonics signal
            method: str, the decoding method to use, either 'projection' or 'pseudoinv'
        """
        assert method in DECODING_METHODS
        if isinstance(speakers_pos, Position):
            speakers_pos = [speakers_pos]
        assert isinstance(speakers_pos, list) and all(
            [isinstance(p, Position) for p in speakers_pos]
        )

        self.speakers_pos = speakers_pos
        self.sph_mat = spherical_harmonics_matrix(
            speakers_pos,
            ambi_format.order,
            ambi_format.ordering,
            ambi_format.normalization,
        )
        self.sph_mat = torch.from_numpy(self.sph_mat).to(dtype=torch.float32)

        self.method = method
        self.gpu = gpu

        if self.method == "pseudoinv":
            self.pinv = torch.from_numpy(np.linalg.pinv(self.sph_mat)).to(
                dtype=torch.float32
            )
            if self.gpu:
                self.pinv = self.pinv.cuda()
        if self.gpu:
            self.sph_mat = self.sph_mat.cuda()

    def decode(self, ambi):
        # if self.gpu:
        # print(f"dtype of ambi: {ambi.dtype}; dtype of sph_mat: {self.sph_mat.dtype}")
        # if not ambi.is_cuda:
        #     raise ValueError(
        #         "Ambisonics signal must be on GPU when the decoder is in gpu mode."
        #     )
        # print(f"=> Ambi shape: {ambi.shape}; Sph_mat shape: {self.sph_mat.shape}")
        if self.method == "projection":
            with torch.no_grad():
                return torch.tensordot(ambi, self.sph_mat.T, dims=([-1], [-2]))
        if self.method == "pseudoinv":
            with torch.no_grad():
                return torch.tensordot(ambi, self.pinv, dims=([-1], [-2]))
        # else:
        #     if self.method == "projection":
        #         return torch.from_numpy(np.dot(ambi, self.sph_mat.T))
        #     if self.method == "pseudoinv":
        #         return torch.from_numpy(np.dot(ambi, self.pinv))
