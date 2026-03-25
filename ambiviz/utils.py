import numpy as np


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
