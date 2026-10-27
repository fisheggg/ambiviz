# ambiviz

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![CI](https://github.com/fisheggg/ambiviz/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/fisheggg/ambiviz/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/ambiviz)](https://pypi.org/project/ambiviz/)

Python visualization tools for ambisonic audio (and 360 video). Supports GPU acceleration with PyTorch.

## Installation

```
pip install ambiviz
```

or if you use uv:

```
uv add ambiviz
```

## Functions

### `ambiviz.anglegram()`

By default, plots the strongest direction of arrival per frame (`mode="argmax"`).
Use `mode="full"` to show all directions as azimuth and elevation heatmaps,
with the strongest direction per frame overlaid as scatter points:

```python
import ambiviz

ambiviz.anglegram("audio.wav", mode="full")
```

Full mode averages linear RMS over elevation for the azimuth heatmap and over
azimuth for the elevation heatmap, then converts to dB relative to the full
audio energy map's maximum. Both modes use `db_threshold` and `db_max` for the
color scale and hide values at or below `db_threshold`.

### `ambiviz.directogram()`

### `ambiviz.aem()`

Blend a yellow–orange–red energy map over an equirectangular video. Stronger
directions appear redder and more opaque; weak regions keep the original video.
Maps interpolate between audio-window centers for smooth motion.

```python
ambiviz.aem(
    audio_path="audio.wav",
    video_paths=["video.mp4"],
    save_path="overlay.mp4",
    duration=10,
    normalization_mode="local",
    aem_mask_alpha=0.5,
)
```

`normalization_mode="global"` uses the clip-wide RMS range, preserving relative
levels over time. The default, `"local"`, emphasizes direction by normalizing each analysis
frame. `to_db=True` reveals quieter directions before normalization.
Use `normalization_mode="local"` for SpatialAudioGen-style directional emphasis.
Constant normalization ranges produce a transparent map; `aem_mask_alpha=0`
leaves all video pixels unchanged before encoding. The display uses
`clip(2*x - 0.7, 0, 1)` for normalized energy `x`, following
[SpatialAudioGen's overlay](https://github.com/pedro-morgado/spatialaudiogen/blob/a3e8ad1a6bfd78c51a8a073cd00f9d80387e164e/myutils.py#L246-L279).
Color and opacity indicate relative energy, not calibrated sound pressure.


## Gallery

### Anglegram (`plot_mode=argmax`)
![](figures/anglegram_argmax.png)

### Anglegram (`plot_mode=full`)
![](figures/anglegram_full.png)

### Directogram
![](figures/directogram.png)

### Audio Energy Map (AEM)
![](figures/aem_frames.png)


## References

- The core ambisonics decoder is modified from [https://github.com/pedro-morgado/spatialaudiogen](https://github.com/pedro-morgado/spatialaudiogen).
