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

### `ambiviz.directogram()`

### `ambiviz.aem()`


## Gallery

### Anglegram
![](figures/anglegram.png)

### Directogram
![](figures/directogram.png)

### Audio Energy Map (AEM)
![](figures/aem.png)


## References

- The core ambisonics decoder is modified from [https://github.com/pedro-morgado/spatialaudiogen](https://github.com/pedro-morgado/spatialaudiogen).
