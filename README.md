# toa_sp — Multi-Strategy Single-Pulse TOA Extraction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)

**toa_sp** extracts times of arrival (TOAs) directly from PSRFITS search-mode data, bypassing the folding step required by conventional pulsar timing pipelines. It implements nine complementary TOA estimation strategies — parametric (Gaussian, EMG, Voigt) and non-parametric (leading edge, centre of mass, peak, shapelet) — together with MCMC uncertainty estimation, automatic time-resolution optimisation, and sub-band cross-validation.

Designed for highly variable radio sources (RRATs, FRBs, mode-changing pulsars) where the integrated profile assumption breaks down.

## Installation

```bash
pip install toa_sp
```

For MCMC support (optional):

```bash
pip install toa_sp[all]
```

## Quick Start

```bash
toa_sp -f FRB20220529_tracking-M01_0158.fits \
    -dm 255.0 -s 1.1 -w 0.06 -bs 8 -bf 16 \
    --auto_res --compare --tim --tim_strategy best \
    -o timing_result
```

Or run as a script:

```bash
python toa_sp.py -f data.fits -dm 175.25 -s 2.5 -w 0.05 -bs 4 -bf 16
```

## TOA Strategies

| Strategy | Type | Description |
|---|---|---|
| `single` | Parametric | Single-component fit; TOA = μ₁ |
| `highest` | Parametric | Brightest fitted component |
| `error_weighted` | Parametric | Inverse-variance weighted average |
| `first_peak` | Parametric | Earliest significant (S/N > 5) component |
| `weighted` | Parametric | Amplitude-weighted average *(deprecated)* |
| `leading_edge` | Non-parametric | Half-max crossing + rise-time correction |
| `center_of_mass` | Non-parametric | Flux-weighted centroid (S > 3σ_off) |
| `peak` | Non-parametric | Profile maximum, quadratic sub-bin interpolation |
| `shapelet` | Non-parametric | Hermite–Gaussian decomposition, AICc order selection |

## Profile Models

- **Gaussian** (default) — symmetric, N-component
- **EMG** — exponentially modified Gaussian for scattering tails
- **Voigt** — combined Gaussian + Lorentzian broadening
- **Shapelet** — flexible basis expansion, automatically regularised

## Key Features

- **Auto resolution optimisation** (`--auto_res`) — maximises peak S/N via dyadic rebinning
- **Sub-band cross-validation** (`--sub_band_fit`) — tests TOA consistency across frequency
- **AICc model selection** (`--auto_n`) — data-driven choice of component number
- **MCMC uncertainty** (`--mcmc`) — robust posterior sampling via `emcee`
- **TEMPO2 output** (`--tim`) — writes `.tim` files for downstream timing analysis
- **Convergence diagnostic** — Δ_conv metric identifies unstable decompositions

## Requirements

- Python ≥ 3.9
- numpy, scipy, astropy, matplotlib
- Optional: emcee, corner

## Citation

If you use toa_sp in your research, please cite:

> Zhang S., Yang X. (2025). TOA_SP: A Multi-Strategy Framework for Single-Pulse Timing. *ApJ*, submitted.

```bibtex
@article{zhang2025toa_sp,
  title={TOA\_SP: A Multi-Strategy Framework for Single-Pulse Timing},
  author={Zhang, Songbo and Yang, Xuan},
  journal={ApJ},
  year={2025},
  note={submitted}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Repository Structure

```
toa_sp/
├── toa_sp.py          # Main pipeline
├── pyproject.toml      # Package metadata
├── tools/
│   ├── plot_pipeline.py       # Pipeline schematic
│   └── plot_rrat_residuals.py # Residual plot generation
└── README.md
```
