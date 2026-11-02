# MagLoss Studio — research prototype

Combines a core-loss calculator and advanced frequency-dependent Steinmetz fitting based on:
S. Barg et al., *An Improved Empirical Formulation for Magnetic Core Losses Estimation Under Nonsinusoidal Induction*, IEEE TPEL 32(3), 2146–2154, 2017. DOI: 10.1109/TPEL.2016.2555359.

## Install and run

Python 3.10+:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To publish on Streamlit Community Cloud, upload this directory to a GitHub repository and set `app.py` as the entrypoint.

## CSV input format

Required columns: `frequency_hz,B_peak_T,loss_kW_m3` (sinusoidal steady-state measurements). Provide >=3 distinct flux-density measurements at each of >=3 distinct frequencies. Every row must be positive. Measurements must correspond to the same temperature and material, with consistent core-loss density units. See `demo_synthetic_data.csv` for the format; **its numbers are synthetic**.

## Implemented

- Eq. 24 (constant Steinmetz ISE) and Eq. 25 (two effective frequency terms) with configurable square-wave correction constant.
- Steinmetz sinusoidal benchmark and IGSE triangular-flux benchmark.
- Sequential identification from sinusoidal data: `ln(P)=H+beta ln(B)`, smooth `H(log f)` and `beta(log f)` with shape-preserving PCHIP, and compute `alpha=dH/d(ln f)`, `k=exp(H-alpha ln f)`.
- Interactive fitting diagnostic plots and downloadable fitted coefficients.
- Strict no-extrapolation policy for Eq. 25 effective-frequency lookup.

## Boundaries, cautions and next work

- The numerical fitting interpolation chosen here is a practical implementation, **not the exact specific interpolation coefficients** shown in the paper's Figures 5–8.
- The paper's Table I reference constants apply at 100 kHz, 100 °C; the reference calculator deliberately flags other use as extrapolation.
- No DC-bias correction, generalized arbitrary waveform/minor-loop decomposition, automatic datasheet graphic digitizer, authentication/database, or tested thermal coupling.
- No independent experimental validation has been performed for this software. In-sample fit metrics do not establish predictive performance.
- When fitting is noisy, derivatives of `H` can be unstable; future production work should add regularization, bootstrap uncertainties, cross-validation and separate nonsinusoidal measurement comparisons.
- Output core loss in W uses `Pv[kW/m^3] * Volume[cm^3] * 0.001`.
- License and distribution decisions should be made by the project owner; the supplied IEEE PDF is not redistributed.
