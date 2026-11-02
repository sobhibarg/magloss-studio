"""MagLoss Studio core mathematical engine.
ISE equations follow Barg et al., IEEE TPEL 2017, DOI 10.1109/TPEL.2016.2555359.
Conventions: frequency Hz; peak flux density Tesla; loss density kW/m^3.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import gamma

DEFAULT_C = np.pi / 4
# Table I at 100 kHz, 100 C; k unit is kW/m^3 with f in Hz and B in Tesla.
REFERENCE = {"3F3 (100 °C)": (0.020, 1.245, 2.69),
             "N67 (100 °C)": (0.172, 1.596, 2.29)}


def validate(f, B, D):
    if not (np.isfinite(f) and f > 0 and np.isfinite(B) and B > 0 and np.isfinite(D) and 0 < D < 1):
        raise ValueError('Require f>0, B>0 and 0<D<1.')


def se(f, B, k, a, beta):
    return k * f**a * B**beta


def ise_constant(f, B, D, k, a, beta, correction=DEFAULT_C):
    validate(f, B, D)
    return correction * k * (f**a) * (B**beta) * (D**(1-a) + (1-D)**(1-a)) / (2**a)


def igse(f, B, D, k, a, beta):
    """IGSE of paper Eq. 6-7 for one symmetric triangular B excursion per cycle.
    Sinusoidal Steinmetz coefficient k, B=peak, DeltaB=2B.
    """
    validate(f, B, D)
    integral = 2 * np.sqrt(np.pi) * gamma((a+1)/2) / gamma((a+2)/2)
    ki = k / (2**(beta-a) * (2*np.pi)**(a-1) * integral)
    return ki * (2*B)**beta * f**a * (D**(1-a) + (1-D)**(1-a))


@dataclass
class MaterialFit:
    frequencies: np.ndarray
    beta_values: np.ndarray
    h_values: np.ndarray
    beta_spline: object
    h_spline: object
    rmse_log: float
    mape: float
    source: pd.DataFrame

    @property
    def domain(self):
        return float(self.frequencies.min()), float(self.frequencies.max())

    def parameters(self, f):
        if not self.domain[0] <= f <= self.domain[1]:
            raise ValueError(f'Effective frequency {f/1e3:g} kHz is outside calibrated range '
                             f'{self.domain[0]/1e3:g}–{self.domain[1]/1e3:g} kHz.')
        x = np.log(f)
        beta = float(self.beta_spline(x))
        alpha = float(self.h_spline.derivative()(x))
        h = float(self.h_spline(x))
        k = float(np.exp(h - alpha*x))
        return k, alpha, beta

    def predict_sine(self, f, B):
        k, a, beta = self.parameters(float(f))
        return se(float(f), float(B), k, a, beta)


def fit_material(df):
    """Fit β vs ln(f) and H=ln(P)-β ln(B) vs ln(f), using shape-preserving interpolation.
    At least 3 distinct frequencies and 3 distinct flux densities per frequency.
    Frequencies should be grouped exactly in the submitted table (Hz values).
    """
    required = ['frequency_hz', 'B_peak_T', 'loss_kW_m3']
    if not all(c in df.columns for c in required):
        raise ValueError('CSV must contain columns: ' + ', '.join(required))
    data = df[required].apply(pd.to_numeric, errors='coerce').dropna().copy()
    if len(data) < 9 or (data <= 0).any().any():
        raise ValueError('At least 9 strictly positive, finite loss measurements required.')
    freq, betas, hs = [], [], []
    for f, group in data.groupby('frequency_hz', sort=True):
        if group.B_peak_T.nunique() < 3:
            raise ValueError(f'Frequency {f:g} Hz needs at least 3 distinct flux densities.')
        x = np.log(group.B_peak_T.to_numpy(dtype=float))
        y = np.log(group.loss_kW_m3.to_numpy(dtype=float))
        beta, h = np.polyfit(x, y, 1)
        freq.append(float(f)); betas.append(beta); hs.append(h)
    if len(freq) < 3:
        raise ValueError('At least 3 distinct frequencies required for continuous fitting.')
    freq = np.asarray(freq); betas = np.asarray(betas); hs = np.asarray(hs)
    x = np.log(freq)
    beta_fit = PchipInterpolator(x, betas, extrapolate=False)
    h_fit = PchipInterpolator(x, hs, extrapolate=False)
    result = MaterialFit(freq, betas, hs, beta_fit, h_fit, 0, 0, data)
    prediction = np.array([result.predict_sine(f, B) for f, B in
                           zip(data.frequency_hz, data.B_peak_T)])
    actual = data.loss_kW_m3.to_numpy()
    result.rmse_log = float(np.sqrt(np.mean(np.log(prediction/actual)**2)))
    result.mape = float(np.mean(np.abs(prediction-actual)/actual)*100)
    return result


def ise_fitted(f, B, D, fit, correction=DEFAULT_C):
    validate(f, B, D)
    f1, f2 = f/(2*D), f/(2*(1-D))
    k1, a1, b1 = fit.parameters(f1)
    k2, a2, b2 = fit.parameters(f2)
    return correction*(D*se(f1, B, k1, a1, b1) + (1-D)*se(f2, B, k2, a2, b2))


def fitted_coefficients(fit, n=100):
    freqs = np.geomspace(*fit.domain, n)
    values = [fit.parameters(float(f)) for f in freqs]
    return pd.DataFrame({'frequency_hz': freqs, 'k': [v[0] for v in values],
                         'alpha': [v[1] for v in values], 'beta': [v[2] for v in values]})
