import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from engine import (REFERENCE, fit_material, ise_fitted, ise_constant, igse,
                    se, fitted_coefficients, DEFAULT_C)

st.set_page_config(page_title='MagLoss Studio', page_icon='🧲', layout='wide')
st.title('🧲 MagLoss Studio')
st.caption('Core-loss modeling and advanced frequency-dependent Steinmetz fitting | Barg et al. (2017)')

@st.cache_data
def sample():
    return pd.read_csv('demo_synthetic_data.csv')

st.sidebar.header('Material dataset')
up = st.sidebar.file_uploader('Upload sinusoidal loss data (CSV)', type='csv')
if up is None:
    df = sample()
    st.sidebar.info('Synthetic demonstration data are loaded. These are NOT measurements from the paper.')
else:
    df = pd.read_csv(up)

fit = None
try:
    fit = fit_material(df)
    st.sidebar.success(f'Fitted {len(fit.frequencies)} frequencies')
except Exception as e:
    st.sidebar.warning(f'Fitting unavailable: {e}')

st.sidebar.download_button('Download CSV template',
    data='frequency_hz,B_peak_T,loss_kW_m3\n100000,0.05,12\n',
    file_name='material_data_template.csv', mime='text/csv')

calc, fitting, comparison, export = st.tabs([
    'Core loss calculator', 'Advanced fitting', 'Model comparison', 'Export & reproducibility'])

with calc:
    col1,col2 = st.columns([1,1.2])
    with col1:
        source = st.radio('Material definition', ['Paper Table I constants', 'Fitted CSV material'])
        mat = st.selectbox('Reference material', list(REFERENCE)) if source=='Paper Table I constants' else None
        freq_khz = st.number_input('Switching frequency (kHz)', min_value=1.0, value=100.0, step=10.0)
        B_mt = st.number_input('Peak flux density (mT)', min_value=1.0, value=100.0, step=10.0)
        D = st.slider('Duty cycle D', 0.02, 0.98, 0.50, step=0.01)
        correction = st.number_input('Square-wave correction C (default π/4)',
                                     min_value=0.01, value=float(DEFAULT_C), step=0.01)
        volume_cm3 = st.number_input('Core volume (cm³)', min_value=0.0, value=0.0, step=0.1,
                                      help='0 = only report volumetric loss.')
    f, B = freq_khz*1e3, B_mt*1e-3
    with col2:
        try:
            if source=='Paper Table I constants':
                k, a, beta = REFERENCE[mat]
                st.warning('Table I coefficients are given for a reference frequency of 100 kHz and 100 °C. '
                           'Predictions at other frequencies using constant parameters are extrapolations.')
            else:
                if fit is None: raise ValueError('Upload suitable CSV data first.')
                k, a, beta = fit.parameters(f)
            models = {'SE (sinusoidal baseline)': se(f,B,k,a,beta),
                      'ISE Eq. 24 (constant parameters)': ise_constant(f,B,D,k,a,beta,correction),
                      'IGSE (triangular B)': igse(f,B,D,k,a,beta)}
            if source=='Fitted CSV material':
                models['ISE Eq. 25 (frequency-dependent)'] = ise_fitted(f,B,D,fit,correction)
            chart = pd.DataFrame({'Model':list(models), 'Loss (kW/m³)':list(models.values())})
            st.plotly_chart(px.bar(chart, x='Model', y='Loss (kW/m³)', title='Predicted loss density'),
                            use_container_width=True)
            st.dataframe(chart.style.format({'Loss (kW/m³)':'{:.4f}'}), hide_index=True)
            if volume_cm3 > 0:
                st.markdown('**Core loss in watts** (calculated from volume)')
                st.dataframe(pd.DataFrame({'Model':list(models),
                        'Core loss (W)':[v*volume_cm3*1e-3 for v in models.values()]}), hide_index=True)
        except Exception as e:
            st.error(str(e))
    st.caption('Assumptions: symmetric peak excursions ±B with piecewise-linear induction; '
               'no DC-bias or temperature correction beyond the calibrated material data. '
               'ISE Eq. 25 requires both effective frequencies to be within the fitted range.')

with fitting:
    st.subheader('Parameter identification from sinusoidal loss curves')
    st.markdown('For each frequency fit **ln(P) = H(f) + β(f) ln(B)**. '
                'Interpolate H and β versus ln(f) with PCHIP; calculate '
                '**α(f) = dH / d ln(f)** and **k(f) = exp[H − α ln(f)]**. '
                'This is a reproducible numerical realization of the paper’s sequential method.')
    if fit:
        st.metric('Training MAPE', f'{fit.mape:.2f}%')
        st.metric('Training RMSE (natural-log loss)', f'{fit.rmse_log:.4f}')
        st.caption('These are in-sample fit diagnostics, not cross-validation scores or evidence of predictive accuracy.')
        st.dataframe(df, hide_index=True)
        coef = fitted_coefficients(fit)
        plot_df = coef.copy(); plot_df['frequency_kHz']=plot_df.frequency_hz/1000
        a,b,c=st.columns(3)
        with a: st.plotly_chart(px.line(plot_df,x='frequency_kHz',y='beta',title='β(f)'),use_container_width=True)
        with b: st.plotly_chart(px.line(plot_df,x='frequency_kHz',y='alpha',title='α(f)'),use_container_width=True)
        with c: st.plotly_chart(px.line(plot_df,x='frequency_kHz',y='k',title='k(f)'),use_container_width=True)
        st.download_button('Export continuous parameter table CSV',
                           coef.to_csv(index=False).encode(), 'fitted_parameters.csv', mime='text/csv')
        st.caption('Potential noise sensitivity: differentiating H to estimate α may amplify measurement noise. '
                   'Review α(f) and validate on withheld measurements before using in design optimization.')

with comparison:
    st.subheader('Measured versus fitted sinusoidal loss')
    if fit:
        plot = df.copy()
        plot['predicted_kW_m3'] = [fit.predict_sine(f,B) for f,B in
                                     zip(plot.frequency_hz,plot.B_peak_T)]
        plot['relative_error_pct'] = 100*(plot.predicted_kW_m3-plot.loss_kW_m3)/plot.loss_kW_m3
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=plot.loss_kW_m3,y=plot.predicted_kW_m3,mode='markers',name='Fitted'))
        hi=float(max(plot.loss_kW_m3.max(),plot.predicted_kW_m3.max()))
        fig.add_trace(go.Scatter(x=[0,hi],y=[0,hi],name='Perfect agreement',mode='lines'))
        fig.update_layout(xaxis_title='Measured loss (kW/m³)',yaxis_title='Predicted loss (kW/m³)')
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(px.scatter(plot,x='frequency_hz',y='relative_error_pct',color='B_peak_T',
                                  title='Training residuals'),use_container_width=True)
        st.download_button('Download predictions and errors',plot.to_csv(index=False).encode(),
                           'fit_predictions.csv',mime='text/csv')
    st.info('To compare Eq. 25 against nonsinusoidal measurements, collect an independent '
            'dataset with columns f, B_peak, D and measured loss. Do not evaluate its '
            'predictive accuracy using the sinusoidal fitting data alone.')

with export:
    st.subheader('Model information and attribution')
    st.write('Source: S. Barg et al., “An Improved Empirical Formulation for Magnetic Core '
             'Losses Estimation Under Nonsinusoidal Induction,” IEEE TPEL, 2017. '
             'DOI: 10.1109/TPEL.2016.2555359.')
    st.write('Model definitions and fitting algorithm are in engine.py; all predictions '
             'use f [Hz], B_peak [T], and loss density [kW/m³].')
    st.write('Reference 3F3 and N67 coefficients are from Table I; demo CSV data are synthetic.')
    st.warning('Prototype for research use. Validate fitted coefficients and loss predictions '
               'against independent measured data before engineering deployment.')
