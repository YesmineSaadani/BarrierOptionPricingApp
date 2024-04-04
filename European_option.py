import streamlit as st
import math
from scipy.stats.distributions import norm
import numpy as np
from streamlit_extras.colored_header import colored_header 
from streamlit_extras.switch_page_button import switch_page 

import streamlit as st

def European_option_page():
    def bsm_barrier_option(X, S, b, T, r, Sigma, Phi):
        d1 = (np.log(S / X) + (r - b + 0.5 * Sigma**2) * T) / (Sigma * np.sqrt(T))
        d2 = d1 - Sigma * np.sqrt(T)

        if Phi == 1:
            option_price = S * np.exp(-b * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
        else:
            option_price = X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-b * T) * norm.cdf(-d1)

        return option_price

    def mc_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, Phi):
        dt = T / time_steps
        paths = np.zeros((N_simulation, time_steps + 1))
        paths[:, 0] = S

        for i in range(1, time_steps + 1):
            z = np.random.normal(0, 1, N_simulation)
            paths[:, i] = paths[:, i - 1] * np.exp((r - b - 0.5 * Sigma**2) * dt + Sigma * np.sqrt(dt) * z)

        if Phi == 1:
            payoff = np.maximum(paths[:, -1] - X, 0)
        else:
            payoff = np.maximum(X - paths[:, -1], 0)

        option_price_mc = np.exp(-r * T) * np.mean(payoff)

        # Calculate the confidence interval for the European Option
        alpha = 0.95
        q = 1.96

        # Calculate the standard error of the mean for European Option
        se_european = np.std(payoff) / np.sqrt(N_simulation)

        # Calculate the confidence interval for the European Option
        lower_bound_european = option_price_mc - q * se_european
        upper_bound_european = option_price_mc + q * se_european

        return option_price_mc, lower_bound_european, upper_bound_european

    np.random.seed(42)

    S = 100
    r = 0.08
    Sigma = 0.25
    b = 0
    T = 1
    X = 100
    time_steps = 1890
    N_simulation = 10000
    Phi = 1


    # Streamlit UI
    colored_header(
        label="European Option Pricing Calculator",
        description="An European option is the type of options contract that allows the option holder to exercise the option only on the expiration date of the option. Option holders have the right but not the obligation to exercise their options. They can also choose not to use the option and let it expire.",
    )
    # Sidebar for user input
    st.sidebar.header('Input Parameters')
    X = st.sidebar.number_input('Strike Price (X)', value=100.0, step=5.0)
    S = st.sidebar.number_input('Initial Stock Price (S)', value=100.0, step=5.0)
    T = st.sidebar.number_input('Time to Maturity (T)', value=1.0, step=0.1)
    r = st.sidebar.number_input('Risk-free Rate (r)', value=0.08, step=0.01)
    Sigma = st.sidebar.number_input('Volatility (Sigma)', value=0.25, step=0.05)
    K = st.sidebar.number_input('Rebate (K)', value=0.0, step=5.0)
    b = st.sidebar.number_input('Dividend yield Rate (r)', value=0.0, step=0.01)
    Phi = st.sidebar.radio('Option Type (call/put)', ['Call', 'Put'], index=0)
    Phi = 1 if Phi == 'Call' else -1
    # Calculate button
    if st.sidebar.button("Calculate!"):
        time_steps = 1890
        N_simulation = 10000

        # Calculate and print the European option prices
        option_price_mc, lower_bound_european, upper_bound_european = mc_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, Phi)
        option_price_bs = bsm_barrier_option(X, S, b, T, r, Sigma, Phi)

        # Display results
        st.write('Black-Scholes Closed-Form:', option_price_bs)
        st.write('Black-Scholes Monte-Carlo:', option_price_mc)
        st.write('Confidence Interval (95%):',({lower_bound_european}, {upper_bound_european}))


    



