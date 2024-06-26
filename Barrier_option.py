import streamlit as st
import math
from scipy.stats.distributions import norm
import numpy as np
from streamlit_extras.colored_header import colored_header 
from streamlit_extras.switch_page_button import switch_page 

def barrier_option_page():
    def bsm_barrier_option(X, S, H, b, T, r, Sigma, K, Pos, Phi, Nu):
        """
        Parameters:
        S = initial stock price
        T = t/T = time to maturity
        r = risk-less short rate
        X = strike price
        Sigma = volatility of stock value
        time_steps = the number of path nodes
        N_simulation = the number of simulation
        H = barrier price
        K = Rebate
        Nu：down = 1  up = -1
        Phi: call = 1  put = -1    
        Pos: 1 = in  ； -1 = out
        """
        lamda = (r - b + ((Sigma ** 2) / 2)) / (Sigma ** 2)
        y1 = math.log(H / S, math.e) / (Sigma * math.sqrt(T)) + lamda * Sigma * math.sqrt(T)
        x1 = math.log(S / H, math.e) / (Sigma * math.sqrt(T)) + lamda * Sigma * math.sqrt(T)
        y = math.log((H ** 2) / (S * X), math.e) / (Sigma * math.sqrt(T)) + lamda * Sigma * math.sqrt(T)

        d1 = (np.log(S / X) + (r - b + 0.5 * Sigma**2) * T) / (Sigma * np.sqrt(T))
        d2 = d1 - Sigma * np.sqrt(T)

        EuroC = S * np.exp(-b * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
        EuroP = X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-b * T) * norm.cdf(-d1)

        CUI = S * norm.cdf(x1) * np.exp(-b * T) - X * np.exp(-r * T) * norm.cdf(x1 - Sigma * np.sqrt(T)) - S * np.exp(-b * T) * ((H / S)**(2 * lamda)) * (norm.cdf(-y) - norm.cdf(-y1)) + X * np.exp(-r * T) * ((H / S)**(2 * lamda - 2)) * (norm.cdf(-y + Sigma * np.sqrt(T)) - norm.cdf(-y1 + Sigma * np.sqrt(T)))
        CDO = S * norm.cdf(x1) * np.exp(-b * T) - X * np.exp(-r * T) * norm.cdf(x1 - Sigma * np.sqrt(T)) - S * np.exp(-b * T) * ((H / S)**(2 * lamda)) * (norm.cdf(y1)) + X * np.exp(-r * T) * ((H / S)**(2 * lamda - 2)) * (norm.cdf(y1 - Sigma * np.sqrt(T)))
        CDI = S * np.exp(-b * T) * ((H / S)** (2 * lamda)) * norm.cdf(y) - X * np.exp(-r * T) * ((H / S)** (2 * lamda - 2)) * norm.cdf(y - Sigma * np.sqrt(T))
        PUI = - S * np.exp(-b * T) * ((H / S)**(2 * lamda)) * (norm.cdf(-y)) + X * np.exp(-r * T) * ((H / S)**(2 * lamda - 2)) * (norm.cdf(-y + Sigma * np.sqrt(T)))
        PUO = - S * norm.cdf(-x1) * np.exp(-b * T) + X * np.exp(-r * T) * norm.cdf(-x1 - Sigma * np.sqrt(T)) + S * np.exp(-b * T) * ((H / S)**(2 * lamda)) * (norm.cdf(-y1)) - X * np.exp(-r * T) * ((H / S)**(2 * lamda - 2)) * (norm.cdf(-y1 - Sigma * np.sqrt(T)))
        PDI = - S * norm.cdf(-x1) * np.exp(-b * T) + X * np.exp(-r * T) * norm.cdf(-x1 + Sigma * np.sqrt(T)) + S * np.exp(-b * T) * ((H / S)**(2 * lamda)) * (norm.cdf(y) - norm.cdf(y1)) - X * np.exp(-r * T) * ((H / S)**(2 * lamda - 2)) * (norm.cdf(y - Sigma * np.sqrt(T)) - norm.cdf(y1 - Sigma * np.sqrt(T)))

        
        if Phi == 1 :
            if Nu == -1 :
                if Pos == 1 :
                    if H >= X :
                        Price = CUI
                    else :
                        Price = EuroC
                else :
                    if H >= X : 
                        Price = EuroC - CUI
                    else :
                        Price = 0
                    
            else :
                if Pos == -1 :
                    if H >= X :
                        Price = CDO
                    else :
                        Price = EuroC - CDI
                else :
                    if H >= X :
                        Price = EuroC - CDO
                    else :
                        Price = CDI
                        
        else :
            if Nu == -1 :
                if Pos == 1 :
                    if H >= X :
                        Price = PUI
                    else :
                        Price = EuroP - PUO
                else :
                    if H >= X : 
                        Price = EuroP - PUI
                    else :
                        Price = PUO
                    
            else :
                if Pos == -1 :
                    if H >= X :
                        Price = 0
                    else :
                        Price = EuroP - PDI
                else :
                    if H >= X :
                        Price = EuroP
                    else :
                        Price = PDI
                        
        return Price
        
    np.random.seed(42)

    def mc_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, H, K, Nu, Phi, Pos):
        """
        Parameters:
        S = initial stock price
        T = t/T = time to maturity
        r = risk-less short rate
        X = strike price
        Sigma = volatility of stock value
        time_steps = the number of path nodes
        N_simulation = the number of simulation
        H = barrier price
        K = Rebate
        Nu：down = 1  up = -1
        Phi: call = 1  put = -1    
        Pos: 1 = in  ； -1 = out
        """
        dt = T / time_steps
        barrier_paths = np.zeros((N_simulation, time_steps + 1))
        barrier_paths[:, 0] = S
        for i in range(1, time_steps + 1):
            z = np.random.normal(0, 1, N_simulation)
            barrier_paths[:, i] = barrier_paths[:, i - 1] * np.exp((r - b - 0.5 * Sigma**2) * dt + Sigma * np.sqrt(dt) * z)
        if Nu == -1 :
            if Pos == -1 :

                max_path = np.max(barrier_paths, axis=1)
                payoff = np.maximum(Phi * barrier_paths[:, -1] - Phi * X, 0)
                price = np.where(max_path < H, payoff, K)
            else :
                max_path = np.max(barrier_paths, axis=1)
                payoff = np.maximum(Phi * barrier_paths[:, -1] - Phi * X, 0)
                price = np.where(max_path >= H, payoff, K)
        else :
            if Pos == -1 :
                min_path = np.min(barrier_paths, axis=1)
                payoff = np.maximum(Phi * barrier_paths[:, -1] - Phi * X, 0)
                price = np.where(min_path > H, payoff, K)
            else :
                min_path = np.min(barrier_paths, axis=1)
                payoff = np.maximum(Phi * barrier_paths[:, -1] - Phi * X, 0)
                price = np.where(min_path < H, payoff, K)


        value = np.mean(price) * np.exp(-r * T)

        # Confidence interval calculation
        std_dev = np.std(price)
        z_score = 1.96 
        lower_bound = value - (z_score * std_dev / np.sqrt(N_simulation))
        upper_bound = value + (z_score * std_dev / np.sqrt(N_simulation))

        return value, (lower_bound, upper_bound)

    # Streamlit UI
    colored_header(
    label="Barrier Option Pricing Calculator",
    description="A barrier option is a type of derivative where the payoff depends on whether or not the underlying asset has reached or exceeded a predetermined price. A barrier option can be a knock-out, meaning it expires worthless if the underlying exceeds a certain price. It can also be a knock-in, meaning it has no value until the underlying reaches a certain price. ",
    )
    # Sidebar for user input
    st.sidebar.header('Input Parameters')
    X = st.sidebar.number_input('Strike Price (X)', value=100.0, step=5.0)
    S = st.sidebar.number_input('Initial Stock Price (S)', value=100.0, step=5.0)
    H = st.sidebar.number_input('Barrier Price (H)', value=105.0, step=5.0)
    T = st.sidebar.number_input('Time to Maturity (T)', value=1.0, step=0.1)
    r = st.sidebar.number_input('Risk-free Rate (r)', value=0.08, step=0.01)
    Sigma = st.sidebar.number_input('Volatility (Sigma)', value=0.25, step=0.05)
    K = st.sidebar.number_input('Rebate (K)', value=0.0, step=5.0)
    b = st.sidebar.number_input('Dividend yield Rate (r)', value=0.0, step=0.01)
    Pos = st.sidebar.radio('Position (in/out)', ['In', 'Out'], index=0)
    Pos = 1 if Pos == 'In' else -1
    Nu = st.sidebar.radio('Barrier Direction (up/down)', ['Up', 'Down'], index=0)
    Nu = -1 if Nu == 'Up' else 1
    Phi = st.sidebar.radio('Option Type (call/put)', ['Call', 'Put'], index=0)
    Phi = 1 if Phi == 'Call' else -1

    # Calculate button
    if st.sidebar.button("Calculate!"):
        time_steps = 1890
        N_simulation = 10000

        # Calculation
        Formule_Fermée = bsm_barrier_option(X, S, H, b, T, r, Sigma, K, Pos, Phi, Nu)
        Monte_Carlo, confidence_interval = mc_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, H, K, Nu, Phi, Pos)

        # Display results
        st.write('Black-Scholes Closed-Form:', Formule_Fermée)
        st.write('Black-Scholes Monte-Carlo:', Monte_Carlo)
        st.write('Confidence Interval (95%):', confidence_interval)

barrier_option_page()
