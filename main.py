import streamlit as st
import math
from scipy.stats.distributions import norm
import numpy as np
from streamlit_extras.colored_header import colored_header 
from streamlit_extras.switch_page_button import switch_page


def main():
    st.title("Welcome to the Option Pricing Calculator!")

    # Introduction section
    st.markdown(
        """
        Option pricing is crucial in finance for making informed decisions in the market. 
        Whether you're exploring European options or breaking through barriers, 
        our calculator empowers you to evaluate different option strategies with ease.
        """
    )

    # Pink line under "Get Started"
    st.markdown('<hr style="border: 2px solid #FF4B4B;">', unsafe_allow_html=True)

    # Display the buttons container with adjusted left margin
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    col4, col1, col2, col3 = st.columns([2.15, 1 , 1, 2.4])
    with col1:
        if st.button("Price Barrier Options"):
            st.session_state.page = "Price Barrier Options"
    with col2:
        if st.button("Price European Options"):
            st.session_state.page = "Price European Options"
    st.markdown('</div>', unsafe_allow_html=True)

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
   
    # Fun finance quote at the end
    st.markdown("> \"The stock market is filled with individuals who know the price of everything, but the value of nothing.\" - Phillip Fisher")

    # Render the selected page
    if "page" not in st.session_state:
        st.session_state.page = "Home"  # Default to home page

    if st.session_state.page == "Price Barrier Options":
        barrier_option_page()
    elif st.session_state.page == "Price European Options":
        european_option_page()

def barrier_option_page():
    st.header('Barrier Option Pricing Calculator')

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
    
    def Heston_Model(S0, K, T, r, q, v0, kappa, theta, sigma, rho, num_simulations, num_time_steps, H, Phi, Nu, Pos):
        # Generate random numbers for Monte Carlo simulation
        np.random.seed(42)
        z1 = np.random.normal(size=(num_simulations, num_time_steps))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=(num_simulations, num_time_steps))

        # Simulate stock price paths using Heston model
        dt = T / num_time_steps
        vt = np.zeros_like(z1)
        vt[:, 0] = v0
        St = np.zeros_like(z1)
        St[:, 0] = S0

        for i in range(1, num_time_steps):
            vt[:, i] = vt[:, i - 1] + kappa * (theta - vt[:, i - 1]) * dt + sigma * np.sqrt(np.maximum(0, vt[:, i - 1] * dt)) * z2[:, i]
            St[:, i] = St[:, i - 1] * np.exp((r - q - 0.5 * vt[:, i]) * dt + np.sqrt(np.maximum(0, vt[:, i] * dt)) * z1[:, i])


        if Nu ==-1 :
            if Pos == -1 :
                max_paths = np.max(St, axis=1)
                payoff = np.maximum(Phi * St[:, -1] - Phi * K, 0)    
                price = np.where(max_paths < H, payoff, 0)
            else :
                max_paths = np.max(St, axis=1)
                payoff = np.maximum(Phi * St[:, -1] - Phi * K, 0)    
                price = np.where(max_paths > H, payoff, 0)
        else :
            if Pos == -1 : 
                min_paths = np.min(St, axis=1)
                payoff = np.maximum(Phi * St[:, -1] - Phi * K, 0)    
                price = np.where(min_paths > H, payoff, 0)
            else :
                min_paths = np.min(St, axis=1)
                payoff = np.maximum(Phi * St[:, -1] - Phi * K, 0)    
                price = np.where(min_paths < H, payoff, 0)

        mean_price = np.mean(price)
        std_error = np.std(price) / np.sqrt(num_simulations)  # Standard error of the mean
        z_score = 1.96  # Z-score for 95% confidence interval
        conf_interval = (mean_price - z_score * std_error, mean_price + z_score * std_error)

        value = mean_price * np.exp(-r * T)

        return value, conf_interval
    


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
    b = st.sidebar.number_input('Dividend yield Rate (q)', value=0.0, step=0.01)
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
        kappa = 2.0
        v0 =0.1
        theta = 0.1
        rho = -0.5

        # Calculation
        Formule_Fermée = bsm_barrier_option(X, S, H, b, T, r, Sigma, K, Pos, Phi, Nu)
        Monte_Carlo, confidence_interval = mc_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, H, K, Nu, Phi, Pos)
        barrier_option_price, conf_interval = Heston_Model(S0, K, T, r, q, v0, kappa, theta, sigma, rho, num_simulations, num_time_steps, H, Phi, Nu, Pos)

        # Display results
        st.write('Black-Scholes Closed-Form:', Formule_Fermée)
        st.write('Black-Scholes Monte-Carlo:', Monte_Carlo)
        st.write('Confidence Interval (95%):', confidence_interval)
        st.write("Heston Model : ", barrier_option_price)
        st.write("Confidence Interval (95%):", conf_interval)



def european_option_page():
    st.header('European Option Pricing Calculator')
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
        
    def SVM_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, Phi, kappa, theta, rho, v0):
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
        # Generate random numbers for Monte Carlo simulation
        np.random.seed(42)
        z1 = np.random.normal(size=(N_simulation, time_steps))
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=(N_simulation, time_steps))

        # Simulate stock price paths using Heston model
        dt = T / time_steps
        vt = np.zeros_like(z1)
        vt[:, 0] = v0
        St = np.zeros_like(z1)
        St[:, 0] = S

        # Calculate European option prices for each simulation path
        option_prices = np.zeros((N_simulation, time_steps))
        for i in range(1, time_steps):
            vt[:, i] = vt[:, i - 1] + kappa * (theta - vt[:, i - 1]) * dt + Sigma * np.sqrt(np.maximum(0, vt[:, i - 1] * dt)) * z2[:, i]
            St[:, i] = St[:, i - 1] * np.exp((r - b - 0.5 * vt[:, i]) * dt + np.sqrt(np.maximum(0, vt[:, i] * dt)) * z1[:, i])
            payoffs = np.maximum(Phi * St[:, i] - Phi * X, 0)  
            option_prices[:, i] = payoffs * np.exp(-r * T)

        # Calculate European option price
        european_option_price = np.mean(option_prices[:, -1])

        # Calculate confidence interval for European call option
        mean = np.mean(option_prices[:, -1])
        std_error = np.std(option_prices[:, -1]) / np.sqrt(N_simulation)
        z_score = 1.96  # Z-score for 95% confidence interval
        conf_interval = (mean - z_score * std_error, mean + z_score * std_error)

        return european_option_price, conf_interval
    

    np.random.seed(42)


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
        v0 = 0.1        
        kappa = 2.0     
        theta = 0.1     
        rho = -0.5     
        

        # Calculate and print the European option prices
        option_price_mc, lower_bound_european, upper_bound_european = mc_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, Phi)
        option_price_bs = bsm_barrier_option(X, S, b, T, r, Sigma, Phi)
        european_option_price, conf_interval = SVM_barrier_option(S, T, r, X, b, Sigma, time_steps, N_simulation, Phi, kappa, theta, rho, v0)


        # Display results
        st.write('Black-Scholes Closed-Form:', option_price_bs)
        st.write('Black-Scholes Monte-Carlo:', option_price_mc)
        st.write('Confidence Interval (95%):',({lower_bound_european}, {upper_bound_european}))
        st.write("Heston Model : ", european_option_price)
        st.write("Confidence Interval (95%):", conf_interval)


if __name__ == "__main__":
    main()
