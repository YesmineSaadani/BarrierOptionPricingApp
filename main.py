import streamlit as st
from Barrier_option import barrier_option_page
from European_option import European_option_page

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
        European_option_page()

if __name__ == "__main__":
    main()


