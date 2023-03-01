import streamlit as st


def init_state(key: str, default_value: any):
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]