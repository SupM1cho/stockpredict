# utils/cache_utils.py
import streamlit as st
from functools import lru_cache

@st.cache_data
def cache_dataframe(func, *args, **kwargs):
    """
    Cache hasil fungsi yang mengembalikan DataFrame.
    """
    return func(*args, **kwargs)


@lru_cache(maxsize=32)
def cache_api_call(func, *args, **kwargs):
    """
    Cache hasil API call untuk efisiensi.
    """
    return func(*args, **kwargs)
