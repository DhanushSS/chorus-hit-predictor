"""Separate approved HSP-S experiment; the original app/model remain unchanged."""
import streamlit as st
from chorus_hit.hsp_s.ui import render

st.set_page_config(page_title='Hit vs Non-Hit feature study', page_icon='🎵', layout='wide')
render(st)
