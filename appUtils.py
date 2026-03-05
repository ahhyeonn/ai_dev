import streamlit as st
import pandas as pd
import os
import easyocr
from pypdf import PdfReader

@st.cache_data
def load_data():
    if os.path.exists("employees.csv"):
        return pd.read_csv("employees.csv")
    return None

def load_text_knowledge():
    file_path = "negotiable.txt"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"파일 읽기 오류: {e}"
    else:
        return "규정집 파일이 없습니다."

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ko', 'en'], gpu=False)