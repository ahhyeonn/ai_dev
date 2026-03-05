import streamlit as st
from appUtils import load_data, load_text_knowledge

import appChatbot
import appScanner
import appLunch

st.set_page_config(page_title="AI", page_icon="", layout="wide")

# ==========================================
# [데이터 초기화 및 사이드바 설정]
# ==========================================
data = load_data()
COMPANY_KNOWLEDGE = load_text_knowledge()

with st.sidebar:
    st.title("AI")
    
    app_mode = st.radio("기능 선택", ["사용설명서 챗봇", "명함 스캐너", "점심 메뉴 추천"])
    st.divider()

    if data is not None:
        employee_list = data['이름'].tolist()
        selected_name = st.selectbox("사원 선택", employee_list, key="selected_user")
        
        if "last_selected_user" not in st.session_state:
            st.session_state.last_selected_user = selected_name
        
        if st.session_state.last_selected_user != selected_name:
            st.session_state.messages = []  
            st.session_state.last_selected_user = selected_name  
            st.rerun() 

        emp_info = data[data['이름'] == selected_name].iloc[0]
    else:
        st.error("employees.csv 파일이 없습니다.")
        st.stop()
        
    st.divider()
    if app_mode == "사용설명서 챗봇" and st.checkbox("텍스트 파일 열어보기"):
        st.text_area("파일 내용", COMPANY_KNOWLEDGE, height=400)


# ==========================================
# [모드별 라우팅]
# ==========================================
if app_mode == "사용설명서 챗봇":
    appChatbot.run(selected_name, emp_info)

elif app_mode == "명함 스캐너":
    appScanner.run(selected_name)

elif app_mode == "점심 메뉴 추천":
    appLunch.run(selected_name)