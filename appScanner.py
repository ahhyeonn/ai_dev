import streamlit as st
import pandas as pd
import ollama
import os
import numpy as np
from PIL import Image
import io 
from appUtils import load_ocr

SAVE_FILE = "business_cards.csv"

@st.cache_data
def process_business_card(img_bytes):
    reader = load_ocr() # 공통 기능에서 OCR 불러오기
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img)
    
    result = reader.readtext(img_array, detail=0)
    raw_text = " ".join(result)
    
    prompt = f"""
    다음은 명함을 스캔한 텍스트입니다. 텍스트에서 정보를 추출해주세요.
    이름, 회사명, 직급, 전화번호, 이메일을 찾아서 아래 양식에 맞게 값만 적어주세요.
    마크다운 기호 없이 ':' 뒤에 값만 적어주세요.
    못 찾겠으면 '없음'이라고 적으세요.
    
    [텍스트]
    {raw_text}
    
    [양식]
    회사명: 
    이름: 
    직급: 
    전화번호: 
    이메일: 
    비고: 
    """
    response = ollama.generate(
        model='llama3.1', 
        prompt=prompt, 
        options={'temperature': 0}
    )
    parsed_text = response['response']
    
    data_dict = {"회사명": "", "이름": "", "직급": "", "전화번호": "", "이메일": "", "비고": ""}
    for line in parsed_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            clean_key = key.replace('*', '').replace('-', '').strip()
            clean_value = value.replace('*', '').strip()
            for valid_key in data_dict.keys():
                if valid_key in clean_key:
                    data_dict[valid_key] = clean_value
                    break
    return raw_text, data_dict

def run(selected_name):
    st.title("명함 스캐너")
    st.caption("명함 사진을 올리면 AI가 분석합니다. 틀린 부분은 직접 수정하고 저장하세요.")

    upload_img = st.file_uploader("명함 사진 업로드", type=['png', 'jpg', 'jpeg'])

    if upload_img is not None:
        img_bytes = upload_img.getvalue()
        
        col_img, col_ui = st.columns([1, 2])
        with col_img:
            st.image(upload_img, use_container_width=True)
            
        with col_ui:
            if "last_img" not in st.session_state or st.session_state.last_img != img_bytes:
                with st.spinner("AI가 정보를 분석 중입니다..."):
                    raw_text, parsed_dict = process_business_card(img_bytes)
                    st.session_state.last_img = img_bytes
                    
                    st.session_state.raw_text = raw_text
                    
                    st.session_state.input_company = parsed_dict.get("회사명", "")
                    st.session_state.input_name = parsed_dict.get("이름", "")
                    st.session_state.input_title = parsed_dict.get("직급", "")
                    st.session_state.input_phone = parsed_dict.get("전화번호", "")
                    st.session_state.input_email = parsed_dict.get("이메일", "")
                    st.session_state.input_note = parsed_dict.get("비고", "")
                    st.session_state.is_editing = False

            with st.expander("EasyOCR 원본 데이터 확인하기"):
                # 저장해둔 raw_text를 텍스트 박스 형태로 보여줍니다.
                st.code(st.session_state.get("raw_text", ""), language="text")

            st.subheader("명함 정보 확인")
            btn_col1, btn_col2, _ = st.columns([2, 2, 6])
            
            with btn_col1:
                if st.button("수정", use_container_width=True):
                    st.session_state.is_editing = True
                    st.rerun()
            with btn_col2:
                if st.button("저장", use_container_width=True):
                    st.session_state.is_editing = False
                    st.rerun()

            read_only = not st.session_state.is_editing
            
            st.text_input("회사명", key="input_company", disabled=read_only)
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.text_input("이름", key="input_name", disabled=read_only)
            with row1_col2:
                st.text_input("직급", key="input_title", disabled=read_only)
                
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.text_input("전화번호", key="input_phone", disabled=read_only)
            with row2_col2:
                st.text_input("이메일", key="input_email", disabled=read_only)
                
            st.text_input("비고", key="input_note", disabled=read_only)

        st.divider()

        if st.button("이 정보를 명함첩(엑셀)에 저장하기", type="secondary"):
            final_data = {
                "등록사원": selected_name,
                "회사명": st.session_state.input_company,
                "이름": st.session_state.input_name,
                "직급": st.session_state.input_title,
                "전화번호": st.session_state.input_phone,
                "이메일": st.session_state.input_email,
                "비고": st.session_state.input_note
            }
            
            new_df = pd.DataFrame([final_data])
            
            if os.path.exists(SAVE_FILE):
                existing_df = pd.read_csv(SAVE_FILE)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                updated_df = new_df
                
            updated_df.to_csv(SAVE_FILE, index=False, encoding='utf-8-sig')
            st.success("성공적으로 저장되었습니다.")
            st.dataframe(updated_df)