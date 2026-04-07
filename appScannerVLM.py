import streamlit as st
import pandas as pd
import ollama
import os
import json

SAVE_FILE = "business_cards.csv"

def run(selected_name="테스트유저"):
    st.title("명함 스캐너_VLM")
    st.caption("LLaVA")

    # 업로더 key 이름도 vlm_ 달기
    upload_img = st.file_uploader("명함 사진 업로드", type=['png', 'jpg', 'jpeg'], key="vlm_uploader")

    if upload_img is not None:
        img_bytes = upload_img.getvalue()
        
        col_img, col_ui = st.columns([1, 2])
        with col_img:
            st.image(upload_img, use_container_width=True)
            
        with col_ui:
            # key 이름표 달기: vlm_last_img
            if "vlm_last_img" not in st.session_state or st.session_state.vlm_last_img != img_bytes:
                
                with st.status("AI 명함 분석 파이프라인 가동 중...", expanded=True) as status:
                    st.write("1. Vision 모델: 이미지 분석 중...")
                    
                    prompt = """
                    이 명함 이미지를 분석해서 정보를 추출하고, 오직 JSON 형식으로만 응답하세요.
                    응답에 마크다운 기호(```json 등)나 추가 설명을 절대 포함하지 마세요.
                    키(Key)값은 "회사명", "이름", "직급", "전화번호", "이메일", "비고" 로 고정하고, 찾을 수 없는 정보는 빈 문자열("")로 남겨두세요.
                    """
                    
                    response_stream = ollama.chat(
                        model='llava', # VLM 모델
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [img_bytes]
                        }],
                        stream=True, 
                        options={'temperature': 0}
                    )

                    st.write("2. LLM: 정보 추출 및 JSON 작성 중...")
                    
                    text_placeholder = st.empty()
                    full_response = ""

                    for chunk in response_stream:
                        if 'message' in chunk and 'content' in chunk['message']:
                            full_response += chunk['message']['content']
                            text_placeholder.markdown(f"```json\n{full_response}▌\n```")
                    
                    text_placeholder.markdown(f"```json\n{full_response}\n```")
                    status.update(label="분석 완료!", state="complete", expanded=False)

                clean_text = full_response.replace('```json', '').replace('```', '').strip()
                try:
                    parsed_dict = json.loads(clean_text)
                except json.JSONDecodeError:
                    st.error("AI 응답을 JSON으로 변환하는 데 실패했습니다. 수동으로 입력해 주세요.")
                    parsed_dict = {}

                default_keys = ["회사명", "이름", "직급", "전화번호", "이메일", "비고"]
                for key in default_keys:
                    if key not in parsed_dict:
                        parsed_dict[key] = ""

                # VLM 전용 session_state 저장
                st.session_state.vlm_last_img = img_bytes
                st.session_state.vlm_raw_text = clean_text
                st.session_state.vlm_input_company = parsed_dict.get("회사명", "")
                st.session_state.vlm_input_name = parsed_dict.get("이름", "")
                st.session_state.vlm_input_title = parsed_dict.get("직급", "")
                st.session_state.vlm_input_phone = parsed_dict.get("전화번호", "")
                st.session_state.vlm_input_email = parsed_dict.get("이메일", "")
                st.session_state.vlm_input_note = parsed_dict.get("비고", "")
                st.session_state.vlm_is_editing = False

            st.subheader("명함 정보 확인")
            btn_col1, btn_col2, _ = st.columns([2, 2, 6])
            
            with btn_col1:
                if st.button("수정", key="vlm_btn_edit", use_container_width=True):
                    st.session_state.vlm_is_editing = True
                    st.rerun()
            with btn_col2:
                if st.button("저장", key="vlm_btn_save", use_container_width=True):
                    st.session_state.vlm_is_editing = False
                    st.rerun()

            read_only = not st.session_state.vlm_is_editing
            
            # 입력칸(text_input)에도 전부 vlm_ key 적용
            st.text_input("회사명", key="vlm_input_company", disabled=read_only)
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.text_input("이름", key="vlm_input_name", disabled=read_only)
            with row1_col2:
                st.text_input("직급", key="vlm_input_title", disabled=read_only)
                
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.text_input("전화번호", key="vlm_input_phone", disabled=read_only)
            with row2_col2:
                st.text_input("이메일", key="vlm_input_email", disabled=read_only)
                
            st.text_input("비고", key="vlm_input_note", disabled=read_only)

        st.divider()

        if st.button("이 정보를 명함첩(엑셀)에 저장하기", key="vlm_btn_excel", type="primary"):
            final_data = {
                "등록사원": selected_name,
                "회사명": st.session_state.vlm_input_company,
                "이름": st.session_state.vlm_input_name,
                "직급": st.session_state.vlm_input_title,
                "전화번호": st.session_state.vlm_input_phone,
                "이메일": st.session_state.vlm_input_email,
                "비고": st.session_state.vlm_input_note
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

if __name__ == "__main__":
    run("기본사용자")