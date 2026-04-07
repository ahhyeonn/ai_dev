import streamlit as st
import pandas as pd
import ollama
import os
import numpy as np
import cv2
import json
import io
from PIL import Image, ImageDraw, ImageFont
from appUtils import load_ocr

SAVE_FILE = "business_cards.csv"

# --- [OpenCV: 명함 영역 찾기 및 자르기] ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_card_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edged = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 5000:
            return approx
    return None

def perspective_transform(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- [UI용 바운딩 박스 그리기 함수] ---
def draw_boxes_on_image(cv_image, ocr_results):
    """
    OpenCV로 잘라낸 이미지(cv_image) 위에 EasyOCR이 찾은 글자 영역을 그립니다.
    """
    # OpenCV의 BGR 포맷을 PIL이 좋아하는 RGB 포맷으로 변환
    img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    detected_texts = []
    
    for idx, (bbox, text, prob) in enumerate(ocr_results):
        p0, p1, p2, p3 = bbox
        draw.polygon([
            (p0[0], p0[1]), (p1[0], p1[1]), 
            (p2[0], p2[1]), (p3[0], p3[1])
        ], outline="lime", width=3)
        
        text_bg_box = [p0[0], p0[1]-20, p0[0]+25, p0[1]]
        draw.rectangle(text_bg_box, fill="red")
        draw.text((p0[0]+5, p0[1]-18), str(idx+1), fill="white")
        
        detected_texts.append(f"[{idx+1}] {text}")
        
    return img, detected_texts

# --- [Streamlit UI Main Function] ---
def run(selected_name="전아현"):
    st.title("명함 스캐너 (명함 인식 + 선택 추출)")
    st.caption("1. AI가 명함 영역을 잘라냅니다. 2. 원하는 글자만 골라냅니다.")

    upload_img = st.file_uploader("명함 사진 업로드", type=['png', 'jpg', 'jpeg'], key="inter_uploader")

    if upload_img is not None:
        img_bytes = upload_img.getvalue()
        
        if "inter_ocr_results" not in st.session_state or st.session_state.get("inter_last_img") != img_bytes:
            with st.spinner("명함 영역을 찾고 글자를 인식하는 중..."):
                # 1. OpenCV 이미지로 변환
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                original_copy = img_cv.copy()
                
                # 2. 명함 윤곽선 찾아서 자르기 (Warp)
                card_cnt = get_card_contour(img_cv)
                if card_cnt is not None:
                    warped_img = perspective_transform(original_copy, card_cnt)
                    st.toast("명함 영역을 성공적으로 잘라냈습니다!", icon="✂️")
                else:
                    warped_img = original_copy # 못 찾으면 원본 유지
                    st.toast("명함 영역을 찾지 못해 원본으로 진행합니다.", icon="⚠️")
                
                # 3. 잘라낸 이미지로 OCR 텍스트 검출
                reader = load_ocr()
                
                # 글자를 더 잘 찾기 위해 이미지를 1.5배 확대
                resized_warped = cv2.resize(warped_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                
                # detail=1 로 박스 좌표까지 가져오기
                results = reader.readtext(resized_warped, detail=1) 
                
                st.session_state.inter_last_img = img_bytes
                st.session_state.inter_warped_img = resized_warped # 화면에 보여줄 잘라낸 이미지 저장
                st.session_state.inter_ocr_results = results
                st.session_state.inter_selected_text = ""
                st.session_state.inter_llm_done = False

        st.divider()
        
        col_img, col_select = st.columns([1.5, 1])
        
        with col_img:
            st.subheader("1. 잘라낸 명함 및 인식된 글자")
            # 잘라낸(Warped) 이미지 위에 박스 그리기
            boxed_img, text_list = draw_boxes_on_image(
                st.session_state.inter_warped_img, 
                st.session_state.inter_ocr_results
            )
            st.image(boxed_img, use_container_width=True)
            
        with col_select:
            st.subheader("2. 추출할 정보 선택")
            st.write("명함에 필요한 정보만 체크하세요.")
            
            selected_items = []
            for t in text_list:
                if st.checkbox(t, value=False): 
                    clean_text = t.split("] ", 1)[1]
                    selected_items.append(clean_text)
            
            st.write("---")
            if st.button("선택한 텍스트로 정보 정리하기", type="primary", use_container_width=True):
                if not selected_items:
                    st.warning("선택된 글자가 없습니다!")
                else:
                    raw_text = "\n".join(selected_items)
                    
                    with st.spinner("LLM이 JSON으로 구조화하는 중..."):
                        prompt = f"""
                        당신은 명함 관리 전문가입니다. 사용자가 직접 선택한 [명함 텍스트]를 바탕으로 정보를 추출하세요.
                        
                        지침:
                        1. 이름, 회사명, 직급, 전화번호, 이메일, 비고를 추출하세요.
                        2. 오타가 있다면 한국어 문맥에 맞게 교정하세요.
                        3. 전화번호는 xxx-xxxx-xxxx 형식에 맞추세요. (ex. 123.4567.6789 라고 추출했다면 123-4567-6789로 교정하세요.)
                        
                        [명함 텍스트]
                        {raw_text}

                        JSON 형식으로 응답하세요 (keys: 회사명, 이름, 직급, 전화번호, 이메일, 비고).
                        """
                        response = ollama.generate(model='llama3.1', prompt=prompt, format='json', options={'temperature': 0})
                        
                        try:
                            parsed_dict = json.loads(response['response'])
                        except:
                            parsed_dict = {}
                            
                        st.session_state.inter_input_company = parsed_dict.get("회사명", "")
                        st.session_state.inter_input_name = parsed_dict.get("이름", "")
                        st.session_state.inter_input_title = parsed_dict.get("직급", "")
                        st.session_state.inter_input_phone = parsed_dict.get("전화번호", "")
                        st.session_state.inter_input_email = parsed_dict.get("이메일", "")
                        st.session_state.inter_input_note = parsed_dict.get("비고", "")
                        st.session_state.inter_llm_done = True
                        st.rerun()

        if st.session_state.get("inter_llm_done", False):
            st.divider()
            st.subheader("3. 최종 정보 확인 및 저장")
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.inter_input_company = st.text_input("회사명", value=st.session_state.inter_input_company)
                st.session_state.inter_input_name = st.text_input("이름", value=st.session_state.inter_input_name)
                st.session_state.inter_input_title = st.text_input("직급", value=st.session_state.inter_input_title)
            with col2:
                st.session_state.inter_input_phone = st.text_input("전화번호", value=st.session_state.inter_input_phone)
                st.session_state.inter_input_email = st.text_input("이메일", value=st.session_state.inter_input_email)
                st.session_state.inter_input_note = st.text_input("비고", value=st.session_state.inter_input_note)
                
            if st.button("명함첩에 저장하기", key="inter_save_btn"):
                final_data = {
                    "등록사원": selected_name,
                    "회사명": st.session_state.inter_input_company,
                    "이름": st.session_state.inter_input_name,
                    "직급": st.session_state.inter_input_title,
                    "전화번호": st.session_state.inter_input_phone,
                    "이메일": st.session_state.inter_input_email,
                    "비고": st.session_state.inter_input_note
                }
                new_df = pd.DataFrame([final_data])
                if os.path.exists(SAVE_FILE):
                    existing_df = pd.read_csv(SAVE_FILE)
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    updated_df = new_df
                updated_df.to_csv(SAVE_FILE, index=False, encoding='utf-8-sig')
                st.success("명함이 엑셀에 성공적으로 저장되었습니다!")
                st.dataframe(updated_df)

if __name__ == "__main__":
    run()