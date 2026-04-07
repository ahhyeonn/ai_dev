import streamlit as st
import pandas as pd
import ollama
import os
import numpy as np
import cv2  # OpenCV 
import json # JSON 파싱
from PIL import Image
import io 
from appUtils import load_ocr

SAVE_FILE = "business_cards.csv"

# --- [OpenCV Helper Functions] ---

def order_points(pts):
    """
    4개의 꼭짓점 좌표를 (좌상, 우상, 우하, 좌하) 순서로 정렬하는 함수.
    이 수학적 계산이 있어야 삐뚤어진 이미지를 반듯하게 펼 수 있어.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # x+y 합이 가장 작은 게 좌상(top-left), 가장 큰 게 우하(bottom-right)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # x-y 차이가 가장 작은 게 우상(top-right), 가장 큰 게 좌하(bottom-left)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_card_contour(image):
    """
    이미지에서 가장 큰 네모 형태(명함 윤곽선)를 찾아내는 함수. (업그레이드 버전)
    """
    # 1. 전처리: 흑백 변환 -> 블러(노이즈 제거)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 일반 가우시안 블러보다 외곽선을 잘 유지하면서 노이즈를 없애는 Bilateral Filter 사용
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. 에지 검출 (Canny): 민감도를 낮춰서 연한 테두리도 잡아내도록 설정 (기존 75, 200 -> 30, 150)
    edged = cv2.Canny(blurred, 30, 150)

    # 3. [핵심 추가] 모폴로지 닫기(Closing) 연산
    # 끊어진 선들을 팽창시켰다가 다시 수축시켜서 선을 하나로 꽉 이어줌
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 4. 윤곽선 찾기 및 크기 순 정렬
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # 윤곽선 근사화: 곡선을 직선으로 단순화
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 꼭짓점이 4개고, 면적이 너무 작지 않은 경우에만 명함으로 간주
        if len(approx) == 4 and cv2.contourArea(approx) > 5000:
            return approx
            
    return None

def perspective_transform(image, pts):
    """
    4개의 꼭짓점을 기준으로 삐뚤어진 이미지를 정면으로 반듯하게 펴는 함수 (Warp)
    """
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect

    # 펼쳐질 이미지의 너비와 높이 계산 (피타고라스 정리 사용)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 변환 대상 좌표 (반듯한 직사각형)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 변환 행렬 계산 및 적용 (Warp)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- [Core Pipeline Function] ---

# 캐시 대상에서 intermediate images를 제외하기 위해, 원본 bytes와 data만 반환하도록 구조 조정
@st.cache_data
def process_business_card_cached(img_bytes):
    reader = load_ocr() # 공통 기능에서 OCR 불러오기
    
    # [OpenCV 핵심 파이프라인 가동]
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    original_copy = img_cv.copy() # 원본 복사본

    # 1단계: 윤곽선 찾기
    card_cnt = get_card_contour(img_cv)
    
    # 2단계: 자르고 펴기 (Warp)
    if card_cnt is not None:
        # 윤곽선이 발견되면 자르고 폄
        warped = perspective_transform(original_copy, card_cnt)
    else:
        # 발견 안 되면 원본 그대로 사용 (안전장치)
        warped = original_copy

    # 3단계: OpenCV 전처리 (흑백 및 선명도 보정)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.adaptiveThreshold(
        gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 텍스트 교정
    resized_img = cv2.resize(processed_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = reader.readtext(resized_img, detail=0, paragraph=True) 
    raw_text = "\n".join(result) # 공백 대신 줄바꿈으로 구분

    # 4단계: EasyOCR 텍스트 추출
    result = reader.readtext(processed_img, detail=0)
    raw_text = " ".join(result)
    
    # 5단계: LLM 정보 추출 (JSON 포맷 강제)
    prompt = f"""
    당신은 명함 관리 전문가입니다. 제공된 [OCR 텍스트]는 노이즈가 많고 오타가 섞여 있을 수 있습니다.
    문맥을 분석하여 정보를 정확하게 추출하고 오타를 수정하세요.

    지침:
    1. 이름: 한국인 이름 3글자 혹은 영문 이름을 찾으세요.
    2. 전화번호: '010', '02', '044' 등으로 시작하는 번호를 찾으세요. (예: 010-8456-6451)
    3. 회사명: '공사', '주식회사', 'Communications', 'Systems' 등 기업 키워드를 참고하세요.
    4. 모든 값은 한국어로 자연스럽게 교정하세요. (예: 'IYWMC' -> '아이와이엠씨')
    
    [텍스트]
    {raw_text}

    JSON 형식으로 응답하세요 (keys: 회사명, 이름, 직급, 전화번호, 이메일, 비고).
    """
    response = ollama.generate(model='llama3.1', prompt=prompt, format='json', options={'temperature': 0})
    parsed_text = response['response']
    
    # JSON 파싱 및 데이터 정리
    try:
        data_dict = json.loads(parsed_text)
    except json.JSONDecodeError:
        data_dict = {}
    default_keys = ["회사명", "이름", "직급", "전화번호", "이메일", "비고"]
    for key in default_keys:
        if key not in data_dict:
            data_dict[key] = ""
            
    return raw_text, data_dict

# 시각화를 위해 캐싱되지 않는 처리 함수 (이미지 전처리 로직이 중복되지만 UX를 위해)
def process_business_card_visualization(img_bytes):
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    original_copy = img_cv.copy()

    # 가시적 확인을 위한 단계별 결과물 저장 딕셔너리
    stages = {}

    # 1. 윤곽선 찾기 이미지
    card_cnt = get_card_contour(img_cv)
    img_with_cnt = original_copy.copy()
    if card_cnt is not None:
        cv2.drawContours(img_with_cnt, [card_cnt], -1, (0, 255, 0), 3) # 초록색 선
    stages['1_contour'] = img_with_cnt

    # 2. 자르고 펴기 (Warp) 이미지
    if card_cnt is not None:
        warped = perspective_transform(original_copy, card_cnt)
    else:
        warped = original_copy
    stages['2_warped'] = warped

    # 3. OpenCV 전처리 이미지
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.adaptiveThreshold(
        gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    stages['3_preprocessed'] = processed_img

    return stages

# --- [Streamlit UI Main Function] ---

def run(selected_name="전아현"):
    st.title("명함 스캐너_LLM")
    st.caption("openCV easyOCR Llama")

    upload_img = st.file_uploader("명함 사진 업로드", type=['png', 'jpg', 'jpeg'], key="llm_uploader")

    if upload_img is not None:
        img_bytes = upload_img.getvalue()
        
        # 새로운 이미지가 올라오면 캐시된 분석 함수 실행 (백엔드 처리)
        if "llm_last_img" not in st.session_state or st.session_state.llm_last_img != img_bytes:
            with st.spinner("이미지 분석 중"):
                raw_text, parsed_dict = process_business_card_cached(img_bytes)
                st.session_state.llm_last_img = img_bytes
                st.session_state.llm_raw_text = raw_text
                st.session_state.llm_input_company = parsed_dict.get("회사명", "")
                st.session_state.llm_input_name = parsed_dict.get("이름", "")
                st.session_state.llm_input_title = parsed_dict.get("직급", "")
                st.session_state.llm_input_phone = parsed_dict.get("전화번호", "")
                st.session_state.llm_input_email = parsed_dict.get("이메일", "")
                st.session_state.llm_input_note = parsed_dict.get("비고", "")
                st.session_state.llm_is_editing = False

        # --- [가시적 확인 영역: 여기가 핵심!] ---
        st.divider()
        st.subheader("AI 명함 분석")
        
        with st.status("단계별 이미지 전처리 과정 확인", expanded=True):
            # 시각화 전용 함수 호출 (캐싱 안 됨)
            with st.spinner("OpenCV로 변환 중"):
                stages = process_business_card_visualization(img_bytes)
            
            # Streamlit 컬럼을 이용해 이미지를 쪼로록 배치
            col_raw, col_cnt, col_warp, col_pre = st.columns(4)
            
            with col_raw:
                st.image(img_bytes, caption="원본 이미지", use_container_width=True)
            with col_cnt:
                st.image(stages['1_contour'], caption="1. 윤곽선 인식", use_container_width=True)
            with col_warp:
                st.image(stages['2_warped'], caption="2. 명함이미지 재구성", use_container_width=True)
            with col_pre:
                st.image(stages['3_preprocessed'], caption="3. OpenCV 전처리", use_container_width=True)

            # 4. EasyOCR 인식한 글자들
            st.write("---")
            st.write("**4. EasyOCR Raw Text**")
            st.code(st.session_state.get("llm_raw_text", ""), language="text")

            # 5. LLM에서 추출한 글자 (JSON)
            st.write("**5. LLM(Llama3.1)이 구조화한 정보 (JSON)**")
            # 세션 스테이트에 있는 값을 기반으로 임시 딕셔너리 생성해서 표시
            final_json = {
                "회사명": st.session_state.get("llm_input_company", ""),
                "이름": st.session_state.get("llm_input_name", ""),
                "직급": st.session_state.get("llm_input_title", ""),
                "전화번호": st.session_state.get("llm_input_phone", ""),
                "이메일": st.session_state.get("llm_input_email", ""),
                "비고": st.session_state.get("llm_input_note", "")
            }
            st.json(final_json)
        
        st.divider()

        # --- [최종 정보 입력 및 저장 영역: 기존과 동일] ---
        # (시각화 결과물 아래에 UI 배치)
        col_ui = st.container()
        with col_ui:
            st.subheader("명함 정보 최종 확인 및 수정")
            btn_col1, btn_col2, _ = st.columns([2, 2, 6])
            
            with btn_col1:
                if st.button("수정", key="llm_btn_edit", use_container_width=True):
                    st.session_state.llm_is_editing = True
                    st.rerun()
            with btn_col2:
                if st.button("저장", key="llm_btn_save", use_container_width=True):
                    st.session_state.llm_is_editing = False
                    st.rerun()

            read_only = not st.session_state.llm_is_editing
            
            st.text_input("회사명", key="llm_input_company", disabled=read_only)
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.text_input("이름", key="llm_input_name", disabled=read_only)
            with row1_col2:
                st.text_input("직급", key="llm_input_title", disabled=read_only)
                
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.text_input("전화번호", key="llm_input_phone", disabled=read_only)
            with row2_col2:
                st.text_input("이메일", key="llm_input_email", disabled=read_only)
                
            st.text_input("비고", key="llm_input_note", disabled=read_only)

        st.divider()

        if st.button("이 정보를 명함첩(엑셀)에 저장하기", key="llm_btn_excel", type="primary"):
            final_data = {
                "등록사원": selected_name,
                "회사명": st.session_state.llm_input_company,
                "이름": st.session_state.llm_input_name,
                "직급": st.session_state.llm_input_title,
                "전화번호": st.session_state.llm_input_phone,
                "이메일": st.session_state.llm_input_email,
                "비고": st.session_state.llm_input_note
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
    # appMain.py에서 호출하겠지만, 직접 실행 테스트용
    run()