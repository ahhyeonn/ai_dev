import streamlit as st
import ollama
import urllib.parse

def run(selected_name):
    st.title("점심 메뉴 & 맛집 추천기")
    st.caption(f"{selected_name} 님의 기분과 위치에 맞춰 AI가 메뉴를 고르고 식당을 찾아줍니다.")

    col1, col2 = st.columns(2)
    with col1:
        office_address = st.text_input("현재 위치 (사무실 주소)", value="대평동")
    with col2:
        user_mood = st.text_input("오늘 땡기는 음식이나 기분은?", placeholder="예: 비 오니까 국물 요리가 먹고 싶다.")

    if st.button("점심 메뉴 추천 & 맛집 찾기", type="secondary"):
        if not user_mood:
            st.warning("오늘의 기분이나 먹고 싶은 스타일을 조금이라도 적어주세요!")
        else:
            with st.spinner("최고의 메뉴를 고민하는 중..."):
                prompt = f"""
                당신은 센스 있는 점심 메뉴 추천 전문가입니다.
                디온컴즈의 {selected_name} 직원이 점심 메뉴를 고민하고 있습니다.
                직원이 남긴 메모: '{user_mood}'
                
                이 메모를 바탕으로 오늘 먹기 딱 좋은 '음식 종류(메뉴)' 딱 1가지를 강력하게 추천해주세요.
                식당 상호명은 절대 말하지 말고, 일반적인 음식 이름(예: 마라탕, 김치찌개, 돈까스)만 말하세요.
                그리고 왜 추천하는지 재치 있고 짧게 2~3줄로 설명해주세요.
                
                [답변 양식]
                추천 메뉴: (여기에 음식 이름만 딱 적으세요.)
                
                추천 이유: (설명)
                 """
                
                response = ollama.generate(
                    model='llama3.1', 
                    prompt=prompt, 
                    options={'temperature': 0.7}
                )
                
                ai_answer = response['response']
                
            st.success("오늘의 추천 메뉴입니다.")
            st.markdown(ai_answer)
            
            st.divider()
            st.subheader("내 주변 맛집 바로 보기")
            
            suggested_menu = "맛집" 
            for line in ai_answer.split('\n'):
                if "추천 메뉴" in line or "추천메뉴" in line:
                    suggested_menu = line.split(":", 1)[-1].replace('*', '').strip()
                    break
            
            search_keyword = f"{office_address} {suggested_menu}"
            encoded_keyword = urllib.parse.quote(search_keyword)
            
            map_html = f"""
            <iframe 
                width="100%" 
                height="450" 
                frameborder="0" 
                style="border:0; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" 
                src="https://maps.google.com/maps?q={encoded_keyword}&z=14&output=embed" 
                allowfullscreen>
            </iframe>
            """
            
            st.info(f"주변의 '{suggested_menu}' 식당을 확인하세요")
            st.components.v1.html(map_html, height=470)
            
            naver_map_url = f"https://map.naver.com/p/search/{search_keyword}"
            st.link_button(f"🔍 네이버 지도로 더 자세히 보기 (새 창 열기)", naver_map_url)