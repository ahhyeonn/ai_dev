import streamlit as st
import ollama
import os
from datetime import datetime

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 문서를 쪼개서 Vector DB 만들기
@st.cache_resource
def build_vector_db():
    file_path = "negotiable.txt"
    if not os.path.exists(file_path):
        return None
    
    # 1-1. 문서 읽기
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    
    # 1-2. 문서를 500글자씩 쪼개기 (문맥이 끊기지 않게 50글자씩 겹치게 설정)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # 1-3. 글자를 숫자(임베딩)로 변환해서 Chroma DB에 저장
    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore
# ---------------RAG 적용--------------------

def run(selected_name, emp_info):
    st.title("사용설명서 챗봇")
    st.caption(f"현재 접속자: {selected_name} {emp_info['직급']} (입사일: {emp_info['입사일']})")

    # DB 구축 (한 번만 실행되고 메모리에 캐싱됨)
    vectorstore = build_vector_db()
    if vectorstore is None:
        st.error("negotiable.txt 파일이 없습니다.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("무엇이든 물어보세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("탐색 중..."):
                
                # 사용자의 질문과 관련 문서 상위 3개 검색
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(prompt)
                
                # 검색된 3개를 합치기
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # 검색 내용을 프롬프트에 insert
                system_prompt = f"""
                당신은 디온컴즈의 인사팀 비서입니다. 
                아래 [검색된 사내 규정] 내용을 바탕으로 답변해주세요.
                없는 내용은 지어내지 말고 해당 내용은 찾을 수 없다고 하세요.
                
                [검색된 사내 규정]: 
                {context_text}
                
                [상담 직원]: {selected_name} {emp_info['직급']} (입사일: {emp_info['입사일']})
                [오늘 날짜]: {datetime.now().strftime('%Y-%m-%d')}
                """
                
                response = ollama.generate(
                    model='llama3.1',
                    prompt=f"{system_prompt}\n\n사용자 질문: {prompt}",
                    options={'temperature': 0}
                )
                answer = response['response']
                st.markdown(answer)
                
                # with st.expander("AI가 참고한 규정 원문 보기"):
                #     st.info(context_text)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})