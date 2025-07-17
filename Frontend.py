import streamlit as st
from Backend import run_analysis_from_user_input
import os
import base64
import pandas as pd

# Report storage location
BASE_PATH = "Report_file"

# ✅ Page configuration
st.set_page_config(page_title="기업 리포트 분석 AI", layout="wide") # UI 제목 다시 한글로

# ✅ Header
st.markdown("## 📊 AI 기반 기업 리포트 분석 서비스") # UI 헤더 다시 한글로
st.caption("금융 애널리스트 리포트와 주가 추이를 기반으로 기업 관련 질문에 대해 LLM이 답변합니다.") # UI 캡션 다시 한글로

# ✅ 사용자 입력
example_questions = [
    "삼성전자에 대한 최근 투자 의견은?",
    "네이버에 대한 리스크 요인은?",
    "카카오의 목표주가 분포는 어떻게 돼?",
    "LG에너지솔루션에 대한 리포트 분위기는 어때?",
    "현대차의 5년 주가 추이는 어때?",
    "삼성전자의 5년 주가와 리포트 분석 해줘",
    "삼성전자 목표주가 분포는 어때?"
]

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("❓ 기업 질문 입력", placeholder="예: 삼성전자에 대한 최근 전망은?") # UI 입력 필드 다시 한글로
with col2:
    selected_example = st.selectbox("📌 예시 질문", [""] + example_questions) # UI 예시 질문 다시 한글로
    if selected_example and not user_input:
        user_input = selected_example

# ✅ 실행 버튼
if st.button("🔍 분석 실행"): # UI 버튼 다시 한글로
    if user_input.strip() == "":
        st.warning("⛔ 질문을 입력하거나 예시를 선택해 주세요.") # UI 경고 메시지 다시 한글로
    else:
        with st.spinner("📚 애널리스트 리포트 및 주가 데이터를 분석 중입니다..."): # UI 스피너 다시 한글로
            result, _ = run_analysis_from_user_input(user_input, BASE_PATH)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("✅ 분석 완료!") # UI 성공 메시지 다시 한글로

            # --- 결과 표시 섹션 ---
            st.markdown("### 📊 시각화 분석") # UI 섹션 제목 다시 한글로

            # 주가 그래프와 목표주가 그래프를 나란히 배치
            graph_col1, graph_col2 = st.columns(2)

            with graph_col1:
                st.subheader("📈 Stock Price Trend") # 그래프 제목은 영어로 유지
                if result.get("graph_image"):
                    st.image(f"data:image/png;base64,{result['graph_image']}", caption="Past 5 Years Stock Price Trend")
                elif result.get("graph_error"):
                    st.warning(result["graph_error"])
                else:
                    st.info("Stock price graph cannot be generated. Check company name or if it's supported.")

            with graph_col2:
                st.subheader("🎯 Target Price Distribution") # 그래프 제목은 영어로 유지
                if result.get("target_price_graph_image"):
                    st.image(f"data:image/png;base64,{result['target_price_graph_image']}", caption="Analyst Target Price Distribution")
                elif result.get("target_price_graph_error"):
                    st.warning(result["target_price_graph_error"])
                else:
                    st.info("Target price data not found in reports or graph cannot be generated.")
            
            # --- 목표주가 데이터 표 ---
            st.markdown("---")
            st.markdown("### 📋 추출된 목표주가 데이터") # UI 섹션 제목 다시 한글로
            df_target_prices = result.get("target_price_dataframe")
            if df_target_prices is not None and not df_target_prices.empty:
                st.dataframe(df_target_prices, hide_index=True)
            else:
                st.info("리포트에서 상세 목표주가 데이터를 추출할 수 없었습니다.") # UI 정보 메시지 다시 한글로

            st.markdown("---")
            st.markdown("### 📝 리포트 분석 결과") # UI 섹션 제목 다시 한글로
            st.markdown(result["pdf_analysis"])