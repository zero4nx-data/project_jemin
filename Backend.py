import os
import pdfplumber
import google.generativeai as genai
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import platform
import re
import numpy as np
import pandas as pd
from datetime import datetime

# ✅ Gemini API configuration

# 하코딩 되어있는 API키를 실제 배포 시에는 환경변수로 관리하는 편이 안전하다.

genai.configure(api_key='AIzaSyA4vIEpJTIPtrzuC2JfRD-bM6_HbCOdE8k') # 여기에 실제 API 키 입력
model = genai.GenerativeModel("gemini-2.5-flash")

# ✅ Matplotlib 한글 폰트 설정 (그래프 라벨은 영어로 하므로, 폰트 깨짐 문제는 없을 것이나, 시스템 폰트 호환성 위함)
# 운영체제 구분
if platform.system() == 'Darwin': # Mac OS
    plt.rcParams['font.family'] = 'AppleGothic' # 한글 폰트 (UI는 한글, 그래프는 영어)
    plt.rcParams['axes.unicode_minus'] = False # 음수 부호 깨짐 방지
elif platform.system() == 'Windows': # Windows OS
    plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 (UI는 한글, 그래프는 영어)
    plt.rcParams['axes.unicode_minus'] = False # 음수 부호 깨짐 방지
else: # Linux 등
    # 시스템에 나눔고딕 폰트 설치 필요 (예: sudo apt-get install fonts-nanum-extra)
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # Fallback to a generic font if NanumGothic is not found
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False


# ✅ Company Name Mapping (Korean to English & Ticker)
# 사용자가 입력한 한글기업명을 매핑해 주가 조회용 ticker와 그래프,프롬프트 용 englishname
COMPANY_INFO = {
    "삼성전자": {"ticker": "005930.KS", "english_name": "Samsung Electronics"},
    "네이버": {"ticker": "035420.KS", "english_name": "NAVER"},
    "카카오": {"ticker": "035720.KS", "english_name": "Kakao"},
    "LG에너지솔루션": {"ticker": "373220.KS", "english_name": "LG Energy Solution"},
    "현대차": {"ticker": "005380.KS", "english_name": "Hyundai Motor Company"},
    "SK하이닉스": {"ticker": "000660.KS", "english_name": "SK Hynix"},
    "삼성바이오로직스": {"ticker": "207940.KS", "english_name": "Samsung Biologics"},
    "셀트리온": {"ticker": "068270.KS", "english_name": "Celltrion"},
    "POSCO홀딩스": {"ticker": "005490.KS", "english_name": "POSCO Holdings"},
    "LG화학": {"ticker": "051910.KS", "english_name": "LG Chem"},
    "삼성SDI": {"ticker": "006400.KS", "english_name": "Samsung SDI"},
    "기아": {"ticker": "000270.KS", "english_name": "Kia"},
    "하나금융지주": {"ticker": "086790.KS", "english_name": "Hana Financial Group"},
    "KB금융": {"ticker": "105560.KS", "english_name": "KB Financial Group"},
    "신한지주": {"ticker": "055550.KS", "english_name": "Shinhan Financial Group"}
}


# ✅ PDF 첫 페이지에서 텍스트 추출
def extract_first_page_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) > 0:
                text = pdf.pages[0].extract_text()
                # Attempt to extract date from filename for better context
                # Example filename: '20231026_삼성전자_리포트.pdf' or '20240101_Report.pdf'
                filename = os.path.basename(pdf_path)
                date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
                report_date = date_match.group(0) if date_match else "N/A"
                
                # Prepend date to text for potential LLM context, but mainly for target price extraction
                return f"[{report_date}] {text.strip()}" if text else ""
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return ""

# ✅ 주가 추세 그래프 생성 및 Base64 인코딩
def generate_stock_price_graph(ticker, company_name_eng, period="5y"):
    try:
        stock_data = yf.download(ticker, period=period, progress=False)
        if stock_data.empty:
            return None, "❌ 해당 티커에 대한 주가 데이터를 찾을 수 없습니다."

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Close'], label='Close Price', color='skyblue', linewidth=2)
        plt.title(f'{company_name_eng} ({ticker}) Stock Price Trend over {period}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price (KRW)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64, None
    except Exception as e:
        return None, f"❌ 주가 그래프를 생성하는 중 오류가 발생했습니다: {e}"

# ✅ 목표주가 추출 및 분포 그래프 생성 함수
def generate_target_price_data_and_graph(pdf_texts_list, company_name_eng):
    extracted_data = [] # To store dicts like {'date': 'YYYYMMDD', 'price': 120000}
    
    # Enhanced patterns to find target prices.
    # Prioritize patterns with explicit units.
    # Look for '목표주가', 'TP', '적정주가', 'target price' (case insensitive)
    # The regex ensures that the number is not followed by Korean characters,
    # helping to avoid capturing unrelated numbers.
    
    # Pattern 1: '원' or 'KRW' unit (e.g., '120,000원', '120000 KRW')
    pattern_won = r'(?:목표주가|TP|적정주가|target price)[\s:]*([\d,]+)\s*(?:원|KRW)\b'

    # Pattern 2: '만' or '만원' unit (e.g., '12만', '120만원')
    pattern_man = r'(?:목표주가|TP|적정주가|target price)[\s:]*([\d,]+)\s*만(?:원)?\b'
    
    # Pattern 3: Standalone number without explicit unit (e.g., '120,000' directly after keyword)
    # This pattern should be less greedy and consider word boundaries or line breaks.
    # It attempts to capture a number that IS a target price, not just any number.
    pattern_no_unit = r'(?:목표주가|TP|적정주가|target price)[\s:]*([\d,]+)\b(?!\s*(?:년|월|일|점|배|시|분|초|억|조))' 
    # Negative lookahead to avoid capturing dates, ratios, times, or large units like '억', '조'
    
    all_patterns = [pattern_won, pattern_man, pattern_no_unit]

    for text_with_date in pdf_texts_list:
        # Extract date prepended by extract_first_page_text
        date_match = re.match(r'\[(\d{8})\]', text_with_date)
        report_date = date_match.group(1) if date_match else "N/A"
        
        # Remove the prepended date string to avoid interfering with price extraction
        clean_text = re.sub(r'^\[\d{8}\]\s*', '', text_with_date)

        found_price_in_this_report = False
        for pattern in all_patterns:
            # Use finditer to get all matches and their span, or match to get only first
            # Here, we'll try to get the first valid price from each report for simplicity
            match_iter = re.finditer(pattern, clean_text, re.IGNORECASE)
            
            for match in match_iter:
                potential_price_str = match.group(1) # The captured number string
                clean_p = potential_price_str.replace(',', '') # Remove commas
                
                try:
                    price = int(clean_p)
                    # Apply '만' multiplier if the pattern was for '만' units
                    # Check the matched string, not just the pattern name for '만'
                    if '만' in match.group(0) or '만원' in match.group(0): # Check if '만' or '만원' was part of the full match
                         price *= 10000
                    
                    # Basic validation: filter out unrealistically low or high prices (e.g., 0, or too high)
                    if price > 1000 and price < 5_000_000: # Example range, adjust as needed (e.g., min 1,000 KRW, max 5M KRW)
                        extracted_data.append({'Date': report_date, 'Target Price (KRW)': price})
                        found_price_in_this_report = True
                        break # Found a valid price for this report, move to next report
                except ValueError:
                    continue
            if found_price_in_this_report:
                break # If a price was found from any pattern, stop checking for this report

    if not extracted_data:
        # Return empty DataFrame and error for graph
        return pd.DataFrame(columns=['Date', 'Target Price (KRW)']), None, "❌ 리포트에서 유효한 목표주가 데이터를 찾을 수 없습니다."

    # Create DataFrame
    df_target_prices = pd.DataFrame(extracted_data)
    # Sort by date for better visualization (optional but good practice)
    df_target_prices['Date'] = pd.to_datetime(df_target_prices['Date'], errors='coerce')
    df_target_prices.sort_values(by='Date', inplace=True)
    df_target_prices.reset_index(drop=True, inplace=True)

    # ✅ 3-시그마(표준편차) 아웃라이어 제거
    if not df_target_prices.empty:
        prices = df_target_prices['Target Price (KRW)']
        mean_price = prices.mean()
        std_price = prices.std()

        # Check if std_price is zero (all values are same), avoid division by zero
        if std_price > 0:
            # Calculate Z-score
            z_scores = np.abs((prices - mean_price) / std_price)
            # Filter out values with Z-score > 3
            df_target_prices_filtered = df_target_prices[z_scores < 3].copy()
        else:
            df_target_prices_filtered = df_target_prices.copy() # No outliers if std_dev is 0

        # If filtering removed all data, revert to original or handle
        if df_target_prices_filtered.empty and not df_target_prices.empty:
            graph_error_message = "⚠️ 이상치 제거 후 목표주가 데이터가 모두 사라졌습니다. 원본 데이터를 사용합니다."
            df_target_prices_for_graph = df_target_prices
        else:
            df_target_prices_for_graph = df_target_prices_filtered
            graph_error_message = None
    else:
        df_target_prices_for_graph = df_target_prices
        graph_error_message = None


    # Generate graph only if there's data after filtering
    if df_target_prices_for_graph.empty:
        return df_target_prices, None, "❌ 이상치 제거 후 그래프를 생성할 유효한 목표주가 데이터가 없습니다."

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 4)) # Adjust figure size to be more compact

        # ✅ Y-coordinates for visual spread (random jitter)
        y_coords = np.random.normal(0, 0.15, size=len(df_target_prices_for_graph)) # Spread vertically
        
        plt.scatter(df_target_prices_for_graph['Target Price (KRW)'], y_coords, 
                    s=100, alpha=0.7, edgecolors='w', linewidths=0.5, label='Target Price')

        # Set title and labels in English
        plt.title(f'{company_name_eng} Analyst Target Price Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Target Price (KRW)', fontsize=12) # X-axis has target price values
        plt.ylabel('') # Y-axis has no specific label, just for vertical spread
        plt.yticks([]) # Hide Y-axis ticks

        # Set X-axis limits to prevent points from being cut off at the edges
        min_price = df_target_prices_for_graph['Target Price (KRW)'].min()
        max_price = df_target_prices_for_graph['Target Price (KRW)'].max()
        buffer = (max_price - min_price) * 0.1 if (max_price - min_price) > 0 else 10000 # Add buffer, handle zero range
        plt.xlim(min_price - buffer, max_price + buffer)
        
        # Add legend with 'Target Price'
        plt.legend(fontsize=10)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return df_target_prices_for_graph, img_base64, graph_error_message
    except Exception as e:
        return df_target_prices, None, f"❌ 목표주가 분포 그래프를 생성하는 중 오류가 발생했습니다: {e}"

# ✅ 메인 분석 함수
def run_analysis_from_user_input(user_input, base_path):
    company_name_kor = None
    company_name_eng = None
    user_query = None

    # Identify company name (Korean) from user input and get its info
    for name_kor, info in COMPANY_INFO.items():
        if name_kor in user_input:
            company_name_kor = name_kor
            company_name_eng = info["english_name"]
            user_query_parts = user_input.replace(name_kor, "").strip()
            if "에 대한" in user_query_parts:
                user_query = user_query_parts.split("에 대한", 1)[1].strip()
            else:
                user_query = user_query_parts.strip()
            break

    if not company_name_kor:
        return {"error": "❌ 질문에서 분석할 기업을 찾을 수 없거나, 지원되지 않는 기업입니다. 예: '삼성전자에 대한 최근 투자 의견은 어때?'"}, None

    # 주가 그래프 생성
    ticker_symbol = COMPANY_INFO[company_name_kor]["ticker"]
    graph_base64 = None
    graph_error = None
    if ticker_symbol:
        graph_base64, graph_error = generate_stock_price_graph(ticker_symbol, company_name_eng) # Pass English name
    else:
        graph_error = f"❌ '{company_name_kor}'에 대한 티커 정보를 찾을 수 없습니다. 주가 그래프를 생성할 수 없습니다."

    # PDF 분석 로직
    pdf_analysis_result = ""
    df_target_prices_data = pd.DataFrame(columns=['Date', 'Target Price (KRW)']) # Initialize empty DataFrame
    target_price_graph_base64 = None
    target_price_graph_error = None

    company_path = os.path.join(base_path, company_name_kor) # Folder name is still Korean
    pdf_folder = os.path.join(company_path, "pdf")

    if not os.path.isdir(pdf_folder):
        pdf_analysis_result = f"❌ '{company_name_kor}' 폴더를 찾을 수 없습니다. PDF 분석을 수행할 수 없습니다."
    else:
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
        texts = []
        for path in pdf_files:
            txt = extract_first_page_text(path)
            if txt:
                texts.append(txt)

        if not texts:
            pdf_analysis_result = f"❌ '{company_name_kor}'의 PDF에서 유효한 텍스트를 찾지 못했습니다. PDF 분석을 수행할 수 없습니다."
        else:
            combined_text = "\n\n".join(texts) # For LLM prompt (only if needed for context, but not for price extraction)
            
            # Generate target price data (DataFrame) and graph
            # df_target_prices_data will be the filtered DataFrame if outliers are removed
            df_target_prices_data, target_price_graph_base64, target_price_graph_error = \
                generate_target_price_data_and_graph(texts, company_name_eng) # Pass individual texts list and English name


            prompt = f"""
당신은 전문 금융 애널리스트 리포트 분석 AI입니다.

당신의 역할은 다음과 같습니다:
1. 사용자의 질문을 이해하고, 주어진 애널리스트 리포트 텍스트를 기반으로 **명확하고 구조화된 답변**을 생성합니다.
2. 리포트에서 언급된 **수치, 데이터, 추세, 문구** 등을 적극적으로 활용하여 **논리적이고 구체적인 설명**을 제공합니다.
3. 리포트의 **전체적인 톤(긍정/부정/혼재)을 파악**하여 분석에 반영하고, 필요시 **투자 의견 및 목표주가 관련 정보**도 포함합니다.
4. 응답은 **전문가적 문체**로 명확하고 간결하게 작성되어야 하며, 중복되거나 불확실한 표현은 피합니다.
5. 실제 보고서 기반임을 강조하며, 가능한 경우 **보고서 문장에서 발췌한 인용(“”)**을 포함해 주세요.

질문 예시:
- 삼성전자에 대한 최근 리포트의 투자 의견과 목표주가 변동은 어떤가요?
- NAVER의 리스크 요인은 어떤 것들이 언급되었나요?
- 현대차에 대한 긍정적 평가의 근거는 무엇인가요?

---

[사용자 질문]
{company_name_kor}에 대한 질문: "{user_query}"

[애널리스트 리포트 내용]
{combined_text}

---

🎯 **다음 섹션별로 답변을 구조화하여 제공해주세요.**

### 1. 주요 분석 요약
(사용자 질문에 대한 핵심 답변을 2~3문장으로 간결하게 요약)

### 2. 세부 분석 (리포트 내용 기반)
(리포트의 각 사업 부문별 또는 핵심 이슈별 상세 분석. 구체적인 수치, 데이터, 추세, 논조 등을 포함하여 설명)

### 3. 전반적 논조 및 투자 의견
(리포트들의 전반적인 톤(긍정/부정/혼재)을 정리하고, 제시된 투자 의견 및 목표주가 변동 사항을 명확히 제시)

### 4. 핵심 인용 구절
(리포트에서 직접 발췌한 중요한 구절들을 인용 부호와 함께 제시)

### 5. 결론 및 투자자 참고 사항
(종합적인 결론과 함께, 투자자가 주의 깊게 살펴볼 필요가 있는 추가적인 요소나 리스크 요인을 명시)
"""
            try:
                response = model.generate_content(prompt)
                pdf_analysis_result = response.text.strip()
            except Exception as e:
                pdf_analysis_result = f"❌ Gemini 오류: {e}"

    return {
        "pdf_analysis": pdf_analysis_result,
        "graph_image": graph_base64,
        "graph_error": graph_error,
        "target_price_graph_image": target_price_graph_base64,
        "target_price_graph_error": target_price_graph_error,
        "target_price_dataframe": df_target_prices_data # Add DataFrame here
    }, None