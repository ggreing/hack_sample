from google import genai
import os
import pandas as pd

# 환경 변수 또는 직접 API 키 입력
API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAf_Gg8gwuVY7l6H0rdp3qlKAih89T6sCQ')
client = genai.Client(api_key=API_KEY)

# 시간별 전체 학생 평균 데이터 로드
csv_file = 'time_avg_for_gemini.csv'
if not os.path.exists(csv_file):
    print('time_avg_for_gemini.csv 파일이 없습니다. 먼저 analyze_students.py를 실행하세요.')
    exit(1)

time_avg = pd.read_csv(csv_file, index_col=0, parse_dates=True)

# 프롬프트 생성
prompt = f"""
아래는 5초 단위로 집계된 전체 학생들의 시간별 평균 집중도입니다. 시간에 따른 집중도 변화 경향, 집중도가 낮거나 높았던 구간, 전체적인 패턴, 개선 방안 등을 한국어로 분석해 주세요. (시간은 UTC 기준입니다)

{time_avg.to_string()}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
print("[Gemini 시간별 집중도 분석 결과]")
print(response.text)
