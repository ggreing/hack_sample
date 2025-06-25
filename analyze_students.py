import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 설정 (Windows: 'Malgun Gothic', macOS: 'AppleGothic', Linux: 'NanumGothic')
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 예시: 서버에서 저장된 학생별 집중도 데이터 파일 경로 (json lines 형식)
DATA_FILE = 'student_attention_log.jsonl'

def load_data(file_path=DATA_FILE):
    """
    저장된 학생별 집중도 데이터 로드 (json lines: 한 줄에 하나의 dict)
    각 dict 예시: {"student_id": "홍길동", "timestamp": 1720000000.0, "attention_score": 0.82}
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(records)

def analyze_attention(df):
    """
    전체 학생별 집중도 통계 분석 및 시각화
    - 시간별(5초 단위) 전체 학생 평균 집중도 선 그래프
    - 학생별 집중도 선 그래프
    - 기존 통계(최근 3분, 전체, TOP5)도 출력
    """
    if df.empty:
        print("데이터가 없습니다.")
        return
    # timestamp를 datetime으로 변환
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    # 5초 단위 구간 생성 (timestamp를 5초 단위로 버림)
    df['time_bin'] = (df['timestamp'] // 5 * 5).astype(int)
    df['time_bin_dt'] = pd.to_datetime(df['time_bin'], unit='s')

    # 1. 시간별 전체 학생 평균 집중도
    time_avg = df.groupby('time_bin_dt')['attention_score'].mean()
    plt.figure(figsize=(10, 4))
    time_avg.plot()
    plt.title('시간별 전체 학생 평균 집중도 (5초 단위)')
    plt.xlabel('시간')
    plt.ylabel('평균 집중도')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # 2. 학생별 집중도 선 그래프
    plt.figure(figsize=(10, 5))
    for sid, g in df.groupby('student_id'):
        plt.plot(g['time_bin_dt'], g['attention_score'], marker='o', label=sid)
    plt.title('학생별 집중도 변화 (5초 단위)')
    plt.xlabel('시간')
    plt.ylabel('집중도')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. 각 학생별 개별 집중도 선 그래프
    for sid, g in df.groupby('student_id'):
        plt.figure(figsize=(8, 3))
        plt.plot(g['time_bin_dt'], g['attention_score'], marker='o')
        plt.title(f'{sid} 집중도 변화 (5초 단위)')
        plt.xlabel('시간')
        plt.ylabel('집중도')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    # 기존 통계
    now = df['dt'].max()
    recent = df[df['dt'] >= now - pd.Timedelta(seconds=180)]
    print("[Recent 3 min Avg Attention Score]")
    recent_avg = recent.groupby('student_id')['attention_score'].mean().round(3)
    print(recent_avg)
    print("\n[Ovaerall Stats]")
    stats = df.groupby('student_id')['attention_score'].agg(['mean','min','max','count']).round(3)
    print(stats)
    print("\n[TOP 5]")
    top5 = df.groupby('student_id')['attention_score'].mean().sort_values(ascending=False).head(5)
    print(top5)

    # 시간별 전체 평균 반환 (Gemini용)
    return time_avg

def main():
    df = load_data()
    time_avg = analyze_attention(df)
    # Gemini 분석용 summary 저장 (선택)
    time_avg.to_csv('time_avg_for_gemini.csv', encoding='utf-8')

if __name__ == "__main__":
    main()
