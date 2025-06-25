# 🎓 실시간 온라인 강의 출석 시스템 테스트 환경

AI 기반 얼굴 분석을 활용한 실시간 온라인 강의 출석 시스템의 완전한 테스트 환경입니다.

## 🛠️ 시스템 구성

### 📁 파일 구조
```
attendance-system/
├── streamlit_app.py          # Streamlit 웹 애플리케이션
├── attendance_client.py      # Python 클라이언트 (OpenCV 기반)
├── server.js                 # Socket.IO 서버 (Node.js)
├── package.json              # Node.js 의존성
├── requirements.txt          # Python 의존성
└── README.md                 # 이 파일
```

### 🎯 주요 기능
- **실시간 얼굴 검출**: OpenCV + MediaPipe
- **라이브니스 검출**: 눈 깜박임 패턴 분석
- **머리 자세 추정**: PnP 솔버 기반 3D 추적
- **참여도 점수**: 다중 지표 종합 평가
- **실시간 통신**: Socket.IO 양방향 통신
- **웹 인터페이스**: Streamlit + WebRTC

## 📋 사전 요구사항

### Python 환경
- Python 3.8 이상
- pip 패키지 관리자
- 웹캠 (카메라)

### Node.js 환경
- Node.js 16.0 이상
- npm 8.0 이상

### 시스템 의존성 (Windows)
```bash
# Visual Studio Build Tools 설치 필요 (dlib 컴파일용)
# 또는 아래 명령으로 미리 컴파일된 버전 설치
pip install --upgrade setuptools wheel
pip install dlib
```

### 시스템 의존성 (macOS)
```bash
# Homebrew로 cmake 설치
brew install cmake

# dlib 설치
pip install dlib
```

### 시스템 의존성 (Ubuntu/Linux)
```bash
# 필수 패키지 설치
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev

# dlib 설치
pip install dlib
```

## 🚀 설치 방법

### 1단계: 저장소 클론 (또는 파일 다운로드)
```bash
# 프로젝트 폴더 생성
mkdir attendance-system
cd attendance-system

# 모든 파일을 이 폴더에 저장
```

### 2단계: Python 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv attendance_env

# 가상환경 활성화
# Windows:
attendance_env\Scripts\activate
# macOS/Linux:
source attendance_env/bin/activate

# Python 의존성 설치
python -m pip install --upgrade pip
pip install --upgrade pip
pip install -r requirements.txt
```

### 3단계: Node.js 환경 설정
```bash
# Node.js 의존성 설치
npm install

# 또는 yarn 사용 시:
# yarn install
```

### 4단계: dlib 얼굴 랜드마크 모델 다운로드
```bash
# 모델 파일 다운로드 (약 100MB)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# 압축 해제
# Windows (7-Zip 필요):
# 7z x shape_predictor_68_face_landmarks.dat.bz2

# macOS/Linux:
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# 파일을 프로젝트 폴더에 배치
# shape_predictor_68_face_landmarks.dat
```

## 🖥️ 실행 방법

### 방법 1: Streamlit 웹 애플리케이션 (권장)

1. **서버 시작** (터미널 1):
```bash
# Node.js 서버 실행
npm start

# 또는
node server.js
```

2. **Streamlit 앱 실행** (터미널 2):
```bash
# Python 가상환경 활성화 후
streamlit run streamlit_app.py
```

3. **브라우저에서 접속**:
   - 자동으로 브라우저가 열림 (보통 http://localhost:8501)
   - "START" 버튼 클릭하여 웹캠 시작
   - 카메라 권한 허용

### 방법 2: Python 클라이언트 직접 실행

1. **서버 시작** (터미널 1):
```bash
node server.js
```

2. **Python 클라이언트 실행** (터미널 2):
```bash
python attendance_client.py
```

3. **사용법**:
   - 학생 ID와 이름 입력
   - 웹캠 창에서 실시간 분석 확인
   - 'q' 키로 종료

## 📊 분석 지표

### 출석 상태 등급
- 🟢 **우수 (0.8 이상)**: 매우 집중적 수강
- 🟡 **정상 (0.7 이상)**: 양호한 참여도
- 🟠 **주의 (0.5 이상)**: 산만하거나 부분 참여
- 🔴 **비정상 (0.5 미만)**: 출석 불인정

### 분석 구성요소
- **얼굴 검출** (20%): MediaPipe 기반 얼굴 인식
- **얼굴 인식** (30%): 등록된 학생과의 일치 여부
- **생체 활동성** (15%): 눈 깜박임 패턴 분석
- **주의집중도** (25%): 머리 자세 기반 시선 추적
- **머리 자세** (10%): Yaw, Pitch, Roll 각도 측정

## 🔧 설정 옵션

### 서버 설정 (server.js)
```javascript
const PORT = process.env.PORT || 3000;  // 포트 변경
```

### 클라이언트 설정 (attendance_client.py)
```python
SERVER_URL = "http://localhost:3000"  # 서버 주소 변경
```

### Streamlit 설정 (streamlit_app.py)
```python
SERVER_URL = "http://localhost:3000"  # 서버 주소 변경
```

## 🐛 문제 해결

### 1. dlib 설치 오류
```bash
# Windows에서 Visual Studio Build Tools 설치 후 재시도
pip install --upgrade setuptools wheel
pip install cmake
pip install dlib

# 또는 conda 사용
conda install -c conda-forge dlib
```

### 2. MediaPipe 설치 오류
```bash
# Python 버전 확인 (3.8-3.11 지원)
python --version

# 업그레이드 후 재설치
pip install --upgrade mediapipe
```

### 3. 웹캠 접근 오류
- 브라우저에서 카메라 권한 허용
- 다른 애플리케이션이 카메라를 사용 중인지 확인
- 웹캠 드라이버 업데이트

### 4. Socket.IO 연결 오류
```bash
# 서버가 실행 중인지 확인
curl http://localhost:3000

# 방화벽 설정 확인
# 포트 3000 허용
```

### 5. 성능 최적화
```python
# 프레임 처리 간격 조정 (attendance_client.py)
time.sleep(0.05)  # 20 FPS로 낮춤

# 얼굴 검출 신뢰도 조정 (streamlit_app.py)
min_detection_confidence=0.7  # 높은 정확도
min_detection_confidence=0.3  # 높은 감도
```

## 📈 사용 시나리오

### 교수자용
1. 서버를 시작하고 강의실 ID 생성
2. 학생들에게 접속 정보 제공
3. 실시간 출석 현황 모니터링
4. 출석 데이터 분석 및 관리

### 학생용
1. 제공받은 서버 주소로 접속
2. 학생 ID와 이름 입력
3. 웹캠 권한 허용
4. 강의 수강 중 자동 출석 체크

## 🔒 개인정보 보호

- **로컬 처리 우선**: 대부분의 분석이 클라이언트에서 수행
- **원본 영상 비저장**: 실시간 분석 후 즉시 삭제
- **암호화 통신**: 분석 결과만 암호화하여 전송
- **최소 데이터**: 출석 확인에 필요한 최소한의 정보만 수집

## 📞 지원 및 기여

### 이슈 리포팅
- 버그 발견 시 상세한 오류 메시지와 함께 보고
- 개선 사항이나 새로운 기능 제안 환영

### 개발 참여
- 코드 개선 및 최적화
- 새로운 분석 알고리즘 추가
- 다국어 지원 및 UI/UX 개선

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

### ⚠️ 주의사항
- 이 시스템은 교육 및 연구 목적으로 제작되었습니다
- 상업적 사용 시 관련 법규 및 윤리 가이드라인을 준수하세요
- 얼굴 인식 데이터의 수집과 처리에 대한 사전 동의를 받으세요
- 개인정보 보호법 및 관련 규정을 준수하세요

**즐거운 테스트 되세요! 🚀**
