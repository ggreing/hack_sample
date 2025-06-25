# 🎓 실시간 온라인 강의 출석 시스템 - 상세 설치 가이드

## 📋 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [사전 준비](#사전-준비)
3. [단계별 설치](#단계별-설치)
4. [실행 방법](#실행-방법)
5. [문제 해결](#문제-해결)

## 🔧 시스템 요구사항

### 하드웨어
- **CPU**: Intel i5 이상 또는 AMD 동급
- **RAM**: 최소 8GB (권장 16GB)
- **저장공간**: 최소 2GB 여유공간
- **웹캠**: HD 해상도 이상 권장
- **네트워크**: 안정적인 인터넷 연결

### 소프트웨어
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 ~ 3.11 (3.12는 일부 라이브러리 호환성 이슈)
- **Node.js**: 16.0 이상
- **브라우저**: Chrome, Firefox, Safari (WebRTC 지원)

## 🛠️ 사전 준비

### 1. Python 설치
```bash
# Python 공식 웹사이트에서 다운로드
https://www.python.org/downloads/

# 설치 시 "Add Python to PATH" 옵션 체크 필수
```

### 2. Node.js 설치
```bash
# Node.js 공식 웹사이트에서 LTS 버전 다운로드
https://nodejs.org/

# 설치 확인
node --version
npm --version
```

### 3. Git 설치 (선택사항)
```bash
https://git-scm.com/downloads
```

## 📦 단계별 설치

### 1단계: 프로젝트 설정
```bash
# 프로젝트 폴더 생성
mkdir attendance-system
cd attendance-system

# 제공된 모든 파일을 이 폴더에 복사
```

### 2단계: Python 환경 구성
```bash
# 가상환경 생성
python -m venv attendance_env

# 가상환경 활성화
# Windows:
attendance_env\Scripts\activate
# macOS/Linux:
source attendance_env/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

### 3단계: Python 의존성 설치
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# 개별 설치 (문제 발생 시)
pip install streamlit>=1.30.0
pip install streamlit-webrtc>=0.47.0
pip install opencv-python>=4.8.0
pip install mediapipe>=0.10.0
pip install python-socketio[client]>=5.8.0
pip install numpy>=1.24.0
pip install pillow>=9.5.0
```

### 4단계: dlib 설치 (중요!)
```bash
# Windows (Visual Studio Build Tools 필요)
pip install cmake
pip install dlib

# macOS
brew install cmake
pip install dlib

# Ubuntu/Linux
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
```

### 5단계: face_recognition 설치
```bash
# dlib 설치 후
pip install face_recognition
```

### 6단계: Node.js 의존성 설치
```bash
# package.json이 있는 폴더에서
npm install

# 또는 개별 설치
npm install express socket.io cors dotenv
```

### 7단계: 얼굴 랜드마크 모델 다운로드
```bash
# 모델 파일 다운로드 (약 100MB)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Windows에서는 브라우저로 직접 다운로드
# 압축 해제 후 프로젝트 폴더에 배치
# 최종 파일명: shape_predictor_68_face_landmarks.dat
```

## 🚀 실행 방법

### 방법 1: Streamlit 웹 애플리케이션

#### 터미널 1 - 서버 시작
```bash
# Node.js 서버 실행
node server.js

# 또는 Windows 배치 파일
start_server.bat
```

#### 터미널 2 - Streamlit 앱
```bash
# Python 가상환경 활성화
source attendance_env/bin/activate  # macOS/Linux
# attendance_env\Scripts\activate   # Windows

# Streamlit 앱 실행
streamlit run streamlit_app.py

# 또는 Windows 배치 파일
start_streamlit.bat
```

#### 브라우저 사용
1. http://localhost:8501 접속
2. "START" 버튼 클릭
3. 카메라 권한 허용
4. 실시간 분석 확인

### 방법 2: Python 클라이언트 직접 실행

#### 터미널 1 - 서버 시작
```bash
node server.js
```

#### 터미널 2 - Python 클라이언트
```bash
# 가상환경 활성화 후
python attendance_client.py

# 또는 Windows 배치 파일
start_client.bat
```

## 🔍 설치 확인

### Python 패키지 확인
```python
# test_packages.py
import cv2
import mediapipe as mp
import socketio
import streamlit as st
import dlib
import face_recognition
import numpy as np

print("✅ 모든 패키지가 정상적으로 설치되었습니다!")
```

### Node.js 패키지 확인
```javascript
// test_server.js
const express = require('express');
const socketio = require('socket.io');
console.log('✅ Node.js 패키지가 정상적으로 설치되었습니다!');
```

## ⚠️ 문제 해결

### 1. dlib 설치 실패
```bash
# Windows - Visual Studio Build Tools 설치
https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 또는 conda 사용
conda install -c conda-forge dlib

# 또는 사전 컴파일된 버전
pip install https://github.com/sachadee/Dlib/raw/master/dlib-19.22.99-cp39-cp39-win_amd64.whl
```

### 2. MediaPipe 설치 실패
```bash
# Python 버전 확인 (3.8-3.11만 지원)
python --version

# 캐시 클리어 후 재설치
pip cache purge
pip install --no-cache-dir mediapipe
```

### 3. OpenCV 에러
```bash
# 기존 설치 제거 후 재설치
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.0.76
```

### 4. Streamlit WebRTC 에러
```bash
# 호환 가능한 버전 설치
pip install streamlit==1.30.0
pip install streamlit-webrtc==0.47.0
```

### 5. Socket.IO 연결 실패
```bash
# 방화벽 설정 확인
# Windows: Windows Defender 방화벽에서 포트 3000 허용
# macOS: 시스템 환경설정 > 보안 및 개인정보보호
# Linux: ufw allow 3000
```

### 6. 웹캠 접근 오류
- 브라우저 설정에서 카메라 권한 허용
- 다른 애플리케이션이 카메라 사용 중인지 확인
- HTTPS 환경에서는 SSL 인증서 필요

### 7. 성능 최적화
```python
# CPU 사용률이 높은 경우
# streamlit_app.py에서 프레임 처리 간격 조정
if self.frame_count % 10 == 0:  # 매 10프레임마다 분석

# 얼굴 검출 신뢰도 조정
min_detection_confidence=0.7  # 높은 정확도, 낮은 감도
min_detection_confidence=0.3  # 낮은 정확도, 높은 감도
```

## 📞 추가 지원

### 로그 확인
```bash
# Python 로그
python -c "import cv2, mediapipe, dlib; print('OK')"

# Node.js 로그
node -e "console.log('Node.js 정상 작동')"

# Streamlit 디버그 모드
streamlit run streamlit_app.py --server.enableXsrfProtection=false
```

### 버전 호환성 확인
```bash
# 권장 버전 조합
Python: 3.9.16
OpenCV: 4.8.0.76
MediaPipe: 0.10.0
dlib: 19.24.0
Streamlit: 1.30.0
Node.js: 18.17.0
```

**설치 완료 후 README.md 파일의 실행 방법을 참조하여 시스템을 테스트하세요! 🚀**