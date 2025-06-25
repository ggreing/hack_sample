@echo off
echo 🎓 Streamlit 출석 시스템 시작 중...
echo.

:: Python이 설치되어 있는지 확인
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    echo Python을 https://python.org에서 다운로드하여 설치하세요.
    pause
    exit /b 1
)

:: 가상환경 활성화 (있는 경우)
if exist attendance_env\Scripts\activate.bat (
    echo 🐍 가상환경 활성화 중...
    call attendance_env\Scripts\activate.bat
)

:: 필요한 패키지 설치 확인
echo 📦 필요한 패키지 확인 중...
pip install -q streamlit streamlit-webrtc opencv-python mediapipe

:: Streamlit 앱 시작
echo 🌐 Streamlit 앱 시작 중...
echo.
echo 브라우저가 자동으로 열립니다...
echo 수동으로 접속하려면: http://localhost:8501
echo.
streamlit run streamlit_app.py

pause
