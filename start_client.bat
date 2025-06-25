@echo off
echo 🎥 Python 클라이언트 시작 중...
echo.

:: Python이 설치되어 있는지 확인
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    pause
    exit /b 1
)

:: 가상환경 활성화 (있는 경우)
if exist attendance_env\Scripts\activate.bat (
    call attendance_env\Scripts\activate.bat
)

:: 필요한 패키지 설치 확인
echo 📦 필요한 패키지 확인 중...
pip install -q opencv-python mediapipe python-socketio[client]

:: 클라이언트 시작
echo 🎥 출석 클라이언트 시작 중...
echo 서버가 실행 중인지 확인하세요.
echo.
python attendance_client.py

pause
