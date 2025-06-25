@echo off
echo 🚀 Socket.IO 서버 시작 중...
echo.

:: Node.js가 설치되어 있는지 확인
node --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js가 설치되지 않았습니다.
    echo Node.js를 https://nodejs.org에서 다운로드하여 설치하세요.
    pause
    exit /b 1
)

:: npm 의존성 설치 (필요한 경우)
if not exist node_modules (
    echo 📦 의존성 설치 중...
    npm install
)

:: 서버 시작
echo 🌐 서버 시작 중... (종료하려면 Ctrl+C)
echo.
node server.js

pause
