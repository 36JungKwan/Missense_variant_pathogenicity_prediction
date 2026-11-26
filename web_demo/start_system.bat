@echo off
chcp 65001 >nul
title TriMVPP System Launcher
color 0A

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%backend\.venv\Scripts\python.exe"
cd /d "%ROOT%"

echo ======================================================================
echo    KHỞI ĐỘNG HỆ THỐNG PHÂN TÍCH BIẾN THỂ TRIMVPP (AIP491 CAPSTONE)
echo ======================================================================
echo.

if not exist "%VENV_PY%" (
    echo [LỖI] Không tìm thấy "%VENV_PY%".
    echo Tạo venv trên ổ F: theo hướng dẫn trong backend\requirements.txt rồi chạy lại.
    pause
    exit /b 1
)

:: 1. Database
echo [1/3] Đang khởi động Database PostgreSQL + pgvector qua Docker...
docker compose up -d db
if %errorlevel% neq 0 (
    echo.
    echo [LỖI] Không thể khởi động Docker!
    echo Vui lòng kiểm tra Docker Desktop đã được mở và chạy chưa ^(biểu tượng màu xanh^).
    pause
    exit /b %errorlevel%
)
echo    * Database đang chạy trên cổng 5433 (Container: rbnet_postgres).
echo.

:: 2. Backend FastAPI - không dùng --reload: mỗi lần reload sẽ nạp lại ~2,3 GB Foundation Models lên GPU.
echo [2/3] Đang khởi động Backend FastAPI (Port 8000)...
start "TriMVPP Backend (FastAPI)" cmd /k "cd /d "%ROOT%backend" && set PYTHONUTF8=1&& "%VENV_PY%" -m uvicorn app.main:app --port 8000"
echo    * Backend đang nạp Foundation Models lên GPU (khoảng 15-30 giây) trên cửa sổ riêng.
echo.

:: 3. Frontend Vite + React
echo [3/3] Đang khởi động Frontend React + Vite (Port 5173)...
start "TriMVPP Frontend (React/Vite)" cmd /k "cd /d "%ROOT%frontend" && npm run dev"
echo    * Frontend đã được mở trên cửa sổ riêng (http://localhost:5173).
echo.

echo Đang chờ các dịch vụ sẵn sàng trong 3 giây...
timeout /t 3 /nobreak >nul

echo Đang mở trình duyệt vào hệ thống...
start http://localhost:5173

echo.
echo ======================================================================
echo    HỆ THỐNG ĐÃ SẴN SÀNG!
echo    - Frontend UI:  http://localhost:5173
echo    - Backend Docs: http://localhost:8000/docs
echo    - Database:     localhost:5433 (PostgreSQL + pgvector)
echo ======================================================================
echo.
echo (Bạn có thể đóng cửa sổ launcher này. Hãy giữ 2 cửa sổ Backend và Frontend đang chạy).
pause
