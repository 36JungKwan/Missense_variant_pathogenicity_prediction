@echo off
chcp 65001 >nul
title RB-Net System Launcher
color 0A

echo ======================================================================
echo    KHỞI ĐỘNG HỆ THỐNG PHÂN TÍCH BIẾN THỂ RB-NET (AIP491 CAPSTONE)
echo ======================================================================
echo.

:: 1. Kiểm tra và khởi động Database
echo [1/3] Đang khởi động Database PostgreSQL + pgvector qua Docker...
cd /d "F:\Demo"
docker compose up -d db
if %errorlevel% neq 0 (
    echo.
    echo [LỖI] Không thể khởi động Docker!
    echo Vui lòng kiểm tra Docker Desktop đã được mở và chạy chưa (biểu tượng màu xanh).
    pause
    exit /b %errorlevel%
)
echo -> Database đang chạy trên cổng 5433 (Container: rbnet_postgres).
echo.

:: 2. Khởi động Backend FastAPI
echo [2/3] Đang khởi động Backend FastAPI (Port 8000)...
start "RB-Net Backend (FastAPI)" cmd /k "cd /d F:\Demo\backend && set PYTHONUTF8=1&& python -m uvicorn app.main:app --reload --port 8000"
echo -> Backend đã được mở trên cửa sổ riêng (http://localhost:8000).
echo.

:: 3. Khởi động Frontend Vite + React
echo [3/3] Đang khởi động Frontend React + Vite (Port 5173)...
start "RB-Net Frontend (React/Vite)" cmd /k "cd /d F:\Demo\frontend && npm run dev"
echo -> Frontend đã được mở trên cửa sổ riêng (http://localhost:5173).
echo.

:: Chờ vài giây để các service sẵn sàng rồi mở trình duyệt
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
