@echo off
chcp 65001 >nul
title TriMVPP System Launcher
color 0A
setlocal

:: Chạy được từ bất kỳ thư mục nào (ổ F, repo clone...): container được gọi theo TÊN (rbnet_postgres, rbnet_vep)
:: chứ không theo project của Docker Compose, vốn đổi theo tên thư mục và sẽ tạo trùng container / database rỗng.
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%backend\.venv\Scripts\python.exe"
cd /d "%ROOT%"

echo ======================================================================
echo    KHỞI ĐỘNG HỆ THỐNG PHÂN TÍCH BIẾN THỂ TRIMVPP (AIP491 CAPSTONE)
echo ======================================================================
echo.

if not exist "%VENV_PY%" (
    echo [LỖI] Không tìm thấy "%VENV_PY%".
    echo Tạo môi trường Python theo hướng dẫn trong README.md, mục Setup, rồi chạy lại.
    goto fail
)
if not exist "%ROOT%backend\.env" (
    echo [LỖI] Chưa có file backend\.env.
    echo Sao chép backend\.env.example thành backend\.env và sửa các đường dẫn, xem README.md.
    goto fail
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [LỖI] Docker Desktop chưa chạy. Mở Docker Desktop, đợi biểu tượng chuyển xanh rồi chạy lại.
    goto fail
)

:: 1. Database PostgreSQL + pgvector
echo [1/4] Đang khởi động Database PostgreSQL + pgvector...
docker container inspect rbnet_postgres >nul 2>&1
if errorlevel 1 goto create_db
docker start rbnet_postgres >nul
if errorlevel 1 goto db_fail
goto wait_db_start

:create_db
:: Lần đầu trên máy này: tạo container từ docker-compose.yml, mật khẩu đọc từ file .env cạnh docker-compose.yml.
if not exist "%ROOT%.env" (
    echo [LỖI] Chưa có file .env cạnh docker-compose.yml.
    echo Sao chép .env.example thành .env và đặt POSTGRES_PASSWORD, xem README.md.
    goto fail
)
docker compose up -d db
if errorlevel 1 goto db_fail

:wait_db_start
set /a TRIES=0
:wait_db
set "DB_STATE="
for /f "delims=" %%s in ('docker inspect -f "{{.State.Health.Status}}" rbnet_postgres 2^>nul') do set "DB_STATE=%%s"
if "%DB_STATE%"=="healthy" goto db_ready
set /a TRIES+=1
if %TRIES% geq 60 goto db_slow
timeout /t 1 /nobreak >nul
goto wait_db

:db_slow
echo    * Database chưa sẵn sàng sau 60 giây; Backend sẽ báo lỗi kết nối cho tới khi Database chạy xong.
goto vep

:db_ready
echo    * Database đang chạy trên cổng 5433 (container rbnet_postgres).
goto vep

:db_fail
echo [LỖI] Không khởi động được container rbnet_postgres. Xem thông báo của Docker ở trên.
goto fail

:: 2. Ensembl VEP (chú giải biến thể mới)
:vep
echo.
echo [2/4] Đang khởi động Ensembl VEP...
docker container inspect rbnet_vep >nul 2>&1
if errorlevel 1 goto vep_missing
docker start rbnet_vep >nul
if errorlevel 1 goto vep_missing
echo    * VEP đang chạy (container rbnet_vep).
goto backend

:vep_missing
echo [CẢNH BÁO] Không khởi động được container rbnet_vep, xem README.md mục Setup.
echo            Biến thể đã có trong Database vẫn tra được; biến thể mới sẽ báo lỗi VEP.

:: 3. Backend FastAPI - không dùng --reload: mỗi lần reload sẽ nạp lại ~2,3 GB Foundation Models lên GPU.
:backend
echo.
echo [3/4] Đang khởi động Backend FastAPI (Port 8000)...
start "TriMVPP Backend (FastAPI)" cmd /k "cd /d "%ROOT%backend" && set PYTHONUTF8=1&& "%VENV_PY%" -m uvicorn app.main:app --port 8000"
echo    * Backend đang nạp Foundation Models lên GPU (khoảng 15-30 giây) trên cửa sổ riêng.
echo.

:: 4. Frontend Vite + React
echo [4/4] Đang khởi động Frontend React + Vite (Port 5173)...
if not exist "%ROOT%frontend\node_modules" (
    echo    * Lần chạy đầu: đang cài thư viện Frontend bằng npm install...
    pushd "%ROOT%frontend"
    call npm install
    popd
)
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
echo (Có thể đóng cửa sổ này. Hãy giữ 2 cửa sổ Backend và Frontend đang chạy.)
pause
exit /b 0

:fail
echo.
pause
exit /b 1
