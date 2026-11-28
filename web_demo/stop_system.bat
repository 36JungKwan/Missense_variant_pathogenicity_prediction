@echo off
chcp 65001 >nul
title TriMVPP System Shutdown
color 0C

echo ======================================================================
echo    ĐANG TẮT HỆ THỐNG PHÂN TÍCH BIẾN THỂ TRIMVPP
echo ======================================================================
echo.

:: Gọi container theo tên nên chạy được từ bất kỳ thư mục nào. Dữ liệu trong volume được giữ nguyên.
echo [1/2] Đang dừng Database PostgreSQL và Ensembl VEP (Docker)...
docker stop rbnet_postgres >nul 2>&1
if errorlevel 1 (
    echo    * Không dừng được rbnet_postgres: container không chạy hoặc Docker Desktop chưa mở.
) else (
    echo    * Database đã dừng an toàn, dữ liệu được bảo toàn.
)
docker stop rbnet_vep >nul 2>&1
if errorlevel 1 (
    echo    * Không dừng được rbnet_vep: container không chạy hoặc chưa được tạo.
) else (
    echo    * VEP đã dừng.
)
echo.

echo [2/2] Vui lòng đóng 2 cửa sổ dòng lệnh của Backend và Frontend:
echo       - Cửa sổ: "TriMVPP Backend (FastAPI)"
echo       - Cửa sổ: "TriMVPP Frontend (React/Vite)"
echo.
echo Có thể nhấn Ctrl + C trong mỗi cửa sổ hoặc bấm dấu X để tắt.
echo.
echo ======================================================================
echo    HỆ THỐNG ĐÃ ĐƯỢC TẮT HOÀN TOÀN!
echo ======================================================================
echo.
pause
