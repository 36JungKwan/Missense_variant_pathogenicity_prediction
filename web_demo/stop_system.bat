@echo off
chcp 65001 >nul
title RB-Net System Shutdown
color 0C

echo ======================================================================
echo    ĐANG TẮT HỆ THỐNG PHÂN TÍCH BIẾN THỂ RB-NET
echo ======================================================================
echo.

:: 1. Tắt Docker database container
echo [1/2] Đang dừng Database PostgreSQL (Docker)...
cd /d "F:\Demo"
docker compose stop db
echo -> Database đã dừng an toàn (Dữ liệu trong Volume được bảo toàn).
echo.

:: 2. Thông báo đóng các cửa sổ dòng lệnh
echo [2/2] Vui lòng đóng 2 cửa sổ dòng lệnh của Backend và Frontend:
echo       - Cửa sổ: "RB-Net Backend (FastAPI)"
echo       - Cửa sổ: "RB-Net Frontend (React/Vite)"
echo.
echo Bạn có thể nhấn tổ hợp Ctrl + C trong mỗi cửa sổ hoặc đơn giản là bấm dấu X để tắt.
echo.
echo ======================================================================
echo    HỆ THỐNG ĐÃ ĐƯỢC TẮT HOÀN TOÀN!
echo ======================================================================
echo.
pause
