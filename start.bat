@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ╔═══════════════════════════════════════════════╗
echo ║     心语罗盘 - AI 社交智能助手启动工具        ║
echo ╚═══════════════════════════════════════════════╝
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo ✓ Python 已安装
python --version

REM 检查 CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  未检测到 NVIDIA GPU
) else (
    echo ✓ NVIDIA GPU 已检测
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

echo.

REM 激活虚拟环境（如果存在）
if exist "qwen_env\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call qwen_env\Scripts\activate.bat
)

REM 检查模型是否已下载
if not exist "qwen_models" (
    echo.
    echo ⚠️  检测到模型未下载
    set /p download_choice="是否现在下载模型？(y/n): "
    if "!download_choice!"=="y" (
        python download_model.py
    ) else (
        echo 请先运行 'python download_model.py' 下载模型
        pause
        exit /b 1
    )
)

echo.
echo 启动后端服务...
echo 访问 http://localhost:8000/docs 查看 API 文档
echo 打开 index.html 使用前端界面
echo.
echo 按 Ctrl+C 停止服务
echo ─────────────────────────────────────────────

python main.py

pause
