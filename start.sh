#!/bin/bash

# 心语罗盘 - 一键启动脚本

set -e

echo "╔═══════════════════════════════════════════════╗"
echo "║     心语罗盘 - AI 社交智能助手启动工具        ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本: $python_version"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU 已检测"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  未检测到 NVIDIA GPU"
fi

echo ""

# 激活虚拟环境（如果存在）
if [ -d "qwen_env" ]; then
    echo "激活虚拟环境..."
    source qwen_env/bin/activate
fi

# 检查模型是否已下载
if [ ! -d "qwen_models" ]; then
    echo ""
    echo "⚠️  检测到模型未下载"
    read -p "是否现在下载模型？(y/n): " download_choice
    if [ "$download_choice" = "y" ]; then
        python download_model.py
    else
        echo "请先运行 'python download_model.py' 下载模型"
        exit 1
    fi
fi

echo ""
echo "启动后端服务..."
echo "访问 http://localhost:8000/docs 查看 API 文档"
echo "访问 index.html 使用前端界面"
echo ""
echo "按 Ctrl+C 停止服务"
echo "─────────────────────────────────────────────"

python main.py
