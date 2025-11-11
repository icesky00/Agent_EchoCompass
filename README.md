# 心语罗盘 - AI 社交智能助手

<div align="center">

🧭 基于 Qwen2.5 模型的智能情感分析与对话建议系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Qwen](https://img.shields.io/badge/Qwen-2.5-purple.svg)](https://qwenlm.github.io/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-orange.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

## 📖 项目简介

心语罗盘是一款基于阿里通义千问 Qwen2.5 大语言模型的社交智能助手，提供两大核心功能：

- **💭 情感分析**：分析文本的情感类型、计算共情得分、提取兴趣关键词
- **💡 对话建议**：基于对话上下文和关系类型，生成温暖、真诚的回复建议

## ✨ 功能特性

### 情感分析
- 识别 6 种情感类型（快乐、悲伤、愤怒、焦虑、恐惧、中性）
- 计算共情得分（0-100%），评估情绪关怀需求程度
- 自动提取兴趣关键词，构建用户画像
- 提供分析置信度评估

### 对话建议
- 支持多种关系类型（朋友、恋人、家人、同事）
- 基于对话历史智能生成回复建议
- 提供建议理由，帮助理解沟通策略
- 根据情感状态调整建议内容

## 🚀 快速开始

### 系统要求

- **操作系统**：Linux / Windows 10+ / macOS
- **Python**：3.10 或更高版本
- **GPU**：NVIDIA GPU（推荐显存 ≥ 12GB）
- **CUDA**：12.1 或更高版本
- **磁盘空间**：至少 15GB 可用空间（用于模型存储）

### 硬件配置建议

| 配置级别 | GPU 型号 | 显存 | 推荐模型 | 性能 |
|---------|---------|------|---------|------|
| **入门级** | RTX 3060 | 12GB | Qwen2.5-7B-Int4 | 20-25 tokens/s |
| **推荐级** | RTX 4090 | 24GB | Qwen2.5-7B-Int8 | 25-30 tokens/s |
| **专业级** | RTX A6000 | 48GB | Qwen2.5-14B-FP16 | 15-20 tokens/s |

## 📦 安装步骤

### 第 1 步：克隆项目

```bash
# 如果你有 Git 仓库，使用 git clone
# 否则直接使用下载的文件夹
cd qwen_backend
```

### 第 2 步：创建 Python 虚拟环境

```bash
# 使用 conda（推荐）
conda create -n qwen_env python=3.10
conda activate qwen_env

# 或使用 venv
python -m venv qwen_env
source qwen_env/bin/activate  # Linux/Mac
# qwen_env\Scripts\activate  # Windows
```

### 第 3 步：安装依赖

```bash
# 安装所有依赖包（约需 5-10 分钟）
pip install -r requirements.txt

# 验证 CUDA 支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# 应输出: CUDA available: True
```

### 第 4 步：下载 Qwen 模型

```bash
# 运行交互式下载脚本
python download_model.py

# 脚本会提示选择模型：
# 1. Qwen2.5-7B-Instruct-GPTQ-Int8 (推荐) - 显存 10GB
# 2. Qwen2.5-7B-Instruct (完整精度) - 显存 17GB
# 3. Qwen2.5-7B-Instruct-GPTQ-Int4 (激进量化) - 显存 6GB
# 4. Qwen2.5-14B-Instruct-GPTQ-Int8 (高精度) - 显存 18GB

# 建议选择选项 1（默认）
```

**注意**：首次下载约 7-8GB，需要 10-20 分钟（取决于网速）。

### 第 5 步：启动后端服务

```bash
python main.py

# 看到以下输出表示成功：
# ✓ 模型加载完成
# ✓ 运行设备: cuda:0
# ✓ 显存占用: 9.87 GB
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 第 6 步：打开前端界面

```bash
# 方法 1：直接用浏览器打开
# 双击 index.html 文件

# 方法 2：启动本地服务器（推荐）
python -m http.server 8000
# 访问 http://localhost:8000/static/frontend.html

# 方法 3：如果在服务器运行
# 建议添加穿透插件如open in default browser
# 访问 http://localhost:xxxx(前端端口号)/static/frontend.html

```

## 📚 使用指南

### 情感分析使用示例

1. 在"情感分析"模块输入文本：
```
今天真是太开心了！完成了一个大项目，感觉特别有成就感。
终于可以放松一下了，打算晚上和朋友们出去庆祝。
```

2. 点击"开始分析"按钮

3. 查看分析结果：
   - 情感类型：😊 快乐
   - 共情得分：25%（情绪较稳定）
   - 兴趣关键词：项目、朋友、庆祝
   - 分析置信度：92%

### 对话建议使用示例

1. 在"对话建议生成"模块输入对话历史：
```
朋友：我最近工作压力好大啊，每天都加班到很晚
我：怎么了，发生什么事了？
朋友：项目deadline快到了，但进度还差很多，领导天天催
我：听起来确实挺辛苦的
```

2. 选择关系类型：朋友

3. 点击"生成建议"按钮

4. 查看建议回复：
```
建议回复：
"这种情况确实很累人，你已经很努力了。要不要我帮你一起想想办法，
或者先放松一下，说不定能找到更好的解决思路？我随时在这里支持你。"

建议理由：
对方处于高压状态，需要情感支持和实际帮助。回复既表达了理解和认可，
又提供了具体的支持选项，同时给对方喘息的空间。
```

## 🔧 API 文档

### 后端 API 接口

启动后端后，访问 `http://localhost:8000/docs` 查看自动生成的 Swagger API 文档。

#### 1. 情感分析接口

**端点**：`POST /analyze_emotion`

**请求体**：
```json
{
  "text": "要分析的文本内容"
}
```

**响应**：
```json
{
  "emotion_type": "快乐",
  "empathy_score": 0.25,
  "interest_keywords": ["项目", "朋友"],
  "confidence": 0.92
}
```

#### 2. 对话建议接口

**端点**：`POST /generate_suggestion`

**请求体**：
```json
{
  "dialogue_context": [
    "朋友：我最近工作压力好大",
    "我：怎么了？",
    "朋友：项目deadline快到了"
  ],
  "relationship_status": "朋友",
  "user_emotion": null
}
```

**响应**：
```json
{
  "suggestion_text": "建议的回复内容",
  "reasoning": "建议理由说明"
}
```

#### 3. 健康检查接口

**端点**：`GET /health`

**响应**：
```json
{
  "status": "healthy",
  "cuda_available": true,
  "model_loaded": true,
  "gpu_memory_allocated_gb": 9.87
}
```

## 🐛 常见问题

### Q1：CUDA out of memory（显存不足）

**解决方案**：

```bash
# 方案 1：下载更小的量化模型
python download_model.py
# 选择 Qwen2.5-7B-GPTQ-Int4 (仅需 6GB)

# 方案 2：修改 main.py，降低 max_new_tokens
# 将 max_new_tokens=512 改为 max_new_tokens=256
```

### Q2：模型推理速度慢

**优化建议**：

1. 确保模型已预热（启动时自动执行）
2. 检查 GPU 是否被正确使用：
```python
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
3. 使用量化模型（Int8/Int4）
4. 降低生成长度（max_new_tokens）

### Q3：前端 CORS 错误

**解决方案**：

确保后端 main.py 中已配置 CORS：
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Q4：JSON 解析失败

**可能原因**：

1. 模型输出格式不符合预期
2. 提示词需要调整

**解决方案**：

查看后端日志中的 "模型原始输出"，检查格式是否正确。如需调整，修改 main.py 中的 system_prompt。

### Q5：Windows 环境 CUDA 安装问题

**解决方案**：

```bash
# 卸载现有 PyTorch
pip uninstall torch torchvision torchaudio

# 重新安装 CUDA 版本（根据你的 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 📈 性能基准

### 响应时间（RTX 4090 24GB + Qwen2.5-7B-Int8）

| 任务类型 | 平均响应时间 | Token 生成速度 |
|---------|------------|--------------|
| 情感分析 | 200-300ms | ~30 tokens/s |
| 对话建议 | 400-600ms | ~28 tokens/s |
| 冷启动（首次） | 1-2s | - |

### 资源占用

- **显存占用**：约 10GB（Int8 量化）
- **内存占用**：约 2-3GB
- **并发支持**：单 GPU 可处理 5-10 个并发请求

## 🔐 生产部署建议

### Docker 部署

```dockerfile
FROM nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip git

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python3", "main.py"]
```

**构建和运行**：

```bash
docker build -t qwen-api:latest .
docker run --gpus all -p 8000:8000 -v $(pwd)/qwen_models:/app/qwen_models qwen-api:latest
```

### Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 监控和日志

后端默认输出日志到控制台，可配置日志文件：

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## 📄 许可证

本项目仅供学习和研究使用。Qwen 模型遵循其官方许可证。

## 🙏 致谢

- [Qwen Team](https://qwenlm.github.io/) - 提供强大的开源模型
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
- [Hugging Face](https://huggingface.co/) - Transformers 库
- [ModelScope](https://modelscope.cn/) - 模型托管平台

## 📞 支持

如遇到问题，请检查：

1. 后端日志输出
2. GPU 显存占用（`nvidia-smi`）
3. 模型文件完整性
4. Python 依赖版本

---

<div align="center">

**💡 让 AI 帮助我们更好地理解和表达情感**

Made with ❤️ by Echo Compass Team

</div>
