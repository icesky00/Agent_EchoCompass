# 📁 项目文件说明

## 文件结构

```
qwen_backend/
├── main.py                 # 后端主程序（FastAPI 服务）
├── download_model.py       # 模型下载脚本
├── index.html              # 前端界面（HTML+JS）
├── requirements.txt        # Python 依赖清单
├── README.md              # 完整使用文档
├── QUICKSTART.md          # 快速开始指南
├── start.sh               # Linux/Mac 启动脚本
└── start.bat              # Windows 启动脚本
```

## 文件详解

### 📄 main.py（核心文件）
**作用**：FastAPI 后端服务，提供 AI 推理接口

**主要功能**：
- 加载 Qwen 模型到 GPU
- 提供情感分析 API（`/analyze_emotion`）
- 提供对话建议 API（`/generate_suggestion`）
- 健康检查端点（`/health`）

**启动方式**：
```bash
python main.py
```

**配置项**（第 87 行）：
```python
model_name = "qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"  # 可修改模型
```

---

### 📄 download_model.py
**作用**：交互式模型下载工具

**支持的模型**：
1. Qwen2.5-7B-Int8（推荐，10GB 显存）
2. Qwen2.5-7B-FP16（完整精度，17GB 显存）
3. Qwen2.5-7B-Int4（激进量化，6GB 显存）
4. Qwen2.5-14B-Int8（高精度，18GB 显存）

**使用方式**：
```bash
python download_model.py
# 根据提示选择模型（建议选 1）
```

**模型存储位置**：
```
./qwen_models/  # 当前目录下
```

---

### 📄 index.html
**作用**：Web 前端界面

**功能**：
- 情感分析输入框和结果展示
- 对话建议输入框和结果展示
- 服务器状态实时监控
- 美观的响应式设计

**使用方式**：
```bash
# 方法 1：直接双击文件
# 方法 2：启动本地服务器
python -m http.server 3000
# 访问 http://localhost:3000
```

**API 配置**（第 257 行）：
```javascript
const API_BASE_URL = 'http://localhost:8000';  // 可修改后端地址
```

---

### 📄 requirements.txt
**作用**：Python 依赖包列表

**主要依赖**：
- `torch` - PyTorch 深度学习框架
- `transformers` - Hugging Face 模型库
- `fastapi` - Web 框架
- `uvicorn` - ASGI 服务器
- `modelscope` - ModelScope 模型下载

**安装方式**：
```bash
pip install -r requirements.txt
```

**注意**：安装时间约 5-10 分钟，需要下载约 3-4GB 的包。

---

### 📄 README.md
**作用**：完整的项目文档

**包含内容**：
- 项目简介和功能特性
- 详细安装步骤
- 使用指南和示例
- API 文档说明
- 常见问题解答
- 生产部署建议
- 性能基准测试

**适合**：需要深入了解项目的用户

---

### 📄 QUICKSTART.md
**作用**：5 分钟快速开始指南

**包含内容**：
- 最简化的安装步骤
- 快速启动命令
- 常见问题速查
- 关键配置说明

**适合**：想要快速上手的用户

---

### 📄 start.sh（Linux/Mac）
**作用**：一键启动脚本

**功能**：
- 自动检查 Python 和 CUDA
- 激活虚拟环境（如存在）
- 检测模型是否已下载
- 启动后端服务

**使用方式**：
```bash
chmod +x start.sh
./start.sh
```

---

### 📄 start.bat（Windows）
**作用**：一键启动脚本（Windows 版）

**功能**：同 start.sh，适配 Windows 环境

**使用方式**：
```cmd
start.bat
```

或直接双击文件。

---

## 💡 使用建议

### 新手用户
1. 阅读 `QUICKSTART.md`
2. 运行 `download_model.py` 下载模型
3. 使用 `start.sh` 或 `start.bat` 启动
4. 打开 `index.html` 使用

### 开发者
1. 阅读 `README.md`
2. 查看 `main.py` 了解 API 实现
3. 根据需求修改配置
4. 访问 `http://localhost:8000/docs` 查看 API 文档

### 生产环境
1. 阅读 README.md 的"生产部署"章节
2. 配置 Docker 容器化
3. 设置 Nginx 反向代理
4. 配置日志和监控

---

## 🔧 常见修改

### 更换模型
修改 `main.py` 第 87 行：
```python
model_name = "qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"
```

### 修改端口
修改 `main.py` 最后一行：
```python
uvicorn.run(app, host="0.0.0.0", port=9000)  # 改为 9000
```

同时修改 `index.html` 第 257 行：
```javascript
const API_BASE_URL = 'http://localhost:9000';
```

### 调整推理参数
修改 `main.py` 第 147 行的 `_generate_response` 函数：
```python
temperature=0.7,      # 降低可减少随机性
top_p=0.9,            # 核采样参数
max_new_tokens=256,   # 减少可提升速度
```

---

## 📞 获取帮助

- **技术问题**：查看 README.md 的"常见问题"章节
- **API 使用**：访问 http://localhost:8000/docs
- **模型文档**：https://qwenlm.github.io/

---

祝使用愉快！🎉
