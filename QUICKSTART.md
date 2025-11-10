# 🚀 快速开始（5 分钟部署）

## 1️⃣ 安装依赖（首次运行）

```bash
# 创建虚拟环境
conda create -n qwen_env python=3.10
conda activate qwen_env

# 安装依赖包
pip install -r requirements.txt
```

## 2️⃣ 下载模型（首次运行）

```bash
python download_model.py
# 选择 1（推荐）：Qwen2.5-7B-Int8，约 7-8GB
```

## 3️⃣ 启动服务

### Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

### Windows
```bash
start.bat
```

### 或手动启动
```bash
python main.py
```

## 4️⃣ 打开前端

- 双击打开 `index.html`
- 或访问 `http://localhost:3000`（需先运行 `python -m http.server 3000`）

---

## 📍 重要链接

- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **前端界面**: index.html

---

## ⚡ 性能要求

| 配置 | GPU | 显存 | 推荐模型 |
|-----|-----|------|---------|
| 最低 | RTX 3060 | 12GB | Qwen2.5-7B-Int4 |
| 推荐 | RTX 4090 | 24GB | Qwen2.5-7B-Int8 |
| 高端 | RTX A6000 | 48GB | Qwen2.5-14B |

---

## 🐛 常见问题

### 显存不足
```bash
# 下载更小的模型
python download_model.py  # 选择选项 3（Int4）
```

### CUDA 未检测到
```bash
# 验证 CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 前端无法连接
- 确保后端已启动（http://localhost:8000）
- 检查防火墙设置
- 查看浏览器控制台错误

---

## 📞 技术支持

查看完整文档：`README.md`

祝使用愉快！🎉
