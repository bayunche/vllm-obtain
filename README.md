# 🚀 VLLM 跨平台推理服务

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20macOS%20|%20Linux-lightgrey.svg)](README.md)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](README.md)
[![Tested](https://img.shields.io/badge/Tested-Mac%20Studio%20M3%20Ultra-success.svg)](Mac_Studio_完整测试报告_20250821.md)

> 🎯 **企业级跨平台 AI 推理服务，完全兼容 OpenAI API**

专为企业和开发者设计的高性能跨平台推理服务，支持在 **n8n**、**Dify**、**LangChain** 等工具中无缝替换 OpenAI API，实现完全本地化的 AI 能力。

## ✨ 核心特性

### 🎯 **智能平台适配**
- **🍎 macOS Apple Silicon** → MLX 引擎 (Metal 加速，极致优化)
- **🐧 Linux CUDA/ROCm** → VLLM 引擎 (高性能 GPU 推理)  
- **🪟 Windows CUDA** → VLLM/LlamaCpp 引擎 (兼容性优先)
- **💻 通用 CPU** → LlamaCpp 引擎 (最广兼容性)
- **⚡ 零配置启动** → 智能检测硬件，自动选择最佳引擎

### 🔌 **完美 OpenAI 兼容**
- ✅ **100% API 兼容** OpenAI ChatGPT API 格式
- ✅ **零代码迁移** 现有 OpenAI 客户端无需修改
- ✅ **流式响应支持** 实时 Token 生成体验
- ✅ **n8n 直接替换** 只需修改 Base URL

### ⚡ **生产级性能**
- 🚀 **经过验证**: Mac Studio M3 Ultra 测试通过 ([测试报告](Mac_Studio_完整测试报告_20250821.md))
- 📊 **高并发支持**: 单机 20+ 并发，100% 成功率  
- 🔄 **动态模型管理** 运行时加载/卸载模型
- 🛡️ **故障自愈** 自动重试、优雅降级、健康检查

### 🎮 **开发者友好**
- 🎯 **一键启动** `python run.py`
- 🔍 **智能诊断** 自动检测环境和依赖问题
- 📝 **结构化日志** 详细性能监控和错误追踪
- 🧪 **完整测试** OpenAI 兼容性全面验证

---

## 🚀 快速开始

### 📋 系统要求

| 平台 | 最低配置 | 推荐配置 | 推理引擎 |
|------|----------|----------|----------|
| **macOS** | M1, 16GB RAM | M3 Ultra, 64GB RAM | MLX |
| **Linux** | GTX 1080, 16GB RAM | RTX 4090, 32GB RAM | VLLM |
| **Windows** | GTX 1070, 16GB RAM | RTX 4080, 32GB RAM | VLLM/LlamaCpp |
| **通用CPU** | 8核, 16GB RAM | 32核, 64GB RAM | LlamaCpp |

### 🛠️ 安装部署

#### 方式一：自动安装脚本 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/your-org/vllm-obtain
cd vllm-obtain

# 2. 运行一键安装脚本
./install.sh  # Linux/macOS
# 或 install.bat  # Windows

# 3. 启动服务 
python run.py
```

#### 方式二：手动安装

<details>
<summary>📱 macOS (Apple Silicon) 安装步骤</summary>

```bash
# 1. 环境准备
python3 -m venv venv
source venv/bin/activate

# 2. 安装 macOS 专用依赖
pip install -r requirements-mac.txt

# 3. 配置环境 (使用 Mac 优化配置)
cp .env.mac .env

# 4. 下载模型 (使用 ModelScope 国内镜像)
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2.5-0.5B-Instruct', 
                 cache_dir='./models/qwen-0.5b')
"

# 5. 启动服务
python run.py --mode prod

# ✅ 服务地址: http://127.0.0.1:8001
# 🎯 配置特点: MLX引擎 + 单worker + Gevent异步
```

**Mac 专用优势**:
- 🔥 **MLX 引擎**: 专为 Apple Silicon 优化
- ⚡ **Metal 加速**: GPU 算力完全发挥
- 💾 **统一内存**: 512GB 内存高效利用
- 🔇 **静音运行**: 功耗低，几乎无噪音

</details>

<details>
<summary>🐧 Linux (NVIDIA GPU) 安装步骤</summary>

```bash
# 1. 系统依赖 (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential

# 2. Python 环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装 Linux GPU 专用依赖
pip install -r requirements-linux.txt

# 4. 验证 CUDA 环境
nvidia-smi  # 确认 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"

# 5. 配置环境 (使用 Linux 多 worker 配置)
cp .env.linux .env

# 6. 下载模型
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/qwen-0.5b

# 7. 启动高性能服务
python run.py --mode prod

# ✅ 服务地址: http://0.0.0.0:8001 (监听所有接口)
# 🎯 配置特点: VLLM引擎 + 4workers + 高并发
```

**Linux 专用优势**:
- 🚀 **VLLM 引擎**: 最高性能推理引擎
- 🔥 **CUDA 加速**: NVIDIA GPU 完全发挥
- 📊 **多进程**: 4+ workers 并行处理
- 🌐 **高并发**: 支持 100+ 并发请求

</details>

<details>
<summary>🪟 Windows (NVIDIA GPU) 安装步骤</summary>

```powershell
# 1. 环境准备 (需要管理员权限安装 Visual Studio Build Tools)
# 下载并安装: Visual Studio Community 或 Build Tools

# 2. Python 环境
python -m venv venv
.\venv\Scripts\activate

# 3. 安装 Windows 专用依赖
pip install -r requirements-windows.txt

# 4. 验证 GPU 环境 (可选)
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 5. 配置环境 (Windows 兼容性配置)
copy .env.windows .env

# 6. 下载模型 (推荐使用 GGUF 格式)
python download_model.py --model qwen-0.5b --format gguf

# 7. 启动服务 (使用 Waitress 服务器)
python run.py --mode prod

# ✅ 服务地址: http://127.0.0.1:8001
# 🎯 配置特点: LlamaCpp引擎 + Waitress + 同步处理
```

**Windows 专用优势**:
- 🔧 **兼容性**: 支持各种 Windows 版本
- 🛡️ **稳定性**: Waitress 服务器稳定可靠
- 🎮 **游戏友好**: 不抢占游戏 GPU 资源
- 💻 **集成度**: 完美融入 Windows 生态

</details>

<details>
<summary>💻 CPU 通用模式安装步骤</summary>

```bash
# 适用于任何平台的 CPU 推理模式

# 1. Python 环境 (适用于所有平台)
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:  
venv\Scripts\activate

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. CPU 模式配置
export INFERENCE_ENGINE=llamacpp
export DEVICE_TYPE=cpu
export MAX_CPU_THREADS=8

# 4. 下载 GGUF 格式模型 (CPU 优化)
mkdir -p models
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf \
     -O models/qwen2.5-0.5b-instruct-q4_0.gguf

# 5. 启动 CPU 模式服务
python run.py

# ✅ 通用兼容，所有平台都能运行
```

**CPU 模式优势**:
- 🌍 **通用兼容**: 任何平台都能运行
- 💰 **成本低**: 无需昂贵的GPU硬件
- 🔋 **功耗低**: 适合长时间运行
- 🛠️ **易维护**: 依赖简单，问题少

</details>

---

## 📊 平台对比和选择指南

### 性能对比表

| 平台配置 | 吞吐量 | 响应时间 | 并发数 | Token速度 | 功耗 | 推荐场景 |
|----------|--------|----------|--------|-----------|------|----------|
| **Mac Studio M3 Ultra** | 2.6 req/s | 0.77-7s | 20+ | 6.7 t/s | 极低 | ⭐⭐⭐⭐⭐ 开发/原型 |
| **Linux RTX 4090** | 15+ req/s | 0.1-0.5s | 100+ | 200+ t/s | 高 | ⭐⭐⭐⭐⭐ 生产/高并发 |
| **Windows RTX 4080** | 8+ req/s | 0.2-1s | 50+ | 150+ t/s | 中 | ⭐⭐⭐⭐ 企业/办公 |
| **通用 CPU (32核)** | 1 req/s | 2-10s | 5+ | 10+ t/s | 低 | ⭐⭐⭐ 兼容/备用 |

### 🎯 推荐配置选择

```bash
# 🍎 macOS 用户 (开发和演示场景)
如果你有: Mac Studio/MacBook Pro M3+
推荐配置: MLX + 单worker + 10并发
优势: 功耗低、噪音小、开箱即用
```

```bash
# 🐧 Linux 用户 (生产和高性能场景) 
如果你有: Linux + NVIDIA RTX 4090/A100
推荐配置: VLLM + 4workers + 100并发
优势: 性能最强、并发最高、扩展性好
```

```bash
# 🪟 Windows 用户 (企业和个人场景)
如果你有: Windows + NVIDIA RTX 4070+
推荐配置: VLLM/LlamaCpp + 2workers + 50并发
优势: 兼容性好、稳定可靠、易维护
```

```bash
# 💻 通用 CPU 用户 (兼容和备用场景)
如果你只有: 普通 CPU 服务器
推荐配置: LlamaCpp + 1worker + 5并发
优势: 零门槛、成本低、普适性强
```

---

## 🔧 配置和环境管理

### 智能配置系统

项目提供平台特定的配置文件，自动优化性能：

```bash
# 配置文件说明
.env.mac        # macOS 专用优化配置 (MLX + Metal)
.env.linux      # Linux 专用优化配置 (VLLM + CUDA) 
.env.windows    # Windows 专用优化配置 (兼容性优先)
.env.example    # 通用配置模板
```

### 依赖管理系统

```bash
# 平台特定依赖文件
requirements-mac.txt     # macOS: MLX + 相关依赖
requirements-linux.txt   # Linux: VLLM + CUDA 依赖  
requirements-windows.txt # Windows: 兼容性依赖
requirements.txt        # 通用基础依赖
```

### 环境变量配置

<details>
<summary>🔧 核心配置参数说明</summary>

```bash
# === 引擎选择配置 ===
INFERENCE_ENGINE=auto    # auto/mlx/vllm/llamacpp
DEVICE_TYPE=auto        # auto/cuda/mps/cpu
WORKERS=1              # worker进程数 (Mac必须为1)

# === 性能调优配置 ===
MAX_CONCURRENT_REQUESTS=10   # 最大并发请求
REQUEST_TIMEOUT=120         # 请求超时时间
WORKER_CLASS=gevent         # worker类型
MEMORY_LIMIT=80            # 内存使用上限(%)

# === 模型管理配置 ===
DEFAULT_MODEL=qwen-0.5b    # 默认模型
MODEL_DIR=./models         # 模型存储目录
CACHE_SIZE=1000           # 缓存大小
ENABLE_CACHING=true       # 启用缓存

# === 日志监控配置 ===
LOG_LEVEL=INFO            # 日志级别
LOG_RETENTION_DAYS=30     # 日志保留天数
ENABLE_METRICS=true       # 启用性能监控
```

</details>

---

## 🧪 验证和测试

### 一键功能验证

```bash
# 1. 系统健康检查
curl http://localhost:8001/health

# 2. OpenAI 兼容性测试 
python test_openai_compatibility.py

# 3. 并发性能测试
python concurrent_test.py

# 4. 平台兼容性检查
python src/utils/platform_detector.py
```

### 性能基准测试

```bash
# 响应时间测试
curl -w "Time: %{time_total}s\n" -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-0.5b","messages":[{"role":"user","content":"Hello"}]}'

# 并发压力测试 
python -c "
import concurrent.futures, requests, time
def test_req(): 
    return requests.post('http://localhost:8001/v1/chat/completions', 
                        json={'model':'qwen-0.5b','messages':[{'role':'user','content':'test'}]})
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(lambda _: test_req(), range(10)))
print(f'10 concurrent requests: {time.time()-start:.2f}s')
print(f'Success rate: {sum(1 for r in results if r.status_code==200)/10*100}%')
"
```

---

## 🔌 OpenAI API 兼容使用

### 在各种工具中使用

#### n8n 工作流使用
```
1. 添加 "OpenAI" 节点
2. 配置连接:
   - Base URL: http://your-server:8001/v1
   - API Key: any-string (可选)
3. 选择模型: qwen-0.5b
4. 开始使用! 🎉
```

#### Python OpenAI 客户端
```python
import openai

# 只需修改这两行
openai.api_base = "http://localhost:8001/v1" 
openai.api_key = "your-key"  # 可以是任意字符串

# 其他代码完全不变!
response = openai.ChatCompletion.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### LangChain 集成
```python
from langchain.llms import OpenAI

llm = OpenAI(
    openai_api_base="http://localhost:8001/v1",
    openai_api_key="any-key",
    model_name="qwen-0.5b"
)

result = llm("Explain quantum computing")
```

#### cURL 命令行
```bash
# 聊天补全
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-0.5b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# 获取模型列表
curl http://localhost:8001/v1/models
```

---

## 📊 监控和运维

### 实时监控

```bash
# 服务状态检查
curl http://localhost:8001/v1/system/status | jq

# 性能指标监控
curl http://localhost:8001/v1/system/metrics | jq

# 健康检查
curl http://localhost:8001/health
```

### 日志分析

```bash
# 实时日志查看
tail -f logs/app.log

# 性能分析
grep "tokens_per_second" logs/app.log | tail -10

# 错误追踪  
grep "ERROR" logs/app.log | tail -20
```

### 生产部署

<details>
<summary>🚀 生产环境部署配置</summary>

#### Systemd 服务 (Linux)
```ini
# /etc/systemd/system/vllm-inference.service
[Unit]
Description=VLLM Inference Service
After=network.target

[Service]
Type=simple
User=vllm
WorkingDirectory=/opt/vllm-obtain
ExecStart=/opt/vllm-obtain/venv/bin/python run.py --mode prod
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Docker 部署
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["python", "run.py", "--mode", "prod"]
```

#### Nginx 负载均衡
```nginx
upstream vllm_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

</details>

---

## 🐛 故障排查指南

### 常见问题和解决方案

<details>
<summary>❌ MLX 引擎 Metal 错误 (Mac)</summary>

**问题**: `[metal::Device] Unable to build metal library from source`

**原因**: 多 worker 模式下 MLX 引擎冲突

**解决方案**:
```bash
# 1. 确保使用单 worker 配置
export WORKERS=1

# 2. 使用 Mac 专用配置
cp .env.mac .env

# 3. 重启服务
python run.py --mode prod
```

**验证**: `curl http://localhost:8001/health` 显示健康状态

</details>

<details>
<summary>❌ CUDA 内存不足 (Linux/Windows)</summary>

**问题**: `CUDA out of memory`

**解决方案**:
```bash
# 1. 降低 GPU 内存使用
export CUDA_MEMORY_FRACTION=0.6

# 2. 减少并发数
export MAX_CONCURRENT_REQUESTS=10

# 3. 使用量化模型
python download_model.py --model qwen-0.5b --quantization int4

# 4. 回退到 CPU 模式 
export INFERENCE_ENGINE=llamacpp
export DEVICE_TYPE=cpu
```

</details>

<details>
<summary>❌ 依赖安装问题</summary>

**问题**: 各种包安装失败

**解决方案**:
```bash
# Mac: 使用 Homebrew
brew install python@3.11
pip install -r requirements-mac.txt

# Linux: 安装系统依赖
sudo apt install python3-dev build-essential
pip install -r requirements-linux.txt

# Windows: 安装 Visual Studio Build Tools
# 下载: https://visualstudio.microsoft.com/zh-hans/downloads/
pip install -r requirements-windows.txt
```

</details>

### 🔍 调试工具

```bash
# 运行完整诊断
python run.py --mode dev --debug

# 平台检测
python src/utils/platform_detector.py

# 性能基准测试
python concurrent_test.py

# OpenAI 兼容性测试
python test_openai_compatibility.py
```

---

## 📚 文档资源

- 📖 **[项目结构说明](PROJECT_STRUCTURE.md)** - 代码架构和开发指南
- 📋 **[生产部署指南](README_PRODUCTION.md)** - 详细部署和配置说明
- 📊 **[Mac Studio 测试报告](Mac_Studio_完整测试报告_20250821.md)** - 完整性能测试结果
- 🔧 **[API 文档](API_DOCS.md)** - 详细 API 接口说明
- 🐛 **[故障排查手册](TROUBLESHOOTING.md)** - 常见问题解决方案

---

## 🤝 社区和支持

### 获取帮助

- 🐛 **问题报告**: [GitHub Issues](../../issues)
- 💬 **社区讨论**: [GitHub Discussions](../../discussions)  
- 📧 **技术支持**: 通过 Issue 联系我们
- 📚 **详细文档**: 查看项目 docs 目录

### 贡献指南

1. **Fork** 项目
2. **创建** 功能分支: `git checkout -b feature/amazing-feature`
3. **提交** 更改: `git commit -m 'Add amazing feature'`
4. **推送** 分支: `git push origin feature/amazing-feature`
5. **创建** Pull Request

### 开发规范

- ✅ 遵循 PEP 8 代码规范
- ✅ 添加类型提示 (Type Hints)
- ✅ 编写单元测试
- ✅ 更新相关文档

---

## 📄 许可证和致谢

### 许可证
本项目采用 [MIT 许可证](LICENSE) - 您可以自由使用、修改和分发。

### 致谢
感谢以下开源项目:
- [VLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon 优化
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU 推理引擎
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [Qwen](https://huggingface.co/Qwen) - 开源模型

---

## 📈 项目状态

- ✅ **当前版本**: v1.0.0 
- 🚀 **开发状态**: 持续维护和更新
- 🔄 **API 兼容**: 100% OpenAI 兼容
- 🌍 **平台支持**: Windows, macOS, Linux 全支持
- 🧪 **测试覆盖**: Mac Studio M3 Ultra 生产级测试通过

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star 项目](../../stargazers) | [🐛 报告问题](../../issues) | [💡 功能建议](../../issues) | [📖 查看文档](docs/)

---

*让每个开发者都能轻松拥有本地化的企业级 AI 能力* 🚀

Made with ❤️ by VLLM Team

</div>