# 🚀 VLLM 跨平台推理服务

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20|%20macOS%20|%20Linux-lightgrey.svg)](README.md)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](README.md)

> 🎯 **一键部署，智能选择，完全兼容 OpenAI API 的本地大语言模型推理服务**

专为企业和开发者设计的高性能跨平台推理服务，支持在 **n8n**、**Dify**、**LangChain** 等工具中无缝替换 OpenAI API，实现完全本地化的 AI 能力。

## ✨ 为什么选择我们

### 🎯 **智能平台适配**
- **macOS Apple Silicon** → 自动启用 MLX 引擎 (Metal 优化)
- **Linux/Windows CUDA** → 自动启用 VLLM 引擎 (高性能推理)
- **通用平台/CPU** → 自动回退 llama.cpp 引擎 (最广兼容)
- **无需手动配置** → 一键启动，智能识别最佳方案

### 🔌 **完美 OpenAI 兼容**
- ✅ **100% 兼容** OpenAI API 格式和响应
- ✅ **直接在 n8n 中使用** - 只需修改 Base URL
- ✅ **支持流式响应** - 实时文本生成体验
- ✅ **零代码迁移** - 现有 OpenAI 代码无需修改

### ⚡ **企业级性能**
- 🚀 **动态模型管理** - 运行时加载/卸载，节省资源
- 📊 **实时性能监控** - Token速度、延迟、资源使用
- 🔄 **多模型并行** - 高配置设备可同时运行多个模型
- 🛡️ **故障自愈** - 自动重试、优雅降级、健康检查

### 🎮 **开发者友好**
- 🎯 **一键启动** - `python run.py --mode dev`
- 🔍 **智能检测** - 自动验证环境和依赖
- 📝 **详细日志** - 结构化日志，便于调试
- 🧪 **完整测试** - OpenAI 兼容性验证

---

## 🚀 5分钟快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd vllm推理框架

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖 (会自动根据平台选择合适的推理引擎)
pip install -r requirements.txt
```

### 2. 一键启动

```bash
# 开发模式启动 (自动检测环境并启动最佳配置)
python run.py --mode dev

# 🎉 服务启动成功！
# 访问地址: http://localhost:8000
# OpenAI API: http://localhost:8000/v1
```

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 运行完整的 OpenAI 兼容性测试
python test_openai_compatibility.py
```

**✅ 完成！** 现在你有一个完全兼容 OpenAI API 的本地推理服务。

---

## 🎮 使用示例

### 在 n8n 中使用 (推荐)

1. **添加 OpenAI 节点**
2. **修改连接配置**:
   ```
   Base URL: http://localhost:8000/v1
   API Key: 可以留空或随意填写
   ```
3. **选择模型**: 使用 `Qwen2.5-7B-Instruct` 或其他已加载模型
4. **开始使用**: 完全兼容所有 ChatGPT 功能

### 使用 OpenAI Python 客户端

```python
import openai

# 替换 API 基础地址，其他代码无需修改
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any-key"  # 可选，可以是任意字符串

# 对话补全 - 完全兼容 OpenAI 格式
response = openai.ChatCompletion.create(
    model="Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手"},
        {"role": "user", "content": "请写一个 Python 函数来计算斐波那契数列"}
    ],
    max_tokens=500,
    temperature=0.7,
    stream=False  # 支持流式: stream=True
)

print(response.choices[0].message.content)
```

### 使用 cURL 调用

```bash
# 聊天对话接口
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "解释什么是机器学习，用简单的话"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'

# 流式对话 (实时生成)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct", 
    "messages": [{"role": "user", "content": "写一首关于AI的诗"}],
    "stream": true
  }'

# 文本补全接口
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "prompt": "人工智能的未来发展趋势包括",
    "max_tokens": 150
  }'
```

---

## 📋 API 文档

### OpenAI 兼容接口

| 端点 | 方法 | 描述 | 兼容性 |
|------|------|------|--------|
| `/v1/models` | GET | 获取可用模型列表 | ✅ 100% |
| `/v1/models/{id}` | GET | 获取特定模型信息 | ✅ 100% |
| `/v1/chat/completions` | POST | 聊天对话补全 | ✅ 100% |
| `/v1/completions` | POST | 文本补全 | ✅ 100% |
| `/v1/embeddings` | POST | 文本嵌入 | 🚧 计划中 |

### 管理接口 (扩展功能)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 服务健康检查 |
| `/v1/system/status` | GET | 系统状态详情 |
| `/v1/system/health` | GET | 深度健康检查 |
| `/v1/system/metrics` | GET | 性能指标监控 |
| `/v1/models/load` | POST | 动态加载模型 |
| `/v1/models/{name}/unload` | DELETE | 卸载指定模型 |
| `/v1/models/{name}/status` | GET | 模型状态查询 |

### 请求示例

#### 聊天补全

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手"},
    {"role": "user", "content": "用Python写一个快速排序"}
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["```"]
}
```

#### 响应格式

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "Qwen2.5-7B-Instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "这里是生成的回复内容..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 120,
    "total_tokens": 148
  }
}
```

---

## ⚙️ 配置参考

### 环境变量配置

```bash
# 复制配置模板
cp .env.example .env
```

#### 核心配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INFERENCE_ENGINE` | `auto` | 推理引擎选择 (auto/vllm/mlx/llama_cpp) |
| `DEVICE_TYPE` | `auto` | 设备类型 (auto/cuda/mps/cpu) |
| `MAX_CONCURRENT_MODELS` | `1` | 最大并发模型数 |
| `DEFAULT_MODEL` | `Qwen2.5-7B-Instruct` | 默认模型名称 |
| `MODEL_BASE_PATH` | `./models` | 模型存储路径 |

#### 服务配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务端口 |
| `WORKERS` | `4` | Gunicorn Worker 数量 |
| `MAX_CONCURRENT_REQUESTS` | `100` | 最大并发请求数 |

### 平台优化配置

#### macOS (M3 Ultra) 高性能配置
```bash
INFERENCE_ENGINE=auto          # 自动选择 MLX
DEVICE_TYPE=mps               # Metal Performance Shaders
MAX_GPU_MEMORY=0.8            # 使用 80% GPU 内存
MAX_CONCURRENT_MODELS=3       # 利用大内存优势
MAX_CONCURRENT_REQUESTS=150   # 高并发支持
```

#### Linux/Windows (CUDA) 高性能配置
```bash
INFERENCE_ENGINE=auto          # 自动选择 VLLM
DEVICE_TYPE=cuda              # CUDA 加速
MAX_GPU_MEMORY=0.8            # GPU 内存限制
MAX_CONCURRENT_MODELS=2       # 根据显存调整
ENABLE_CACHING=True           # 启用智能缓存
```

#### CPU 通用配置
```bash
INFERENCE_ENGINE=llama_cpp     # CPU 优化引擎
DEVICE_TYPE=cpu               # CPU 模式
MAX_CPU_THREADS=8             # CPU 线程数
MAX_CONCURRENT_MODELS=1       # 单模型模式
```

---

## 📦 模型管理

### 支持的模型格式

| 推理引擎 | 支持格式 | 最佳使用场景 |
|----------|----------|--------------|
| **VLLM** | HuggingFace, Safetensors | Linux/Windows GPU 高性能推理 |
| **MLX** | MLX 格式, HuggingFace | macOS Apple Silicon 优化 |
| **LlamaCpp** | GGUF 格式 | 跨平台兼容，资源受限环境 |

### 快速获取模型

#### 1. GGUF 模型 (推荐新手)
```bash
# 创建模型目录
mkdir -p models

# 下载 Qwen2.5-7B GGUF 模型 (约 4.3GB)
cd models
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_0.gguf

# 或使用 Hugging Face CLI
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_0.gguf --local-dir models
```

#### 2. HuggingFace 模型
```bash
# 下载完整模型 (约 15GB)
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct
```

### 动态模型管理

```bash
# 加载新模型
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-custom-model",
    "model_path": "./models/my-model.gguf",
    "engine_type": "llama_cpp"
  }'

# 查看已加载模型
curl http://localhost:8000/v1/models

# 查看模型详细状态
curl http://localhost:8000/v1/models/my-custom-model/status

# 卸载模型释放资源
curl -X DELETE http://localhost:8000/v1/models/my-custom-model/unload
```

---

## 📊 监控和日志

### 实时监控

```bash
# 系统整体状态
curl http://localhost:8000/v1/system/status

# 性能指标
curl http://localhost:8000/v1/system/metrics

# 最近日志 (最后100行)
curl http://localhost:8000/v1/logs/recent?limit=100
```

### 日志分析

```bash
# 查看推理性能
tail -f logs/inference.log | grep "tokens_per_second"

# 查看错误日志
grep "ERROR" logs/inference.log | tail -20

# 查看模型操作日志
grep "model_operation" logs/metrics.log
```

### 性能指标解读

```json
{
  "system": {
    "cpu_usage": 45.2,          // CPU 使用率 %
    "memory_usage": 67.8,       // 内存使用率 %
    "gpu_memory_usage": 82.3    // GPU 显存使用率 %
  },
  "inference": {
    "active_models": 2,         // 当前活跃模型数
    "total_requests": 1542,     // 总请求数
    "avg_response_time": 1.23,  // 平均响应时间 (秒)
    "tokens_per_second": 125.4  // 平均生成速度 (token/秒)
  }
}
```

---

## 🚀 部署指南

### 开发环境

```bash
# 启动开发服务器 (自动重载)
python run.py --mode dev

# 启动时跳过依赖检查 (加快启动)
python run.py --mode dev --skip-check
```

### 生产环境

```bash
# 生产模式启动 (使用 Gunicorn)
python run.py --mode prod

# 手动使用 Gunicorn 启动
pip install gunicorn gevent
gunicorn --bind 0.0.0.0:8000 --workers 4 --worker-class gevent \
  --timeout 300 --keepalive 5 --max-requests 1000 \
  src.api.app:get_app
```

### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run.py", "--mode", "prod"]
```

```bash
# 构建和运行
docker build -t vllm-inference-service .
docker run -d -p 8000:8000 \
  -v ./models:/app/models \
  -v ./logs:/app/logs \
  --name vllm-service \
  vllm-inference-service
```

### 使用 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  vllm-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./cache:/app/cache
    environment:
      - INFERENCE_ENGINE=auto
      - MAX_CONCURRENT_MODELS=2
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vllm-service
```

### 系统服务 (Linux)

```ini
# /etc/systemd/system/vllm-inference.service
[Unit]
Description=VLLM Inference Service
After=network.target

[Service]
Type=simple
User=vllm
WorkingDirectory=/opt/vllm-inference
ExecStart=/opt/vllm-inference/venv/bin/python run.py --mode prod
Restart=always
RestartSec=10
Environment=PATH=/opt/vllm-inference/venv/bin

[Install]
WantedBy=multi-user.target
```

```bash
# 安装和启动系统服务
sudo systemctl enable vllm-inference
sudo systemctl start vllm-inference
sudo systemctl status vllm-inference
```

---

## 🐛 故障排除

### 常见问题和解决方案

#### ❌ 模型加载失败

**问题**: `ModelLoadError: 模型加载失败`

**解决方案**:
```bash
# 1. 检查模型文件是否存在
ls -la models/

# 2. 检查模型文件权限
chmod 644 models/*.gguf

# 3. 检查可用内存/显存
# GPU 显存检查 (CUDA)
nvidia-smi
# 系统内存检查
free -h

# 4. 降低内存使用
export MAX_GPU_MEMORY=0.6
export MAX_CONCURRENT_MODELS=1
```

#### ❌ 推理速度慢

**问题**: Token 生成速度 < 10 tokens/sec

**解决方案**:
```bash
# 1. 检查是否使用了正确的推理引擎
curl http://localhost:8000/v1/system/status

# 2. 优化并发配置
export MAX_CONCURRENT_REQUESTS=50
export WORKERS=8

# 3. 启用缓存加速
export ENABLE_CACHING=True
export CACHE_SIZE=1000

# 4. 检查设备配置
# 确保使用 GPU 加速
export DEVICE_TYPE=cuda  # 或 mps (macOS)
```

#### ❌ CUDA/GPU 相关错误

**问题**: `CUDA out of memory` 或 GPU 不可用

**解决方案**:
```bash
# 1. 降低 GPU 内存使用
export MAX_GPU_MEMORY=0.5

# 2. 使用 CPU 模式
export INFERENCE_ENGINE=llama_cpp
export DEVICE_TYPE=cpu

# 3. 检查 CUDA 安装
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 4. 重启服务清理内存
sudo systemctl restart vllm-inference
```

#### ❌ n8n 连接问题

**问题**: n8n 中无法连接到服务

**解决方案**:
```bash
# 1. 确保服务运行在正确端口
curl http://localhost:8000/health

# 2. 检查防火墙设置
sudo ufw allow 8000/tcp

# 3. n8n 配置检查
# Base URL: http://YOUR_SERVER_IP:8000/v1
# API Key: 可以是任意字符串

# 4. 网络连通性测试
curl -X POST http://YOUR_SERVER_IP:8000/v1/models
```

### 调试工具

```bash
# 运行完整的系统诊断
python run.py --mode dev --debug

# OpenAI API 兼容性测试
python test_openai_compatibility.py

# 平台兼容性检查
python src/utils/platform_detector.py

# 查看详细日志
tail -f logs/inference.log

# 性能基准测试
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{...}' \
  -w "Response Time: %{time_total}s\n"
```

---

## 📋 系统要求

### 最低要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **Python** | 3.11+ | 3.11+ |
| **内存** | 8GB | 32GB+ |
| **存储** | 20GB | 500GB+ SSD |
| **网络** | 1Mbps | 100Mbps+ |

### 硬件推荐

#### 🍎 macOS (推荐配置)
- **CPU**: M3 Ultra 或更高
- **内存**: 64GB+ (支持多个大模型并行)
- **存储**: 1TB+ SSD
- **推理引擎**: MLX (Metal 优化)

#### 🐧 Linux 高性能配置
- **CPU**: Intel Xeon 或 AMD EPYC
- **GPU**: RTX 4090 (24GB) / A100 (80GB)
- **内存**: 64GB+ DDR4/DDR5
- **存储**: 1TB+ NVMe SSD
- **推理引擎**: VLLM (CUDA 优化)

#### 🪟 Windows 配置
- **CPU**: Intel i7-12700K 或更高
- **GPU**: RTX 4070 Ti (12GB) 或更高
- **内存**: 32GB+ DDR4
- **存储**: 500GB+ NVMe SSD
- **推理引擎**: VLLM 或 LlamaCpp

#### 💻 CPU 通用配置
- **CPU**: 8核心 16线程以上
- **内存**: 16GB+ (推荐 32GB)
- **存储**: 250GB+ SSD
- **推理引擎**: LlamaCpp

---

## 🤝 开发指南

### 项目结构

```
vllm推理框架/
├── src/
│   ├── core/                  # 核心业务逻辑
│   │   ├── inference_engine.py   # 推理引擎抽象
│   │   ├── model_manager.py      # 模型管理器
│   │   └── exceptions.py         # 异常定义
│   ├── engines/               # 推理引擎实现
│   │   ├── vllm_engine.py        # VLLM 引擎
│   │   ├── mlx_engine.py         # MLX 引擎
│   │   └── llamacpp_engine.py    # LlamaCpp 引擎
│   ├── api/                   # Web API 层
│   │   ├── app.py                # Flask 应用
│   │   └── routes/               # API 路由
│   └── utils/                 # 工具模块
│       ├── config.py             # 配置管理
│       ├── logger.py             # 日志系统
│       └── platform_detector.py  # 平台检测
├── models/                    # 模型存储目录
├── logs/                      # 日志文件
├── tests/                     # 测试代码
├── run.py                     # 启动脚本
└── test_openai_compatibility.py  # 兼容性测试
```

### 本地开发

```bash
# 克隆并设置开发环境
git clone <repository-url>
cd vllm推理框架
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# 启动开发服务器
python run.py --mode dev

# 运行测试
python -m pytest tests/
python test_openai_compatibility.py

# 代码格式化
pip install black isort
black src/
isort src/
```

### 添加新的推理引擎

1. 在 `src/engines/` 中创建新的引擎实现
2. 继承 `InferenceEngine` 基类
3. 实现必需的抽象方法
4. 在 `src/engines/__init__.py` 中注册
5. 更新平台检测逻辑

### 提交代码

```bash
# 代码质量检查
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# 运行测试
python -m pytest tests/ -v
python test_openai_compatibility.py

# 提交代码
git add .
git commit -m "feat: 添加新功能"
git push origin feature/your-feature
```

---

## 🤝 贡献和支持

### 如何贡献

我们欢迎所有形式的贡献！

1. **🐛 报告 Bug**: [创建 Issue](../../issues)
2. **💡 功能建议**: [功能请求](../../issues)
3. **📝 改进文档**: 提交文档 PR
4. **💻 代码贡献**: Fork 项目并提交 PR

### 贡献流程

1. Fork 本项目
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建 Pull Request

### 开发规范

- ✅ 遵循 PEP 8 代码规范
- ✅ 添加类型提示 (Type Hints)
- ✅ 编写单元测试
- ✅ 更新相关文档
- ✅ 使用语义化版本号

### 获取帮助

- 📚 **详细文档**: [开发文档.md](开发文档.md) | [使用指南.md](使用指南.md)
- 🐛 **问题报告**: [GitHub Issues](../../issues)
- 💬 **社区讨论**: [GitHub Discussions](../../discussions)
- 📧 **技术支持**: 通过 Issue 联系我们

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。您可以自由使用、修改和分发本软件。

## 🙏 致谢

感谢以下开源项目的贡献：

- **[VLLM](https://github.com/vllm-project/vllm)** - 高性能推理引擎
- **[MLX](https://github.com/ml-explore/mlx)** - Apple Silicon 优化框架
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - 跨平台 CPU 推理
- **[Flask](https://flask.palletsprojects.com/)** - Web 应用框架
- **[Qwen](https://huggingface.co/Qwen)** - 优秀的开源模型

## 📈 项目状态

- ✅ **稳定版本**: v1.0.0
- 🚀 **活跃开发**: 持续更新和改进
- 🔄 **兼容性**: 完全 OpenAI API 兼容
- 🌍 **跨平台**: Windows, macOS, Linux 全支持

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

[⭐ Star this repo](../../stargazers) | [🐛 Report bug](../../issues) | [💡 Request feature](../../issues) | [📖 Documentation](开发文档.md)

---

*让每个开发者都能轻松拥有本地化的 AI 能力* 🚀

</div>