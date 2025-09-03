# VLLM 跨平台推理服务 - Mac Studio 生产部署指南

## 🌟 项目概述

VLLM 跨平台推理服务是一个高性能的大语言模型推理框架，专为Mac Studio等Apple Silicon设备优化。支持MLX、VLLM、llama.cpp等多种推理引擎，提供OpenAI兼容的API接口。

### 核心特性

- ✅ **Apple Silicon 优化**: 专为M系列芯片优化的MLX引擎
- ✅ **OpenAI API 兼容**: 完全兼容OpenAI ChatGPT API格式
- ✅ **多引擎支持**: MLX、VLLM、llama.cpp自动检测和切换
- ✅ **生产级部署**: Gunicorn + Gevent异步并发处理
- ✅ **完善监控**: 详细的性能日志和系统监控
- ✅ **高并发支持**: 单worker支持20+并发请求

## 📊 性能指标 (Mac Studio M3 Ultra)

| 指标 | 开发模式(Flask) | 生产模式(Gunicorn) |
|------|----------------|-------------------|
| 最大并发 | 7 (崩溃) | 20+ (稳定) |
| 吞吐量 | 2.86 请求/秒 | 2.64 请求/秒 |
| 成功率 | 100% (低并发) | 100% (高并发) |
| 响应时间 | 0.25秒 | 0.77-7.07秒 |
| Token生成速度 | 186.5 tokens/秒 | 0.7-6.7 tokens/秒 |

## 🚀 快速开始

### 环境要求

- **操作系统**: macOS 15.0+
- **芯片**: Apple M系列 (M1/M2/M3/M4)
- **Python**: 3.8+
- **内存**: 16GB+ (推荐32GB+)
- **存储**: 10GB+ 可用空间

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd vllm-obtain
```

2. **创建虚拟环境**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载模型**
```bash
# 使用 ModelScope (国内推荐)
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models/qwen-0.5b')"
```

5. **配置环境**
```bash
cp .env.example .env
# 编辑 .env 文件，设置合适的配置
```

## ⚙️ 配置说明

### 环境配置 (.env)

```bash
# 服务配置
HOST=127.0.0.1
PORT=8001
DEBUG=false
WORKERS=1

# 推理引擎配置
INFERENCE_ENGINE=mlx
DEVICE_TYPE=mps
DEFAULT_MODEL=qwen-0.5b
MAX_CONCURRENT_MODELS=1

# 日志配置
LOG_LEVEL=INFO
ENABLE_LOG_FILE=true
LOG_RETENTION_DAYS=30

# 性能配置
REQUEST_TIMEOUT=300
MAX_CONCURRENT_REQUESTS=100
ENABLE_CACHING=true
CACHE_SIZE=1000
```

### 关键配置说明

- **INFERENCE_ENGINE=mlx**: Mac Studio必须使用MLX引擎
- **WORKERS=1**: MLX引擎只支持单worker模式
- **DEVICE_TYPE=mps**: Apple Silicon Metal性能着色器
- **MAX_CONCURRENT_MODELS=1**: 单模型实例避免内存冲突

## 🎯 部署模式

### 开发模式

```bash
# 使用内置启动脚本
python run.py --mode dev

# 或直接运行
python -m src.api.app
```

**特点**:
- 自动重载代码更改
- 详细调试日志
- 单线程处理
- 适合开发调试

### 生产模式 (推荐)

```bash
# 使用启动脚本
python run.py --mode prod

# 或直接使用Gunicorn
source venv/bin/activate
gunicorn --bind 127.0.0.1:8001 \
         --workers 1 \
         --worker-class gevent \
         --worker-connections 1000 \
         --timeout 300 \
         --keep-alive 5 \
         --max-requests 1000 \
         --max-requests-jitter 100 \
         --access-logfile - \
         --error-logfile - \
         src.api.app:application
```

**特点**:
- 高并发异步处理
- 生产级稳定性
- 完善错误处理
- 性能监控

## 📡 API 使用

### 健康检查

```bash
curl http://127.0.0.1:8001/health
```

### OpenAI兼容接口

```bash
curl -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-0.5b",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下人工智能"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 模型管理

```bash
# 查看可用模型
curl http://127.0.0.1:8001/v1/models

# 系统状态
curl http://127.0.0.1:8001/v1/system/status

# 加载模型
curl -X POST http://127.0.0.1:8001/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen-0.5b"}'
```

## 🔧 性能优化

### Mac Studio 专用优化

1. **MLX引擎优化**
```bash
# .env 配置
INFERENCE_ENGINE=mlx
DEVICE_TYPE=mps
MLX_MEMORY_FRACTION=0.8
```

2. **并发配置**
```bash
# 单worker + Gevent
WORKERS=1
WORKER_CLASS=gevent
WORKER_CONNECTIONS=1000
```

3. **内存管理**
```bash
# 模型缓存
ENABLE_CACHING=true
CACHE_SIZE=1000
MAX_CONCURRENT_MODELS=1
```

### 推荐硬件配置

| 配置 | 最低要求 | 推荐配置 | 高性能配置 |
|------|----------|----------|------------|
| 芯片 | M1 8核 | M2 Pro | M3 Ultra |
| 内存 | 16GB | 32GB | 64GB+ |
| 存储 | 256GB SSD | 512GB SSD | 1TB+ SSD |
| 并发数 | 5 | 10 | 20+ |

## 📈 监控和日志

### 日志系统

```bash
# 日志目录结构
logs/
├── app.log              # 应用日志
├── api_requests.log     # API请求日志  
├── model_operations.log # 模型操作日志
├── system_metrics.log   # 系统指标日志
└── errors.log          # 错误日志
```

### 关键监控指标

1. **性能指标**
   - 响应时间 < 5秒 (P95)
   - 吞吐量 > 2.0 请求/秒
   - 成功率 > 99%
   - Token生成速度 > 100 tokens/秒

2. **系统指标**
   - CPU使用率 < 60%
   - 内存使用率 < 70%
   - 磁盘I/O延迟 < 100ms
   - 网络延迟 < 50ms

3. **业务指标**
   - 模型加载时间 < 10秒
   - 推理错误率 < 1%
   - 并发处理数 > 10

### Prometheus 监控 (可选)

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm-api'
    static_configs:
      - targets: ['127.0.0.1:8001']
    scrape_interval: 5s
    metrics_path: '/metrics'
```

## 🛠️ 故障排查

### 常见问题

1. **Metal库构建错误**
```
[metal::Device] Unable to build metal library from source
```
**解决方案**: 使用单worker模式 (`--workers 1`)

2. **内存不足**
```
MLX Error: Unable to allocate memory
```
**解决方案**: 减少`max_tokens`或使用更小的模型

3. **模型加载失败**
```
Model not found: models/qwen-0.5b
```
**解决方案**: 检查模型路径，重新下载模型

4. **端口占用**
```
Address already in use: 127.0.0.1:8001
```
**解决方案**: 
```bash
lsof -ti:8001 | xargs kill -9
```

### 调试模式

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python run.py --mode dev

# 查看实时日志
tail -f logs/app.log
```

## 🔄 更新和维护

### 定期维护

1. **日志轮转**
```bash
# 清理30天前的日志
find logs/ -name "*.log" -mtime +30 -delete
```

2. **模型更新**
```bash
# 更新模型
python scripts/update_models.py
```

3. **依赖更新**
```bash
pip list --outdated
pip install --upgrade package_name
```

### 备份策略

```bash
# 配置文件备份
cp .env .env.backup.$(date +%Y%m%d)

# 日志备份
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/

# 模型备份
rsync -av models/ backup/models/
```

## 🚨 安全建议

1. **网络安全**
   - 仅绑定内网IP (127.0.0.1)
   - 使用防火墙限制访问
   - 启用HTTPS (生产环境)

2. **认证授权**
   - 实现API密钥验证
   - 限制请求频率
   - 记录访问日志

3. **数据安全**
   - 不记录敏感输入内容
   - 定期清理临时文件
   - 加密存储配置信息

## 📋 最佳实践

### 开发规范

1. **代码质量**
   - 遵循PEP8规范
   - 编写单元测试
   - 定期代码审查

2. **配置管理**
   - 使用环境变量
   - 分离开发/生产配置
   - 版本化配置文件

3. **错误处理**
   - 优雅处理异常
   - 详细错误日志
   - 用户友好错误信息

### 生产部署

1. **服务管理**
```bash
# systemd 服务配置
[Unit]
Description=VLLM Inference Service
After=network.target

[Service]
Type=simple
User=vllm
WorkingDirectory=/path/to/vllm-obtain
ExecStart=/path/to/vllm-obtain/venv/bin/python run.py --mode prod
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

2. **负载均衡** (多实例)
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

## 📚 参考资料

- [MLX 官方文档](https://ml-explore.github.io/mlx/build/html/index.html)
- [Gunicorn 配置指南](https://docs.gunicorn.org/en/stable/configure.html)
- [OpenAI API 参考](https://platform.openai.com/docs/api-reference)
- [Flask 生产部署](https://flask.palletsprojects.com/en/2.3.x/deploying/)

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

**版本**: v1.0  
**最后更新**: 2025-08-21  
**测试环境**: Mac Studio M3 Ultra, macOS 15.6, Python 3.13