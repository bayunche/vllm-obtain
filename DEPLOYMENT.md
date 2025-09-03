# 🚀 部署指南

## 🎯 部署方式选择

| 部署方式 | 适用场景 | 难度 | 性能 |
|---------|----------|------|------|
| **本地开发** | 开发测试 | ⭐ | 中等 |
| **生产单机** | 小规模生产 | ⭐⭐ | 高 |
| **Docker容器** | 标准化部署 | ⭐⭐⭐ | 高 |
| **负载均衡集群** | 大规模生产 | ⭐⭐⭐⭐ | 极高 |

---

## 🔧 环境准备

### 系统要求

#### macOS (推荐)
```bash
# 最低配置
- macOS 12.0+
- Apple Silicon (M1/M2/M3/M4)
- 16GB RAM
- 50GB 可用存储

# 推荐配置  
- macOS 14.0+
- M3 Ultra / M4 Pro
- 64GB RAM
- 500GB 可用存储
```

#### Linux
```bash
# 最低配置
- Ubuntu 20.04+ / CentOS 8+
- NVIDIA GTX 1080+ (8GB VRAM)
- 16GB RAM
- 50GB 可用存储

# 推荐配置
- Ubuntu 22.04+
- NVIDIA RTX 4090 (24GB VRAM)  
- 32GB RAM
- 1TB NVMe SSD
```

#### Windows  
```bash
# 最低配置
- Windows 10/11
- NVIDIA GTX 1070+ (8GB VRAM)
- 16GB RAM
- 50GB 可用存储

# 推荐配置
- Windows 11
- NVIDIA RTX 4080+ (16GB VRAM)
- 32GB RAM
- 1TB NVMe SSD
```

---

## 🏠 本地开发部署

### 快速启动
```bash
# 1. 克隆项目
git clone <your-repo-url>
cd vllm-obtain

# 2. 自动安装依赖
python run.py

# 3. 注册模型
python register_models.py

# 4. 启动服务
python run.py
```

### 配置调优
```bash
# 编辑配置文件
vim .env.mac     # macOS
vim .env.linux   # Linux
vim .env.windows # Windows

# 关键参数
MAX_CONCURRENT_MODELS=2        # 并发模型数
MAX_CONCURRENT_REQUESTS=10     # 并发请求数  
MEMORY_LIMIT=80               # 内存使用限制(%)
REQUEST_TIMEOUT=300           # 请求超时(秒)
```

---

## 🏭 生产环境部署

### 使用 Gunicorn (推荐)
```bash
# 安装 Gunicorn
pip install gunicorn gevent

# 启动服务
gunicorn -c gunicorn.conf.py src.api.app:application

# 后台运行
nohup gunicorn -c gunicorn.conf.py src.api.app:application > gunicorn.log 2>&1 &
```

### Gunicorn 配置文件
```python
# gunicorn.conf.py
import multiprocessing
import os

# 基础配置
bind = "0.0.0.0:8001"
workers = 1  # MLX只支持单进程，VLLM可增加
worker_class = "gevent"
worker_connections = 500

# 性能调优
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 300
keepalive = 2

# 日志配置
accesslog = "./logs/gunicorn_access.log"
errorlog = "./logs/gunicorn_error.log" 
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程管理
pid = "./gunicorn.pid"
daemon = False  # 设为True后台运行

# 资源限制
worker_tmp_dir = "/dev/shm"  # 使用内存临时目录
tmp_upload_dir = None
```

### 使用 Waitress (Windows推荐)
```bash
# 安装 Waitress
pip install waitress

# 启动服务
waitress-serve --host=0.0.0.0 --port=8001 --threads=10 src.api.app:application
```

---

## 🐳 Docker 部署

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 创建必要目录
RUN mkdir -p models logs cache

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8001/health || exit 1

# 启动命令
CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.api.app:application"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  vllm-inference:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./cache:/app/cache
    environment:
      - INFERENCE_ENGINE=auto
      - MAX_CONCURRENT_MODELS=3
      - DEBUG=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # 可选：Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vllm-inference
    restart: unless-stopped
```

### 构建和运行
```bash
# 构建镜像
docker build -t vllm-inference:latest .

# 运行容器
docker run -d \\
  --name vllm-service \\
  -p 8001:8001 \\
  -v $(pwd)/models:/app/models \\
  -v $(pwd)/logs:/app/logs \\
  --restart unless-stopped \\
  vllm-inference:latest

# 使用 Docker Compose
docker-compose up -d
```

---

## ⚖️ 负载均衡部署

### 多实例负载均衡
```bash
# 启动负载均衡器
python -m src.api.load_balanced_app --workers 3 --port 8001

# 或使用配置文件
python -m src.api.load_balanced_app --config load_balance.yaml
```

### 负载均衡配置
```yaml
# load_balance.yaml
load_balancer:
  port: 8001
  workers: 3
  strategy: "round_robin"  # round_robin, least_loaded, hash
  
instances:
  - host: "127.0.0.1"
    port: 8002
    weight: 1
    max_connections: 100
  - host: "127.0.0.1"  
    port: 8003
    weight: 1
    max_connections: 100
  - host: "127.0.0.1"
    port: 8004
    weight: 2
    max_connections: 150

health_check:
  interval: 30
  timeout: 5
  path: "/health"
```

### Nginx 负载均衡
```nginx
upstream vllm_backend {
    least_conn;
    server 127.0.0.1:8002 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8004 weight=2 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 流式响应支持
        proxy_buffering off;
        proxy_cache off;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # 健康检查
    location /health {
        access_log off;
        proxy_pass http://vllm_backend/health;
    }
}
```

---

## 🔒 安全配置

### SSL/TLS 配置
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # 其他配置...
}
```

### 防火墙配置
```bash
# Ubuntu/Debian
ufw allow 8001/tcp
ufw allow 22/tcp
ufw enable

# CentOS/RHEL
firewall-cmd --permanent --add-port=8001/tcp
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --reload
```

### API 访问控制
```python
# 在 .env 文件中配置
API_KEY_REQUIRED=true
ALLOWED_HOSTS=127.0.0.1,your-domain.com
CORS_ORIGINS=https://your-frontend.com,https://n8n.your-domain.com
RATE_LIMIT_PER_MINUTE=60
```

---

## 📊 监控和日志

### 日志配置
```python
# .env 配置
LOG_LEVEL=INFO
LOG_DIR=./logs
ENABLE_LOG_FILE=true
LOG_RETENTION_DAYS=30
LOG_ROTATE_SIZE=100MB
ENABLE_METRICS=true
METRICS_INTERVAL=60
```

### 监控指标
```bash
# 查看系统状态
curl http://localhost:8001/v1/system/status

# 查看健康状态  
curl http://localhost:8001/health

# 查看日志
tail -f logs/inference.log
tail -f logs/metrics.log
```

### Prometheus 集成 (可选)
```python
# 安装依赖
pip install prometheus-client

# 在 .env 中启用
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=9090
```

---

## 🚨 故障排除

### 常见问题

#### 模型加载失败
```bash
# 检查模型路径
ls -la models/

# 检查配置
cat .env.mac

# 查看详细错误
tail -f logs/inference.log
```

#### 内存不足
```bash
# 检查内存使用
free -h

# 调整配置
vim .env.mac
# 减少 MAX_CONCURRENT_MODELS
# 降低 MEMORY_LIMIT
```

#### 端口占用
```bash
# 检查端口占用
lsof -i :8001
netstat -tulpn | grep 8001

# 杀死占用进程
kill -9 <PID>
```

#### 权限问题
```bash
# 修复权限
chmod +x run.py
chmod -R 755 models/
chown -R $USER:$USER logs/ cache/
```

### 性能调优

#### 内存优化
```bash
# .env 配置调优
MEMORY_LIMIT=70                # 降低内存限制
MAX_CONCURRENT_MODELS=2        # 减少并发模型
MLX_MEMORY_FRACTION=0.6        # 降低MLX内存使用
```

#### 并发优化
```bash
# 根据硬件调整
MAX_CONCURRENT_REQUESTS=20     # Mac Studio M3 Ultra
MAX_CONCURRENT_REQUESTS=30     # Linux RTX 4090  
MAX_CONCURRENT_REQUESTS=15     # Windows GTX 1080
```

#### 磁盘优化
```bash
# 使用 SSD 存储模型
MODEL_BASE_PATH=/fast/ssd/models

# 启用磁盘缓存
ENABLE_CACHING=true
CACHE_SIZE=2000
```

---

## 📈 扩容指南

### 垂直扩容 (升级硬件)
- **CPU**: 增加核心数提升并发处理
- **内存**: 增加RAM支持更多模型并发
- **存储**: 使用NVMe SSD提升模型加载速度
- **GPU**: 升级显卡提升推理性能

### 水平扩容 (增加实例)
1. **多实例部署**: 启动多个服务实例
2. **负载均衡**: 使用Nginx或专业负载均衡器
3. **健康检查**: 自动检测和剔除故障实例
4. **会话粘性**: 根据需要配置会话保持

### 自动扩容
```bash
# 使用 systemd 自动重启
sudo systemctl enable vllm-inference
sudo systemctl start vllm-inference

# 使用 Docker Swarm 或 Kubernetes
# 实现容器编排和自动扩容
```