# 🚀 VLLM 推理服务 - 快速参考

## 📦 一键安装

```bash
# 自动检测平台并安装
./install.sh
```

## 🎯 快速启动

```bash
# 激活环境
source venv/bin/activate

# 生产模式 (推荐)
python run.py --mode prod

# 开发模式
python run.py --mode dev
```

## 🔧 平台配置

| 平台 | 配置文件 | 推理引擎 | Worker数 | 特点 |
|------|----------|----------|----------|------|
| **Mac (M1/M2/M3)** | `.env.mac` | MLX | 1 | Metal加速，功耗低 |
| **Linux GPU** | `.env.linux` | VLLM | 4+ | CUDA加速，高并发 |
| **Windows** | `.env.windows` | LlamaCpp | 2 | 兼容性好 |

## 📡 API 端点

### 健康检查
```bash
curl http://localhost:8001/health
```

### 聊天补全 (OpenAI兼容)
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-0.5b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 模型列表
```bash
curl http://localhost:8001/v1/models
```

## 🧪 测试

```bash
# OpenAI 兼容性测试
python scripts/tests/test_openai_compatibility.py

# 并发性能测试
python scripts/benchmarks/concurrent_test.py

# 综合测试
python scripts/benchmarks/comprehensive_test.py
```

## 📁 项目结构

```
vllm-obtain/
├── src/           # 源代码
│   ├── api/       # Web API
│   ├── core/      # 核心逻辑
│   ├── engines/   # 推理引擎
│   └── utils/     # 工具函数
├── scripts/       # 测试脚本
│   ├── tests/     # 功能测试
│   └── benchmarks/# 性能测试
├── docs/          # 文档
├── models/        # 模型文件
├── logs/          # 日志
├── README.md      # 主文档
├── run.py         # 启动脚本
└── install.sh     # 安装脚本
```

## 🌟 核心特性

- ✅ **100% OpenAI API 兼容**
- ✅ **智能平台检测**
- ✅ **自动引擎选择**
- ✅ **生产级稳定性**
- ✅ **完整测试覆盖**

## 🆘 常见问题

### Mac: MLX Metal错误
```bash
# 解决方案: 使用单worker配置
export WORKERS=1
cp .env.mac .env
```

### Linux: CUDA内存不足
```bash
# 解决方案: 降低内存使用
export CUDA_MEMORY_FRACTION=0.6
```

### 端口被占用
```bash
# 解决方案: 杀掉进程
lsof -ti:8001 | xargs kill -9
```

## 📚 更多文档

- [完整README](README.md)
- [生产部署指南](docs/README_PRODUCTION.md)
- [项目结构说明](PROJECT_STRUCTURE.md)
- [测试报告](docs/Mac_Studio_完整测试报告_20250821.md)

---
**版本**: v1.0.0 | **最后更新**: 2025-08-21