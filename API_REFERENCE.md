# 🔌 API 参考文档

## OpenAI 兼容接口

### 基础信息
- **Base URL**: `http://localhost:8001/v1`
- **认证**: 本地服务无需API Key，填入任意值即可
- **格式**: 100% OpenAI API 兼容

---

## 🎯 模型管理

### GET /v1/models
列出所有可用模型

**响应示例:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-0.5b",
      "object": "model",
      "created": 1693363200,
      "owned_by": "local",
      "permission": [],
      "root": "qwen-0.5b",
      "parent": null
    }
  ]
}
```

### GET /v1/models/{model_id}
获取特定模型信息

---

## 💬 聊天补全

### POST /v1/chat/completions
聊天补全接口，支持动态模型切换

**请求体:**
```json
{
  "model": "qwen-0.5b",          // 模型名称，支持动态切换
  "messages": [                   // 对话消息
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user", 
      "content": "Hello!"
    }
  ],
  "max_tokens": 100,             // 可选：最大token数
  "temperature": 0.7,            // 可选：创造性参数
  "top_p": 0.9,                  // 可选：核采样参数
  "stream": false,               // 可选：流式响应
  "stop": ["\\n"]                // 可选：停止词
}
```

**响应示例:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1693363200,
  "model": "qwen-0.5b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  },
  "system_fingerprint": "fp_1693363200"
}
```

### 流式响应
设置 `"stream": true` 启用流式响应

**响应格式:**
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1693363200,"model":"qwen-0.5b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1693363200,"model":"qwen-0.5b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: [DONE]
```

---

## 📝 文本补全

### POST /v1/completions
传统文本补全接口

**请求体:**
```json
{
  "model": "qwen-0.5b",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

---

## 🔧 系统管理

### GET /health
服务健康检查

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": 1693363200,
  "manager_initialized": true,
  "engines": {
    "mlx": {
      "status": "healthy",
      "loaded_models": 2
    }
  },
  "models": 2,
  "issues": []
}
```

### GET /v1/system/status
获取详细系统状态

**响应示例:**
```json
{
  "initialized": true,
  "total_engines": 1,
  "registered_models": 3,
  "loaded_models": 2,
  "max_concurrent_models": 3,
  "models": [
    {
      "name": "qwen-0.5b",
      "engine_type": "mlx",
      "status": "loaded",
      "loaded_at": "2024-01-01T12:00:00",
      "memory_usage": "2.1GB",
      "context_length": 4096
    }
  ],
  "engines": {
    "mlx": {
      "status": "healthy",
      "memory_usage": "4.2GB"
    }
  }
}
```

---

## 🚨 错误处理

### 错误格式
所有错误都遵循OpenAI格式：

```json
{
  "error": {
    "message": "模型未找到: invalid-model",
    "type": "not_found_error",
    "param": null,
    "code": "model_not_found"
  }
}
```

### 常见错误码

| HTTP状态码 | 错误类型 | 描述 |
|-----------|---------|------|
| 400 | invalid_request_error | 请求格式错误 |
| 404 | not_found_error | 模型或资源未找到 |
| 429 | rate_limit_error | 请求频率过高 |
| 500 | server_error | 服务器内部错误 |
| 503 | server_error | 服务未就绪 |

---

## 🔄 动态模型切换

### 特性说明
- **零停机切换**: 运行时动态加载/卸载模型
- **自动加载**: 调用未加载模型时自动加载
- **智能卸载**: 超出并发限制时自动卸载旧模型
- **并发安全**: 多请求并发访问同一模型安全

### 切换示例
```python
import openai

client = openai.OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8001/v1"
)

# 第一次调用自动加载模型
response1 = client.chat.completions.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Hello"}]
)

# 切换到另一个模型，自动加载
response2 = client.chat.completions.create(
    model="GLM-4.5V",
    messages=[{"role": "user", "content": "你好"}]
)
```

---

## 🛠️ 客户端集成

### Python OpenAI Client
```python
import openai

client = openai.OpenAI(
    api_key="sk-dummy",  # 任意值
    base_url="http://localhost:8001/v1"
)

response = client.chat.completions.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### cURL
```bash
curl -X POST http://localhost:8001/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-dummy" \\
  -d '{
    "model": "qwen-0.5b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### n8n 节点配置
1. **Connection**: Custom
2. **Base URL**: `http://localhost:8001/v1`
3. **API Key**: 任意值 (如: `sk-local`)
4. **Model**: 在节点中选择具体模型

### LangChain
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-dummy",
    base_url="http://localhost:8001/v1",
    model="qwen-0.5b"
)

response = llm.invoke("Hello!")
```

---

## ⚡ 性能优化

### 并发控制
- 默认支持最多3个模型同时运行
- 通过 `MAX_CONCURRENT_MODELS` 配置调整
- 自动LRU清理策略

### 缓存机制
- 模型推理结果缓存
- 可通过配置启用/禁用
- TTL过期自动清理

### 内存管理
- 智能内存监控
- 低内存时自动卸载模型
- 支持内存限制配置