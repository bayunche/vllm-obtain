"""
推理引擎抽象层
定义统一的推理引擎接口，支持不同的推理后端
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
from enum import Enum

from ..utils import get_logger


class ModelStatus(Enum):
    """模型状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class EngineConfig:
    """推理引擎配置"""
    engine_type: str  # vllm, mlx, llama_cpp
    device_type: str = "auto"  # auto, cuda, mps, cpu
    max_gpu_memory: float = 0.8
    max_cpu_threads: int = 8
    enable_streaming: bool = True
    batch_size: int = 1
    max_sequence_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    # 引擎特定参数
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    status: ModelStatus = ModelStatus.UNLOADED
    size_mb: Optional[float] = None
    parameters: Optional[int] = None
    context_length: Optional[int] = None
    loaded_at: Optional[float] = None
    memory_usage: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status.value,
            "size_mb": self.size_mb,
            "parameters": self.parameters,
            "context_length": self.context_length,
            "loaded_at": self.loaded_at,
            "memory_usage": self.memory_usage,
            "error_message": self.error_message
        }


@dataclass
class InferenceRequest:
    """推理请求"""
    model_name: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    
    # OpenAI 兼容参数
    messages: Optional[List[Dict[str, str]]] = None
    request_id: Optional[str] = None
    user: Optional[str] = None
    
    # 多模态支持
    multimodal_data: Optional[List[Dict[str, Any]]] = None  # 多模态内容列表
    image_path: Optional[str] = None  # 图片文件路径（向后兼容）
    
    def to_prompt(self) -> str:
        """转换为推理提示词"""
        if self.messages:
            # 将对话消息转换为prompt
            prompt_parts = []
            for message in self.messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt_parts.append("Assistant:")
            return "\n".join(prompt_parts)
        
        return self.prompt


@dataclass
class InferenceResponse:
    """推理响应"""
    model_name: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    inference_time: float
    tokens_per_second: float
    finish_reason: str = "stop"
    request_id: Optional[str] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI API格式"""
        return {
            "id": f"chatcmpl-{self.request_id or int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.text
                },
                "finish_reason": self.finish_reason
            }],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens
            }
        }


class InferenceEngine(ABC):
    """推理引擎抽象基类"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = get_logger()
        self.models: Dict[str, ModelInfo] = {}
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化推理引擎
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    async def load_model(self, model_name: str, model_path: str, **kwargs) -> bool:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
            **kwargs: 额外参数
            
        Returns:
            是否加载成功
        """
        pass
    
    @abstractmethod
    async def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否卸载成功
        """
        pass
    
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        执行推理生成
        
        Args:
            request: 推理请求
            
        Returns:
            推理响应
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """
        流式推理生成
        
        Args:
            request: 推理请求
            
        Yields:
            生成的文本片段
        """
        pass
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[ModelInfo]:
        """列出所有模型"""
        return list(self.models.values())
    
    def get_loaded_models(self) -> List[ModelInfo]:
        """获取已加载的模型"""
        return [model for model in self.models.values() 
                if model.status == ModelStatus.LOADED]
    
    def is_model_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        model = self.models.get(model_name)
        return model is not None and model.status == ModelStatus.LOADED
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        loaded_models = self.get_loaded_models()
        
        return {
            "engine_type": self.config.engine_type,
            "initialized": self._initialized,
            "total_models": len(self.models),
            "loaded_models": len(loaded_models),
            "device_type": self.config.device_type,
            "max_gpu_memory": self.config.max_gpu_memory,
            "models": [model.to_dict() for model in loaded_models]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        status = {
            "status": "healthy",
            "engine_type": self.config.engine_type,
            "initialized": self._initialized,
            "timestamp": time.time()
        }
        
        try:
            # 检查是否有可用的模型
            loaded_models = self.get_loaded_models()
            if not loaded_models:
                status["status"] = "no_models"
                status["message"] = "没有已加载的模型"
                return status
            
            # 可以添加更多健康检查逻辑
            # 例如：简单推理测试、资源使用检查等
            
        except Exception as e:
            status["status"] = "error"
            status["message"] = str(e)
            self.logger.error(f"健康检查失败: {e}")
        
        return status
    
    def _update_model_status(self, model_name: str, status: ModelStatus, 
                           error_message: str = None):
        """更新模型状态"""
        if model_name in self.models:
            self.models[model_name].status = status
            if error_message:
                self.models[model_name].error_message = error_message
            
            self.logger.info(f"模型状态更新: {model_name} -> {status.value}")
    
    def _create_model_info(self, model_name: str, model_path: str, **kwargs) -> ModelInfo:
        """创建模型信息对象"""
        return ModelInfo(
            name=model_name,
            path=model_path,
            **kwargs
        )
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理推理引擎资源")
        
        # 卸载所有已加载的模型
        loaded_models = self.get_loaded_models()
        for model in loaded_models:
            try:
                await self.unload_model(model.name)
            except Exception as e:
                self.logger.error(f"卸载模型失败 {model.name}: {e}")
        
        self._initialized = False
        self.logger.info("推理引擎资源清理完成")


class DummyEngine(InferenceEngine):
    """虚拟推理引擎 (用于测试)"""
    
    async def initialize(self) -> bool:
        """初始化虚拟引擎"""
        self.logger.info("初始化虚拟推理引擎")
        self._initialized = True
        return True
    
    async def load_model(self, model_name: str, model_path: str, **kwargs) -> bool:
        """加载虚拟模型"""
        self.logger.info(f"加载虚拟模型: {model_name}")
        
        model_info = self._create_model_info(
            model_name, model_path,
            size_mb=100.0,
            parameters=1000000,
            context_length=2048,
            loaded_at=time.time(),
            memory_usage="100MB"
        )
        model_info.status = ModelStatus.LOADED
        
        self.models[model_name] = model_info
        return True
    
    async def unload_model(self, model_name: str) -> bool:
        """卸载虚拟模型"""
        if model_name in self.models:
            self.logger.info(f"卸载虚拟模型: {model_name}")
            del self.models[model_name]
            return True
        return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """虚拟推理生成"""
        import random
        
        # 模拟推理时间
        inference_time = random.uniform(0.5, 2.0)
        await asyncio.sleep(inference_time)
        
        # 生成虚拟响应
        completion_text = f"这是对 '{request.prompt}' 的虚拟回复。"
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(completion_text.split())
        
        return InferenceResponse(
            model_name=request.model_name,
            text=completion_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            inference_time=inference_time,
            tokens_per_second=completion_tokens / inference_time,
            request_id=request.request_id
        )
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """虚拟流式生成"""
        import asyncio
        
        words = f"这是对 '{request.prompt}' 的虚拟流式回复。".split()
        
        for word in words:
            await asyncio.sleep(0.1)  # 模拟生成延迟
            yield word + " "


if __name__ == "__main__":
    # 测试推理引擎抽象层
    import asyncio
    
    async def test_dummy_engine():
        print("=== 推理引擎抽象层测试 ===")
        
        config = EngineConfig(engine_type="dummy")
        engine = DummyEngine(config)
        
        # 初始化
        await engine.initialize()
        print("✅ 引擎初始化完成")
        
        # 加载模型
        await engine.load_model("test-model", "/path/to/model")
        print("✅ 模型加载完成")
        
        # 推理测试
        request = InferenceRequest(
            model_name="test-model",
            prompt="你好，世界！",
            max_tokens=50
        )
        
        response = await engine.generate(request)
        print(f"✅ 推理完成: {response.text}")
        print(f"Token统计: {response.prompt_tokens}+{response.completion_tokens}={response.total_tokens}")
        print(f"推理速度: {response.tokens_per_second:.1f} tokens/s")
        
        # 流式推理测试
        print("\n🔄 流式推理测试:")
        async for chunk in engine.generate_stream(request):
            print(chunk, end="", flush=True)
        print("\n")
        
        # 获取引擎状态
        status = engine.get_engine_status()
        print(f"✅ 引擎状态: {status}")
        
        # 健康检查
        health = await engine.health_check()
        print(f"✅ 健康检查: {health}")
        
        # 清理
        await engine.cleanup()
        print("✅ 资源清理完成")
    
    asyncio.run(test_dummy_engine())