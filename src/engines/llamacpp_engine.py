"""
LlamaCpp 推理引擎实现
提供跨平台 CPU/GPU 推理支持，作为保底方案
"""

import os
import time
import asyncio
import threading
from typing import Optional, AsyncGenerator
from pathlib import Path

from ..core.inference_engine import InferenceEngine, ModelInfo, ModelStatus
from ..core.inference_engine import InferenceRequest, InferenceResponse, EngineConfig
from ..core.exceptions import ModelLoadError, InferenceError, ResourceLimitError
from ..utils import get_logger, performance_monitor


class LlamaCppEngine(InferenceEngine):
    """LlamaCpp 推理引擎"""
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.llama_models = {}  # 存储加载的 llama-cpp 模型实例
        self._lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """初始化 LlamaCpp 引擎"""
        try:
            # 检查 llama-cpp-python 是否可用
            import llama_cpp
            self.llama_cpp = llama_cpp
            
            self.logger.info("LlamaCpp 引擎初始化成功")
            self.logger.info(f"llama-cpp-python 版本: {llama_cpp.__version__}")
            
            # 设置设备类型
            self._setup_device()
            
            self._initialized = True
            return True
            
        except ImportError as e:
            self.logger.error(f"LlamaCpp 引擎初始化失败: {e}")
            self.logger.error("请安装 llama-cpp-python: pip install llama-cpp-python")
            return False
        except Exception as e:
            self.logger.error(f"LlamaCpp 引擎初始化异常: {e}")
            return False
    
    def _setup_device(self):
        """设置设备配置"""
        device_type = self.config.device_type
        
        if device_type == "auto":
            # 自动检测最佳设备
            if self._is_cuda_available():
                self.device_type = "cuda"
                self.n_gpu_layers = -1  # 全部层在GPU
            elif self._is_metal_available():
                self.device_type = "metal"
                self.n_gpu_layers = -1
            else:
                self.device_type = "cpu"
                self.n_gpu_layers = 0
        else:
            self.device_type = device_type
            if device_type in ["cuda", "metal"]:
                self.n_gpu_layers = -1
            else:
                self.n_gpu_layers = 0
        
        self.logger.info(f"设备类型: {self.device_type}, GPU层数: {self.n_gpu_layers}")
    
    def _is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _is_metal_available(self) -> bool:
        """检查Metal是否可用 (macOS)"""
        import platform
        return platform.system().lower() == "darwin"
    
    @performance_monitor("load_model")
    async def load_model(self, model_name: str, model_path: str, **kwargs) -> bool:
        """
        加载LlamaCpp模型
        
        Args:
            model_name: 模型名称
            model_path: 模型文件路径 (.gguf格式)
            **kwargs: 额外参数
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化")
        
        # 检查模型是否已加载
        if self.is_model_loaded(model_name):
            self.logger.warning(f"模型已加载: {model_name}")
            return True
        
        try:
            # 创建模型信息
            model_info = self._create_model_info(model_name, model_path)
            model_info.status = ModelStatus.LOADING
            self.models[model_name] = model_info
            
            self.logger.info(f"开始加载LlamaCpp模型: {model_name} from {model_path}")
            
            # 检查模型文件是否存在
            model_file = Path(model_path)
            if not model_file.exists():
                raise ModelLoadError(model_name, f"模型文件不存在: {model_path}")
            
            # 获取文件大小
            file_size_mb = model_file.stat().st_size / 1024 / 1024
            model_info.size_mb = file_size_mb
            
            # 检查内存是否足够
            if not self._check_memory_limit(file_size_mb):
                raise ResourceLimitError("内存", file_size_mb, self._get_available_memory())
            
            # 设置模型参数
            model_params = {
                "model_path": str(model_path),
                "n_ctx": kwargs.get("context_length", self.config.max_sequence_length),
                "n_threads": kwargs.get("n_threads", self.config.max_cpu_threads),
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": False,
                "seed": kwargs.get("seed", -1),
                "f16_kv": kwargs.get("f16_kv", True),
                "use_mlock": kwargs.get("use_mlock", False),
                "use_mmap": kwargs.get("use_mmap", True),
            }
            
            # 在线程池中加载模型 (避免阻塞事件循环)
            loop = asyncio.get_event_loop()
            
            def _load_model():
                with self._lock:
                    return self.llama_cpp.Llama(**model_params)
            
            start_time = time.time()
            llama_model = await loop.run_in_executor(None, _load_model)
            load_time = time.time() - start_time
            
            # 存储模型实例
            self.llama_models[model_name] = llama_model
            
            # 更新模型信息
            model_info.status = ModelStatus.LOADED
            model_info.loaded_at = time.time()
            model_info.context_length = model_params["n_ctx"]
            model_info.memory_usage = self._estimate_memory_usage(file_size_mb)
            
            self.logger.info(f"模型加载成功: {model_name} (耗时: {load_time:.2f}s, 大小: {file_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            self._update_model_status(model_name, ModelStatus.ERROR, str(e))
            self.logger.error(f"模型加载失败: {model_name} - {e}")
            
            # 清理失败的加载
            if model_name in self.llama_models:
                del self.llama_models[model_name]
            
            raise ModelLoadError(model_name, str(e))
    
    @performance_monitor("unload_model")
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name not in self.models:
            self.logger.warning(f"模型未找到: {model_name}")
            return False
        
        try:
            self._update_model_status(model_name, ModelStatus.UNLOADING)
            
            # 删除模型实例
            if model_name in self.llama_models:
                with self._lock:
                    del self.llama_models[model_name]
            
            # 删除模型信息
            del self.models[model_name]
            
            self.logger.info(f"模型卸载成功: {model_name}")
            return True
            
        except Exception as e:
            self._update_model_status(model_name, ModelStatus.ERROR, str(e))
            self.logger.error(f"模型卸载失败: {model_name} - {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理生成"""
        if not self.is_model_loaded(request.model_name):
            raise InferenceError(f"模型未加载: {request.model_name}")
        
        try:
            llama_model = self.llama_models[request.model_name]
            prompt = request.to_prompt()
            
            self.logger.debug(f"开始推理: {request.model_name} - {prompt[:100]}...")
            
            # 构建生成参数
            generate_params = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repeat_penalty": request.repetition_penalty,
                "stop": request.stop_sequences or [],
                "echo": False
            }
            
            # 在线程池中执行推理
            loop = asyncio.get_event_loop()
            
            def _generate():
                with self._lock:
                    start_time = time.time()
                    result = llama_model(prompt, **generate_params)
                    inference_time = time.time() - start_time
                    return result, inference_time
            
            result, inference_time = await loop.run_in_executor(None, _generate)
            
            # 解析结果
            completion_text = result["choices"][0]["text"]
            prompt_tokens = result["usage"]["prompt_tokens"]
            completion_tokens = result["usage"]["completion_tokens"]
            total_tokens = result["usage"]["total_tokens"]
            
            tokens_per_second = completion_tokens / inference_time if inference_time > 0 else 0
            
            response = InferenceResponse(
                model_name=request.model_name,
                text=completion_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                inference_time=inference_time,
                tokens_per_second=tokens_per_second,
                finish_reason=result["choices"][0].get("finish_reason", "stop"),
                request_id=request.request_id
            )
            
            # 记录推理日志
            self.logger.log_inference(
                model_name=request.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                inference_time=inference_time,
                request_id=request.request_id
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"推理失败: {request.model_name} - {e}")
            raise InferenceError(str(e), request.model_name)
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """流式推理生成"""
        if not self.is_model_loaded(request.model_name):
            raise InferenceError(f"模型未加载: {request.model_name}")
        
        try:
            llama_model = self.llama_models[request.model_name]
            prompt = request.to_prompt()
            
            # 构建生成参数
            generate_params = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repeat_penalty": request.repetition_penalty,
                "stop": request.stop_sequences or [],
                "stream": True,
                "echo": False
            }
            
            # 使用线程安全的流式生成
            def _stream_generate():
                with self._lock:
                    return llama_model(prompt, **generate_params)
            
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(None, _stream_generate)
            
            # 逐步返回生成的文本
            for chunk in stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                    elif chunk["choices"][0].get("finish_reason"):
                        break
            
        except Exception as e:
            self.logger.error(f"流式推理失败: {request.model_name} - {e}")
            raise InferenceError(str(e), request.model_name)
    
    def _check_memory_limit(self, model_size_mb: float) -> bool:
        """检查内存限制"""
        try:
            import psutil
            
            # 获取可用内存
            available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
            
            # 预留一些内存给系统 (至少2GB)
            required_memory_mb = model_size_mb * 1.5 + 2048
            
            return available_memory_mb > required_memory_mb
            
        except ImportError:
            # 如果没有psutil，跳过检查
            return True
    
    def _get_available_memory(self) -> float:
        """获取可用内存 (MB)"""
        try:
            import psutil
            return psutil.virtual_memory().available / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _estimate_memory_usage(self, file_size_mb: float) -> str:
        """估算内存使用量"""
        # 通常模型在内存中的大小约为文件大小的1.2-1.5倍
        estimated_mb = file_size_mb * 1.3
        return f"{estimated_mb:.1f}MB"
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理LlamaCpp引擎资源")
        
        # 卸载所有模型
        model_names = list(self.llama_models.keys())
        for model_name in model_names:
            try:
                await self.unload_model(model_name)
            except Exception as e:
                self.logger.error(f"清理模型失败 {model_name}: {e}")
        
        # 清理模型字典
        self.llama_models.clear()
        
        await super().cleanup()


if __name__ == "__main__":
    # 测试LlamaCpp引擎
    import asyncio
    
    async def test_llamacpp_engine():
        print("=== LlamaCpp 引擎测试 ===")
        
        config = EngineConfig(
            engine_type="llama_cpp",
            device_type="auto",
            max_sequence_length=2048,
            max_cpu_threads=4
        )
        
        engine = LlamaCppEngine(config)
        
        # 初始化
        success = await engine.initialize()
        if not success:
            print("❌ 引擎初始化失败")
            return
        
        print("✅ 引擎初始化成功")
        
        # 注意：这里需要真实的GGUF模型文件进行测试
        # model_path = "/path/to/your/model.gguf"
        
        print("🔍 为了完整测试，需要提供真实的GGUF模型文件路径")
        print("请参考 README.md 中的模型下载和配置说明")
        
        # 获取引擎状态
        status = engine.get_engine_status()
        print(f"✅ 引擎状态: {status}")
        
        # 健康检查
        health = await engine.health_check()
        print(f"✅ 健康检查: {health}")
        
        await engine.cleanup()
        print("✅ 资源清理完成")
    
    asyncio.run(test_llamacpp_engine())