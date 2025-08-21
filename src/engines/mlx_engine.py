"""
MLX 推理引擎实现
提供 Apple Silicon (M1/M2/M3) Metal 优化推理支持
"""

import time
import asyncio
from typing import Optional, AsyncGenerator
from pathlib import Path

from ..core.inference_engine import InferenceEngine, ModelInfo, ModelStatus
from ..core.inference_engine import InferenceRequest, InferenceResponse, EngineConfig
from ..core.exceptions import ModelLoadError, InferenceError, EngineNotSupportedError
from ..utils import get_logger, performance_monitor


class MlxEngine(InferenceEngine):
    """MLX 推理引擎"""
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.mlx_models = {}  # 存储加载的 MLX 模型实例
        self.tokenizers = {}  # 存储对应的分词器
    
    async def initialize(self) -> bool:
        """初始化 MLX 引擎"""
        try:
            # 检查平台兼容性
            import platform
            if platform.system().lower() != "darwin":
                raise EngineNotSupportedError("mlx", platform.system())
            
            if platform.machine().lower() not in ["arm64", "aarch64"]:
                raise EngineNotSupportedError("mlx", f"architecture: {platform.machine()}")
            
            # 导入 MLX 模块
            import mlx.core as mx
            import mlx_lm
            from mlx_lm import load, generate
            
            self.mx = mx
            self.mlx_lm = mlx_lm
            self.mlx_load = load
            self.mlx_generate = generate
            
            self.logger.info("MLX 引擎初始化成功")
            self.logger.info(f"MLX 版本: {mx.__version__}")
            
            self._initialized = True
            return True
            
        except ImportError as e:
            self.logger.error(f"MLX 引擎初始化失败: {e}")
            self.logger.error("请安装 MLX: pip install mlx-lm")
            return False
        except EngineNotSupportedError as e:
            self.logger.error(str(e))
            return False
        except Exception as e:
            self.logger.error(f"MLX 引擎初始化异常: {e}")
            return False
    
    @performance_monitor("load_model")
    async def load_model(self, model_name: str, model_path: str, **kwargs) -> bool:
        """
        加载 MLX 模型
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
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
            
            self.logger.info(f"开始加载MLX模型: {model_name} from {model_path}")
            
            # 在线程池中加载模型 (避免阻塞事件循环)
            loop = asyncio.get_event_loop()
            
            def _load_model():
                # 加载模型和分词器
                model, tokenizer = self.mlx_load(model_path)
                return model, tokenizer
            
            start_time = time.time()
            model, tokenizer = await loop.run_in_executor(None, _load_model)
            load_time = time.time() - start_time
            
            # 存储模型和分词器
            self.mlx_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # 更新模型信息
            model_info.status = ModelStatus.LOADED
            model_info.loaded_at = time.time()
            model_info.context_length = kwargs.get("context_length", self.config.max_sequence_length)
            model_info.memory_usage = self._estimate_memory_usage(model)
            
            self.logger.info(f"MLX模型加载成功: {model_name} (耗时: {load_time:.2f}s)")
            return True
            
        except Exception as e:
            self._update_model_status(model_name, ModelStatus.ERROR, str(e))
            self.logger.error(f"MLX模型加载失败: {model_name} - {e}")
            
            # 清理失败的加载
            if model_name in self.mlx_models:
                del self.mlx_models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            raise ModelLoadError(model_name, str(e))
    
    @performance_monitor("unload_model")
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name not in self.models:
            self.logger.warning(f"模型未找到: {model_name}")
            return False
        
        try:
            self._update_model_status(model_name, ModelStatus.UNLOADING)
            
            # 删除模型和分词器实例
            if model_name in self.mlx_models:
                del self.mlx_models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            # 强制内存回收
            import gc
            gc.collect()
            
            # 删除模型信息
            del self.models[model_name]
            
            self.logger.info(f"MLX模型卸载成功: {model_name}")
            return True
            
        except Exception as e:
            self._update_model_status(model_name, ModelStatus.ERROR, str(e))
            self.logger.error(f"MLX模型卸载失败: {model_name} - {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理生成"""
        if not self.is_model_loaded(request.model_name):
            raise InferenceError(f"模型未加载: {request.model_name}")
        
        try:
            model = self.mlx_models[request.model_name]
            tokenizer = self.tokenizers[request.model_name]
            prompt = request.to_prompt()
            
            self.logger.debug(f"开始MLX推理: {request.model_name} - {prompt[:100]}...")
            
            # 在线程池中执行推理
            loop = asyncio.get_event_loop()
            
            def _generate():
                start_time = time.time()
                
                # 使用基础的MLX generate函数（不带参数）
                response = self.mlx_generate(
                    model, 
                    tokenizer, 
                    prompt,
                    max_tokens=request.max_tokens,
                    verbose=False
                )
                
                inference_time = time.time() - start_time
                return response, inference_time
            
            response_text, inference_time = await loop.run_in_executor(None, _generate)
            
            # 计算token统计
            prompt_tokens = len(tokenizer.encode(prompt))
            # 只计算生成的部分
            generated_text = response_text[len(prompt):] if response_text.startswith(prompt) else response_text
            completion_tokens = len(tokenizer.encode(generated_text))
            total_tokens = prompt_tokens + completion_tokens
            
            tokens_per_second = completion_tokens / inference_time if inference_time > 0 else 0
            
            response = InferenceResponse(
                model_name=request.model_name,
                text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                inference_time=inference_time,
                tokens_per_second=tokens_per_second,
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
            self.logger.error(f"MLX推理失败: {request.model_name} - {e}")
            raise InferenceError(str(e), request.model_name)
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """流式推理生成"""
        if not self.is_model_loaded(request.model_name):
            raise InferenceError(f"模型未加载: {request.model_name}")
        
        try:
            model = self.mlx_models[request.model_name]
            tokenizer = self.tokenizers[request.model_name]
            prompt = request.to_prompt()
            
            # MLX 流式生成实现
            loop = asyncio.get_event_loop()
            
            def _stream_generate():
                # 使用 MLX 的流式生成 API
                from mlx_lm.generate import stream_generate
                
                return stream_generate(
                    model, 
                    tokenizer, 
                    prompt,
                    max_tokens=request.max_tokens
                )
            
            # 获取生成器
            generator = await loop.run_in_executor(None, _stream_generate)
            
            # 逐步返回生成的文本
            for token in generator:
                if token:
                    yield token
            
        except Exception as e:
            self.logger.error(f"MLX流式推理失败: {request.model_name} - {e}")
            raise InferenceError(str(e), request.model_name)
    
    def _estimate_memory_usage(self, model) -> str:
        """估算模型内存使用量"""
        try:
            # 尝试获取模型参数数量
            param_count = sum(param.size for param in model.parameters())
            
            # 估算内存使用 (假设float16，每个参数2字节)
            memory_bytes = param_count * 2
            memory_mb = memory_bytes / 1024 / 1024
            
            return f"{memory_mb:.1f}MB"
        except:
            return "未知"
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理MLX引擎资源")
        
        # 清理所有模型
        self.mlx_models.clear()
        self.tokenizers.clear()
        
        # 强制内存回收
        import gc
        gc.collect()
        
        await super().cleanup()


if __name__ == "__main__":
    # 测试MLX引擎
    import asyncio
    
    async def test_mlx_engine():
        print("=== MLX 引擎测试 ===")
        
        config = EngineConfig(
            engine_type="mlx",
            device_type="mps",
            max_sequence_length=2048
        )
        
        engine = MlxEngine(config)
        
        # 初始化
        success = await engine.initialize()
        if not success:
            print("❌ MLX引擎初始化失败")
            return
        
        print("✅ MLX引擎初始化成功")
        
        # 获取引擎状态
        status = engine.get_engine_status()
        print(f"✅ 引擎状态: {status}")
        
        # 健康检查
        health = await engine.health_check()
        print(f"✅ 健康检查: {health}")
        
        await engine.cleanup()
        print("✅ 资源清理完成")
    
    asyncio.run(test_mlx_engine())