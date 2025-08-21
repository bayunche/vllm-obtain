"""
VLLM 推理引擎实现
提供高性能 CUDA GPU 推理支持 (Linux/Windows)
"""

import time
import asyncio
from typing import Optional, AsyncGenerator, List, Dict, Any

from ..core.inference_engine import InferenceEngine, ModelInfo, ModelStatus
from ..core.inference_engine import InferenceRequest, InferenceResponse, EngineConfig
from ..core.exceptions import ModelLoadError, InferenceError, EngineNotSupportedError
from ..utils import get_logger, performance_monitor


class VllmEngine(InferenceEngine):
    """VLLM 推理引擎"""
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.vllm_engines = {}  # 存储加载的 VLLM 引擎实例
        self.sampling_params_cache = {}
    
    async def initialize(self) -> bool:
        """初始化 VLLM 引擎"""
        try:
            # 检查平台兼容性
            import platform
            if platform.system().lower() == "darwin":
                raise EngineNotSupportedError("vllm", "macOS")
            
            # 导入 VLLM 模块
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            self.vllm = LLM
            self.SamplingParams = SamplingParams
            self.AsyncLLMEngine = AsyncLLMEngine
            self.AsyncEngineArgs = AsyncEngineArgs
            
            # 检查 CUDA 是否可用
            if not self._is_cuda_available():
                self.logger.warning("CUDA 不可用，VLLM 性能可能受限")
            
            self.logger.info("VLLM 引擎初始化成功")
            self._initialized = True
            return True
            
        except ImportError as e:
            self.logger.error(f"VLLM 引擎初始化失败: {e}")
            self.logger.error("请安装 VLLM: pip install vllm")
            return False
        except EngineNotSupportedError as e:
            self.logger.error(str(e))
            return False
        except Exception as e:
            self.logger.error(f"VLLM 引擎初始化异常: {e}")
            return False
    
    def _is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @performance_monitor("load_model")
    async def load_model(self, model_name: str, model_path: str, **kwargs) -> bool:
        """
        加载 VLLM 模型
        
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
            
            self.logger.info(f"开始加载VLLM模型: {model_name} from {model_path}")
            
            # 设置 VLLM 引擎参数
            engine_args = self.AsyncEngineArgs(
                model=model_path,
                tokenizer=model_path,
                tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
                dtype=kwargs.get("dtype", "auto"),
                max_model_len=kwargs.get("max_model_len", self.config.max_sequence_length),
                gpu_memory_utilization=self.config.max_gpu_memory,
                trust_remote_code=kwargs.get("trust_remote_code", True),
                disable_log_stats=True,
                disable_log_requests=True,
                max_num_seqs=kwargs.get("max_num_seqs", 256),
                max_num_batched_tokens=kwargs.get("max_num_batched_tokens", None),
            )
            
            # 在执行器中创建异步引擎
            start_time = time.time()
            engine = self.AsyncLLMEngine.from_engine_args(engine_args)
            load_time = time.time() - start_time
            
            # 存储引擎实例
            self.vllm_engines[model_name] = engine
            
            # 更新模型信息
            model_info.status = ModelStatus.LOADED
            model_info.loaded_at = time.time()
            model_info.context_length = engine_args.max_model_len
            model_info.memory_usage = self._get_gpu_memory_usage()
            
            self.logger.info(f"VLLM模型加载成功: {model_name} (耗时: {load_time:.2f}s)")
            return True
            
        except Exception as e:
            self._update_model_status(model_name, ModelStatus.ERROR, str(e))
            self.logger.error(f"VLLM模型加载失败: {model_name} - {e}")
            
            # 清理失败的加载
            if model_name in self.vllm_engines:
                del self.vllm_engines[model_name]
            
            raise ModelLoadError(model_name, str(e))
    
    @performance_monitor("unload_model")
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name not in self.models:
            self.logger.warning(f"模型未找到: {model_name}")
            return False
        
        try:
            self._update_model_status(model_name, ModelStatus.UNLOADING)
            
            # 停止引擎
            if model_name in self.vllm_engines:
                engine = self.vllm_engines[model_name]
                # VLLM 引擎的清理方法可能因版本而异
                if hasattr(engine, 'shutdown'):
                    await engine.shutdown()
                del self.vllm_engines[model_name]
            
            # 清理采样参数缓存
            if model_name in self.sampling_params_cache:
                del self.sampling_params_cache[model_name]
            
            # 删除模型信息
            del self.models[model_name]
            
            self.logger.info(f"VLLM模型卸载成功: {model_name}")
            return True
            
        except Exception as e:
            self._update_model_status(model_name, ModelStatus.ERROR, str(e))
            self.logger.error(f"VLLM模型卸载失败: {model_name} - {e}")
            return False
    
    def _get_sampling_params(self, request: InferenceRequest):
        """获取采样参数"""
        cache_key = f"{request.model_name}_{request.temperature}_{request.top_p}_{request.top_k}"
        
        if cache_key not in self.sampling_params_cache:
            sampling_params = self.SamplingParams(
                n=1,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop_sequences,
                skip_special_tokens=True,
                spaces_between_special_tokens=False
            )
            self.sampling_params_cache[cache_key] = sampling_params
        
        return self.sampling_params_cache[cache_key]
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理生成"""
        if not self.is_model_loaded(request.model_name):
            raise InferenceError(f"模型未加载: {request.model_name}")
        
        try:
            engine = self.vllm_engines[request.model_name]
            prompt = request.to_prompt()
            sampling_params = self._get_sampling_params(request)
            
            self.logger.debug(f"开始VLLM推理: {request.model_name} - {prompt[:100]}...")
            
            start_time = time.time()
            
            # 生成请求ID
            from vllm.utils import random_uuid
            request_id = random_uuid()
            
            # 添加请求到引擎
            results_generator = engine.generate(prompt, sampling_params, request_id)
            
            # 等待完成
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            inference_time = time.time() - start_time
            
            if final_output is None:
                raise InferenceError("VLLM推理未返回结果")
            
            # 解析结果
            output = final_output.outputs[0]
            completion_text = output.text
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(output.token_ids)
            total_tokens = prompt_tokens + completion_tokens
            
            tokens_per_second = completion_tokens / inference_time if inference_time > 0 else 0
            
            response = InferenceResponse(
                model_name=request.model_name,
                text=completion_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                inference_time=inference_time,
                tokens_per_second=tokens_per_second,
                finish_reason=output.finish_reason or "stop",
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
            self.logger.error(f"VLLM推理失败: {request.model_name} - {e}")
            raise InferenceError(str(e), request.model_name)
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """流式推理生成"""
        if not self.is_model_loaded(request.model_name):
            raise InferenceError(f"模型未加载: {request.model_name}")
        
        try:
            engine = self.vllm_engines[request.model_name]
            prompt = request.to_prompt()
            sampling_params = self._get_sampling_params(request)
            
            # 生成请求ID
            from vllm.utils import random_uuid
            request_id = random_uuid()
            
            # 开始流式生成
            results_generator = engine.generate(prompt, sampling_params, request_id)
            
            previous_text = ""
            async for request_output in results_generator:
                if request_output.outputs:
                    current_text = request_output.outputs[0].text
                    # 只返回新生成的部分
                    new_text = current_text[len(previous_text):]
                    if new_text:
                        yield new_text
                        previous_text = current_text
            
        except Exception as e:
            self.logger.error(f"VLLM流式推理失败: {request.model_name} - {e}")
            raise InferenceError(str(e), request.model_name)
    
    def _get_gpu_memory_usage(self) -> Optional[str]:
        """获取GPU内存使用情况"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                cached = torch.cuda.memory_reserved() / 1024**3  # GB
                return f"{allocated:.1f}GB/{cached:.1f}GB"
        except ImportError:
            pass
        return None
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理VLLM引擎资源")
        
        # 关闭所有引擎
        for model_name, engine in self.vllm_engines.items():
            try:
                if hasattr(engine, 'shutdown'):
                    await engine.shutdown()
                self.logger.info(f"VLLM引擎已关闭: {model_name}")
            except Exception as e:
                self.logger.error(f"关闭VLLM引擎失败 {model_name}: {e}")
        
        # 清理缓存
        self.vllm_engines.clear()
        self.sampling_params_cache.clear()
        
        await super().cleanup()


if __name__ == "__main__":
    # 测试VLLM引擎
    import asyncio
    
    async def test_vllm_engine():
        print("=== VLLM 引擎测试 ===")
        
        config = EngineConfig(
            engine_type="vllm",
            device_type="cuda",
            max_sequence_length=4096,
            max_gpu_memory=0.8
        )
        
        engine = VllmEngine(config)
        
        # 初始化
        success = await engine.initialize()
        if not success:
            print("❌ VLLM引擎初始化失败")
            return
        
        print("✅ VLLM引擎初始化成功")
        
        # 获取引擎状态
        status = engine.get_engine_status()
        print(f"✅ 引擎状态: {status}")
        
        # 健康检查
        health = await engine.health_check()
        print(f"✅ 健康检查: {health}")
        
        await engine.cleanup()
        print("✅ 资源清理完成")
    
    asyncio.run(test_vllm_engine())