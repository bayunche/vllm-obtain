"""
MLX-VLM 多模态推理引擎
支持视觉语言模型的推理，如 GLM-4.5V
"""

import asyncio
import time
from typing import Dict, Optional, Any, List, AsyncGenerator
from pathlib import Path
import json

from ..core.inference_engine import (
    InferenceEngine, 
    EngineConfig,
    ModelInfo,
    ModelStatus,
    InferenceRequest,
    InferenceResponse
)
from ..core.exceptions import (
    EngineInitError,
    ModelLoadError,
    InferenceError
)
from ..utils import get_logger


class MLXVLMEngine(InferenceEngine):
    """MLX-VLM 多模态推理引擎实现"""
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.logger = get_logger()
        self.models = {}
        self.model_configs = {}
        self.model_infos = {}  # 模型信息字典
        self._inference_locks = {}  # 并发控制锁
        
    async def initialize(self) -> bool:
        """初始化引擎"""
        try:
            # 检查 MLX-VLM 是否可用
            import mlx_vlm
            import mlx
            from mlx_vlm import load, generate
            
            self.mlx_vlm = mlx_vlm
            self.mlx = mlx
            self.vlm_load = load
            self.vlm_generate = generate
            
            self.logger.info("MLX-VLM 引擎初始化成功")
            try:
                mlx_version = getattr(mlx, '__version__', 'unknown')
                self.logger.info(f"MLX 版本: {mlx_version}")
            except:
                self.logger.info("MLX 版本: 无法获取")
            
            return True
            
        except ImportError as e:
            self.logger.error(f"MLX-VLM 导入失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"MLX-VLM 初始化失败: {e}")
            return False
    
    async def load_model(
        self,
        model_name: str,
        model_path: str,
        **kwargs
    ) -> bool:
        """加载多模态模型"""
        try:
            self.logger.info(f"开始加载多模态模型: {model_name} 从 {model_path}")
            
            # 添加并发控制锁
            if model_name not in self._inference_locks:
                self._inference_locks[model_name] = asyncio.Lock()
            
            # 检查模型路径
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise ModelLoadError(model_name, f"模型路径不存在: {model_path}")
            
            # 检查配置文件
            config_path = model_path_obj / "config.json"
            if not config_path.exists():
                raise ModelLoadError(model_name, "缺少 config.json 文件")
            
            # 读取模型配置
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # 获取模型类型
            model_type = config_data.get("model_type", "")
            architectures = config_data.get("architectures", [])
            
            self.logger.info(f"模型类型: {model_type}, 架构: {architectures}")
            
            # 尝试映射模型到 MLX-VLM 支持的格式
            model_id = str(model_path_obj)
            
            # 特殊处理 GLM-4.5V 模型
            if "glm" in model_type.lower() or any("GLM" in arch for arch in architectures):
                # GLM 模型可能需要特殊的加载参数
                self.logger.info("检测到 GLM 模型，尝试使用 MLX-VLM 加载")
                
                # 检查是否有预处理器配置
                preprocessor_config = model_path_obj / "preprocessor_config.json"
                if preprocessor_config.exists():
                    self.logger.info("找到预处理器配置")
            
            # 在异步执行器中加载模型
            loop = asyncio.get_event_loop()
            
            def _load_model():
                try:
                    # 加载模型和处理器
                    self.logger.info(f"正在加载模型: {model_id}")
                    model, processor = self.vlm_load(model_id)
                    return model, processor
                except Exception as e:
                    self.logger.error(f"模型加载失败: {e}")
                    # 尝试其他加载方式
                    try:
                        # 尝试直接从 HuggingFace 格式加载
                        from mlx_vlm.utils import load_model
                        model = load_model(model_id)
                        processor = None
                        return model, processor
                    except:
                        raise e
            
            try:
                model, processor = await loop.run_in_executor(None, _load_model)
            except Exception as e:
                # 如果 MLX-VLM 不支持该模型，提供详细错误信息
                error_msg = str(e)
                if "not supported" in error_msg.lower() or "glm4v" in error_msg.lower():
                    raise ModelLoadError(
                        model_name, 
                        f"MLX-VLM 暂不支持 {model_type} 模型。支持的模型包括: "
                        "LLaVA, Qwen-VL, Phi-3-Vision 等。"
                    )
                else:
                    raise ModelLoadError(model_name, f"加载失败: {error_msg}")
            
            # 保存模型和处理器
            self.models[model_name] = {
                "model": model,
                "processor": processor,
                "config": config_data,
                "loaded_at": time.time()
            }
            
            # 更新模型信息
            self.model_infos[model_name] = ModelInfo(
                name=model_name,
                path=model_path,
                status=ModelStatus.LOADED,
                size_mb=self._get_model_size(model_path_obj),
                parameters=config_data.get("num_parameters"),
                context_length=config_data.get("max_position_embeddings", 2048),
                loaded_at=time.time()
            )
            
            self.logger.info(f"多模态模型加载成功: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载多模态模型失败: {e}")
            
            # 记录失败状态
            self.model_infos[model_name] = ModelInfo(
                name=model_name,
                path=model_path,
                status=ModelStatus.ERROR,
                error_message=str(e)
            )
            
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name not in self.models:
            self.logger.warning(f"模型未加载: {model_name}")
            return False
        
        try:
            # 删除模型
            del self.models[model_name]
            
            # 删除锁
            if model_name in self._inference_locks:
                del self._inference_locks[model_name]
            
            # 更新状态
            if model_name in self.model_infos:
                self.model_infos[model_name].status = ModelStatus.UNLOADED
            
            # 清理内存
            import gc
            gc.collect()
            
            self.logger.info(f"模型卸载成功: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"卸载模型失败: {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """执行多模态推理"""
        model_name = request.model_name
        
        if model_name not in self.models:
            raise InferenceError(f"模型未加载: {model_name}")
        
        # 使用锁进行并发控制
        async with self._inference_locks[model_name]:
            try:
                model_data = self.models[model_name]
                model = model_data["model"]
                processor = model_data["processor"]
                
                # 处理输入
                prompt = request.prompt
                
                # 检查是否有多模态数据
                images = []
                
                # 向后兼容：检查旧的 image_path 字段
                if hasattr(request, 'image_path') and request.image_path:
                    from PIL import Image
                    images.append(Image.open(request.image_path))
                
                # 处理新的 multimodal_data
                if request.multimodal_data:
                    from PIL import Image
                    import io
                    import base64
                    
                    for mm_data in request.multimodal_data:
                        if mm_data.get('type') == 'image':
                            # 处理base64图片数据
                            image_data = mm_data.get('data')
                            if isinstance(image_data, bytes):
                                # 直接是二进制数据
                                image = Image.open(io.BytesIO(image_data))
                                images.append(image)
                            elif isinstance(image_data, str):
                                # base64字符串
                                try:
                                    image_bytes = base64.b64decode(image_data)
                                    image = Image.open(io.BytesIO(image_bytes))
                                    images.append(image)
                                except Exception as e:
                                    self.logger.warning(f"无法解码图片数据: {e}")
                
                # 构建输入
                if images:
                    # 多模态输入（图片+文本）
                    self.logger.info(f"处理多模态输入: {len(images)}张图片, 文本={prompt[:100]}...")
                    
                    # 使用处理器处理输入
                    if processor:
                        # 如果有多张图片，使用第一张（大多数VLM只支持单张图片）
                        image = images[0] if images else None
                        inputs = processor(text=prompt, images=image, return_tensors="mlx")
                    else:
                        # 如果没有处理器，尝试直接构建输入
                        inputs = {"prompt": prompt, "image": images[0] if images else None}
                else:
                    # 纯文本输入
                    inputs = {"prompt": prompt}
                
                # 设置生成参数
                generation_args = {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                }
                
                # 记录开始时间
                start_time = time.time()
                
                # 在异步执行器中运行生成
                loop = asyncio.get_event_loop()
                
                def _generate():
                    if images and processor:
                        # 多模态生成（使用第一张图片）
                        image = images[0]
                        
                        # 将PIL Image对象保存为临时文件
                        import tempfile
                        import os
                        
                        temp_path = None
                        try:
                            # 创建临时文件，使用更健壮的路径生成策略
                            temp_path = self._create_safe_temp_file(image)
                            if not temp_path:
                                raise RuntimeError("无法创建临时文件")
                            
                            # 使用临时文件路径进行推理
                            output = self.vlm_generate(
                                model, 
                                processor,
                                prompt,          # prompt作为第3个参数
                                image=temp_path, # image作为关键字参数
                                **generation_args
                            )
                            
                        finally:
                            # 清理临时文件
                            if temp_path and os.path.exists(temp_path):
                                try:
                                    os.unlink(temp_path)
                                except:
                                    pass
                    else:
                        # 纯文本生成
                        output = self.vlm_generate(
                            model,
                            processor,
                            prompt=prompt,
                            **generation_args
                        )
                    return output
                
                output = await loop.run_in_executor(None, _generate)
                
                # 计算时间和token
                generation_time = time.time() - start_time
                
                # 处理MLX-VLM的GenerationResult输出
                generated_text = self._extract_clean_text(output, prompt)
                
                # 获取精确的token统计（如果可用）
                if hasattr(output, 'prompt_tokens') and hasattr(output, 'generation_tokens'):
                    prompt_tokens = output.prompt_tokens
                    completion_tokens = output.generation_tokens
                    total_tokens = output.total_tokens
                else:
                    # 回退到估算
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(generated_text.split())
                    total_tokens = prompt_tokens + completion_tokens
                
                response = InferenceResponse(
                    text=generated_text,
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    inference_time=generation_time,
                    tokens_per_second=completion_tokens/generation_time if generation_time > 0 else 0,
                    finish_reason="stop"
                )
                
                self.logger.info(
                    f"多模态推理完成: 模型={model_name}, "
                    f"Token={total_tokens}, 时间={generation_time:.2f}s"
                )
                
                return response
                
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否是文件名过长的错误
                if "File name too long" in error_msg or "Errno 63" in error_msg:
                    self.logger.error("检测到文件名过长错误，这可能是系统临时目录配置问题")
                    self.logger.error("建议检查环境变量TMPDIR和系统临时目录设置")
                    raise InferenceError("临时文件创建失败，文件名过长。请检查系统临时目录配置。")
                
                # 检查是否是路径相关的错误
                elif "No such file or directory" in error_msg:
                    self.logger.error("临时文件路径创建失败")
                    raise InferenceError("临时文件创建失败，无法访问临时目录。")
                
                # 其他推理错误
                self.logger.error(f"多模态推理失败: {e}")
                raise InferenceError(f"推理失败: {error_msg}")
    
    async def generate_stream(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[str, None]:
        """流式生成（多模态）"""
        model_name = request.model_name
        
        if model_name not in self.models:
            raise InferenceError(f"模型未加载: {model_name}")
        
        # 使用锁进行并发控制
        async with self._inference_locks[model_name]:
            try:
                model_data = self.models[model_name]
                model = model_data["model"]
                processor = model_data["processor"]
                
                # 暂时使用非流式生成
                # MLX-VLM 的流式支持可能需要特殊处理
                response = await self.generate(request)
                
                # 模拟流式输出
                words = response.text.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.01)  # 模拟延迟
                    
            except Exception as e:
                self.logger.error(f"流式生成失败: {e}")
                yield f"[错误: {str(e)}]"
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.model_infos.get(model_name)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        loaded_models = [
            name for name, info in self.model_infos.items()
            if info.status == ModelStatus.LOADED
        ]
        
        return {
            "engine_type": "mlx_vlm",
            "initialized": True,
            "loaded_models": len(loaded_models),
            "models": loaded_models,
            "capabilities": [
                "text_generation",
                "image_understanding",
                "multimodal_reasoning"
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            import mlx
            
            loaded_count = sum(
                1 for info in self.model_infos.values()
                if info.status == ModelStatus.LOADED
            )
            
            if loaded_count == 0:
                return {
                    "status": "no_models",
                    "message": "没有已加载的多模态模型",
                    "engine_type": "mlx_vlm",
                    "initialized": True,
                    "timestamp": time.time()
                }
            
            return {
                "status": "healthy",
                "message": f"MLX-VLM 引擎运行正常，已加载 {loaded_count} 个模型",
                "engine_type": "mlx_vlm",
                "initialized": True,
                "models": loaded_count,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "engine_type": "mlx_vlm",
                "initialized": False,
                "timestamp": time.time()
            }
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("清理 MLX-VLM 引擎资源")
        
        # 卸载所有模型
        for model_name in list(self.models.keys()):
            await self.unload_model(model_name)
        
        self.models.clear()
        self.model_infos.clear()
        self._inference_locks.clear()
    
    def _get_model_size(self, model_path: Path) -> float:
        """计算模型大小（MB）"""
        total_size = 0
        for file in model_path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)
    
    def _create_safe_temp_file(self, image) -> Optional[str]:
        """
        安全地创建临时文件，避免文件名过长问题
        
        Args:
            image: PIL Image对象
            
        Returns:
            临时文件路径，失败时返回None
        """
        import tempfile
        import os
        import uuid
        
        # 尝试多种策略创建临时文件
        strategies = [
            # 策略1: 使用/tmp目录（Unix系统）
            lambda: ('/tmp', 'v_'),
            # 策略2: 使用系统临时目录，但用很短的前缀
            lambda: (tempfile.gettempdir(), 'v_'),
            # 策略3: 使用当前工作目录下的temp文件夹
            lambda: ('./temp', 'v_'),
            # 策略4: 使用UUID作为文件名（无前缀）
            lambda: (tempfile.gettempdir(), '')
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                temp_dir, prefix = strategy()
                
                # 确保目录存在
                if temp_dir != tempfile.gettempdir() and not os.path.exists(temp_dir):
                    os.makedirs(temp_dir, exist_ok=True)
                
                if prefix:
                    # 使用前缀
                    temp_fd, temp_path = tempfile.mkstemp(
                        suffix='.png', 
                        prefix=prefix,
                        dir=temp_dir
                    )
                else:
                    # 使用UUID生成简短文件名
                    filename = f"{uuid.uuid4().hex[:8]}.png"
                    temp_path = os.path.join(temp_dir, filename)
                    temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                
                os.close(temp_fd)  # 关闭文件描述符
                
                # 保存图片到临时文件
                image.save(temp_path, 'PNG')
                
                self.logger.debug(f"成功创建临时文件（策略{i+1}): {temp_path}")
                return temp_path
                
            except (OSError, IOError) as e:
                self.logger.warning(f"临时文件创建策略{i+1}失败: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"临时文件创建策略{i+1}异常: {e}")
                continue
        
        # 所有策略都失败了
        self.logger.error("所有临时文件创建策略都失败了")
        return None
    
    def _extract_clean_text(self, output, prompt: str) -> str:
        """
        从MLX-VLM输出中提取并清理文本
        
        Args:
            output: MLX-VLM的生成结果（可能是GenerationResult对象或字符串）
            prompt: 原始提示词
            
        Returns:
            清理后的生成文本
        """
        # 提取原始文本
        if hasattr(output, 'text'):
            # GenerationResult对象
            raw_text = output.text
        elif isinstance(output, dict):
            raw_text = output.get("text", str(output))
        else:
            raw_text = str(output)
        
        # 清理文本
        cleaned_text = self._clean_generated_text(raw_text, prompt)
        
        self.logger.debug(f"原始输出长度: {len(raw_text)}")
        self.logger.debug(f"清理后长度: {len(cleaned_text)}")
        
        return cleaned_text
    
    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """
        清理生成的文本，移除无用标记和格式
        
        Args:
            text: 原始生成文本
            prompt: 原始提示词
            
        Returns:
            清理后的文本
        """
        import re
        
        # 如果文本包含原始prompt，移除它
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        
        # 移除各种标记
        patterns_to_remove = [
            r'<\|im_start\|>.*?<\|im_end\|>',  # 移除对话标记
            r'<think>.*?</think>',             # 移除思考标记
            r'<\|begin_of_box\|>.*?<\|end_of_box\|>',  # 移除框标记
            r'<tool_call>.*?</tool_call>',     # 移除工具调用标记
            r'<script>.*?</script>',           # 移除脚本标记
            r'<table[^>]*>.*?</table>',        # 移除HTML表格
            r'<[^>]+>',                        # 移除其他HTML标记
            r'\*\*[^*]*\*\*',                  # 移除markdown粗体
            r'```[^`]*```',                    # 移除代码块
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理多余的空白字符
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # 多个连续换行合并为两个
        text = re.sub(r'[ \t]+', ' ', text)  # 多个空格合并为一个
        text = text.strip()
        
        # 如果清理后文本太短，尝试从原始文本中提取JSON
        if len(text) < 50:
            json_match = re.search(r'\{[^{}]*"items"[^{}]*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
            else:
                # 如果没有找到有效内容，返回一个错误消息
                text = "抱歉，我无法正确处理这个图片。请确保图片清晰可见并重试。"
        
        return text