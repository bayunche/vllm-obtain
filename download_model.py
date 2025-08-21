#!/usr/bin/env python3
"""
下载测试模型
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_test_model():
    """下载小型测试模型"""
    print("下载测试模型...")
    
    # 创建模型目录
    model_dir = Path("models/Qwen2.5-0.5B-Instruct-GGUF")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载小型的 Qwen2.5-0.5B 量化模型（GGUF格式）
        model_file = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_0.gguf",
            local_dir="models/Qwen2.5-0.5B-Instruct-GGUF",
            cache_dir=".cache"
        )
        
        print(f"模型下载成功: {model_file}")
        return model_file
        
    except Exception as e:
        print(f"模型下载失败: {e}")
        
        # 尝试下载更小的模型
        print("尝试下载备用小模型...")
        try:
            model_file = hf_hub_download(
                repo_id="microsoft/DialoGPT-small",
                filename="pytorch_model.bin", 
                local_dir="models/DialoGPT-small",
                cache_dir=".cache"
            )
            print(f"备用模型下载成功: {model_file}")
            return model_file
        except Exception as e2:
            print(f"备用模型下载也失败: {e2}")
            return None

if __name__ == "__main__":
    download_test_model()