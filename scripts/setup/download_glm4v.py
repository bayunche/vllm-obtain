#!/usr/bin/env python3
"""
GLM-4.5V 模型下载脚本
从 ModelScope 下载 GLM-4.5V 模型到本地
"""

import os
import sys
from pathlib import Path

def download_glm4v():
    """下载 GLM-4.5V 模型"""
    try:
        # 安装必要的依赖
        print("🔧 安装必要依赖...")
        os.system("pip install addict -q")
        
        from modelscope import snapshot_download
        
        # 模型配置
        model_id = 'ZhipuAI/GLM-4.5V'
        local_dir = './models/GLM-4.5V'
        
        print(f"📦 开始下载模型: {model_id}")
        print(f"📁 下载路径: {local_dir}")
        print("⏳ 这可能需要较长时间，请耐心等待...")
        
        # 创建本地目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 下载模型
        cache_dir = snapshot_download(
            model_id,
            cache_dir=local_dir,
            revision='master'
        )
        
        print(f"✅ 模型下载成功!")
        print(f"📍 模型位置: {cache_dir}")
        
        # 检查下载的文件
        model_path = Path(cache_dir)
        if model_path.exists():
            files = list(model_path.rglob('*'))
            print(f"📊 共下载 {len(files)} 个文件")
            
            # 显示主要文件
            important_files = []
            for f in files:
                if f.is_file() and f.name in ['config.json', 'pytorch_model.bin', 'model.safetensors', 
                                             'tokenizer.json', 'tokenizer_config.json']:
                    important_files.append(f)
            
            if important_files:
                print("📋 关键文件:")
                for f in important_files:
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  ✓ {f.name}: {size_mb:.1f}MB")
        
        return cache_dir
        
    except ImportError as e:
        print(f"❌ 依赖导入失败: {e}")
        print("请安装 modelscope: pip install modelscope")
        return None
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def main():
    """主函数"""
    print("🚀 GLM-4.5V 模型下载器")
    print("=" * 50)
    
    # 检查是否已存在
    local_path = Path("./models/GLM-4.5V")
    if local_path.exists() and any(local_path.iterdir()):
        print("⚠️  检测到模型已存在")
        choice = input("是否重新下载? (y/N): ").strip().lower()
        if choice != 'y':
            print("取消下载")
            return
    
    # 下载模型
    result = download_glm4v()
    
    if result:
        print(f"\n🎉 下载完成!")
        print(f"模型路径: {result}")
        print("\n📝 接下来需要:")
        print("1. 更新 .env.mac 配置文件")
        print("2. 重启 VLLM 服务")
        print("3. 测试新模型")
    else:
        print("\n❌ 下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main()