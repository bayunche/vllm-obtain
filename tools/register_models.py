#!/usr/bin/env python3
"""
多模型注册脚本
自动发现并注册所有可用模型
"""

import asyncio
import os
from pathlib import Path
from src.core.model_manager import ModelManager, ModelConfig
from src.utils import get_logger, get_config

async def register_all_models():
    """注册所有可用的模型"""
    logger = get_logger()
    config = get_config()
    
    print("🔍 开始扫描可用模型...")
    
    # 获取模型管理器
    manager = ModelManager(config)
    await manager.initialize()
    
    # 扫描模型目录
    models_path = Path(config.model_base_path)
    available_models = []
    
    if not models_path.exists():
        print(f"❌ 模型目录不存在: {models_path}")
        return
    
    # 扫描所有模型目录
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            # 检查是否包含config.json (直接模型目录)
            if (model_dir / 'config.json').exists():
                available_models.append({
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'type': 'direct'
                })
                print(f"✅ 发现模型: {model_dir.name} -> {model_dir}")
            
            # 检查子目录(ModelScope缓存结构)
            else:
                for sub_dir in model_dir.rglob('config.json'):
                    parent_dir = sub_dir.parent
                    model_name = f"{model_dir.name}"
                    available_models.append({
                        'name': model_name,
                        'path': str(parent_dir),
                        'type': 'modelscope'
                    })
                    print(f"✅ 发现模型: {model_name} -> {parent_dir}")
    
    if not available_models:
        print("❌ 未发现任何可用模型")
        return
    
    print(f"\n📋 共发现 {len(available_models)} 个模型")
    
    # 注册所有模型
    registered_count = 0
    for i, model in enumerate(available_models):
        try:
            model_config = ModelConfig(
                name=model['name'],
                path=model['path'],
                auto_load=False,  # 不自动加载，需要时再加载
                priority=i + 2    # 默认模型优先级为1，其他从2开始
            )
            
            manager.register_model(model_config)
            registered_count += 1
            print(f"✅ 已注册: {model['name']}")
            
        except Exception as e:
            print(f"❌ 注册失败 {model['name']}: {e}")
    
    print(f"\n🎉 成功注册 {registered_count}/{len(available_models)} 个模型")
    
    # 显示系统状态
    status = await manager.get_system_status()
    print(f"\n📊 系统状态:")
    print(f"  - 已注册模型: {status['registered_models']}")
    print(f"  - 已加载模型: {status['loaded_models']}")
    print(f"  - 最大并发数: {status['max_concurrent_models']}")
    
    # 显示所有注册的模型
    print(f"\n📝 已注册模型列表:")
    for model_config in manager.list_registered_models():
        print(f"  - {model_config.name}: {model_config.path}")
    
    return manager

async def test_model_switching():
    """测试模型切换功能"""
    print("\n🧪 测试模型动态切换...")
    
    manager = await register_all_models()
    
    if not manager:
        return
    
    # 获取已注册的模型
    registered_models = manager.list_registered_models()
    
    if len(registered_models) < 2:
        print("⚠️ 需要至少2个模型才能测试切换功能")
        return
    
    # 测试前两个模型
    test_models = registered_models[:2]
    
    for model_config in test_models:
        try:
            print(f"\n🔄 测试加载模型: {model_config.name}")
            success = await manager.load_model(model_config.name)
            
            if success:
                print(f"✅ 模型 {model_config.name} 加载成功")
                
                # 显示模型信息
                info = manager.get_model_info(model_config.name)
                if info:
                    print(f"  - 状态: {info['status']}")
                    print(f"  - 路径: {info['path']}")
                    
            else:
                print(f"❌ 模型 {model_config.name} 加载失败")
                
        except Exception as e:
            print(f"❌ 测试模型 {model_config.name} 失败: {e}")
    
    # 显示当前加载的模型
    loaded_models = manager.list_loaded_models()
    print(f"\n📊 当前加载的模型 ({len(loaded_models)}):")
    for model in loaded_models:
        print(f"  - {model['name']} ({model['engine_type']})")

def main():
    """主函数"""
    print("🚀 多模型注册器")
    print("=" * 50)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 测试模式
        asyncio.run(test_model_switching())
    else:
        # 仅注册模式
        asyncio.run(register_all_models())
    
    print("\n💡 使用方法:")
    print("1. 通过API切换模型:")
    print("   POST /v1/chat/completions")
    print("   {\"model\": \"your-model-name\", \"messages\": [...]}")
    print("")
    print("2. 查看可用模型:")
    print("   GET /v1/models")
    print("")
    print("3. 运行测试:")
    print("   python register_models.py --test")

if __name__ == "__main__":
    main()