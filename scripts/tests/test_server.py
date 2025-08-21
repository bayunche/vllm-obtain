#!/usr/bin/env python3
"""
简化的测试服务器启动脚本
避免编码问题
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_config
from src.api.app import create_app

def main():
    print("启动VLLM推理测试服务...")
    
    # 设置环境变量避免负载均衡模式
    os.environ['INFERENCE_MODE'] = 'single'
    os.environ['HOST'] = '127.0.0.1'
    os.environ['PORT'] = '8001'
    
    # 获取配置
    config = get_config()
    
    # 创建应用
    print(f"服务地址: http://{config.host}:{config.port}")
    print(f"推理引擎: {config.inference_engine}")
    
    # 创建Flask应用
    app = create_app(config)
    
    try:
        app.run(
            host=config.host,
            port=config.port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("服务已停止")

if __name__ == "__main__":
    main()