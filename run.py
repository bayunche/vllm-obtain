#!/usr/bin/env python3
"""
VLLM 跨平台推理服务启动脚本
支持开发和生产环境启动
"""

import os
import sys
import argparse
import signal
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_config, setup_logger
from src.api.app import create_app
from src.api.load_balanced_app import create_load_balanced_app


def handle_signal(signum, frame):
    """信号处理"""
    print(f"\n收到信号 {signum}，正在优雅关闭服务...")
    sys.exit(0)


def run_development():
    """开发模式启动"""
    print("🔧 开发模式启动")
    
    # 获取配置并选择应用类型
    config = get_config()
    
    if config.inference_mode == "load_balance":
        print("⚖️ 负载均衡模式")
        app = create_load_balanced_app(config)
    else:
        print("🎯 单实例模式")
        app = create_app(config)
    
    app_config = app.config['APP_CONFIG']
    
    print(f"📍 服务地址: http://{app_config.host}:{app_config.port}")
    print(f"🔧 推理引擎: {app_config.inference_engine}")
    print(f"📚 OpenAI 接口: http://{app_config.host}:{app_config.port}/v1/")
    print(f"🏥 健康检查: http://{app_config.host}:{app_config.port}/health")
    print(f"🛠️ 管理接口: http://{app_config.host}:{app_config.port}/v1/system/status")
    
    if config.inference_mode == "load_balance":
        print(f"⚖️ 负载均衡策略: {config.load_balance_strategy}")
        print(f"🔢 集群实例数: {config.cluster_instances}")
        print(f"📊 集群状态: http://{app_config.host}:{app_config.port}/v1/cluster/status")
    print("\n按 Ctrl+C 停止服务")
    
    # 注册信号处理
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        app.run(
            host=app_config.host,
            port=app_config.port,
            debug=app_config.debug,
            threaded=True,
            use_reloader=False  # 避免重复初始化
        )
    except KeyboardInterrupt:
        print("\n服务已停止")


def run_production():
    """生产模式启动"""
    print("🚀 生产模式启动")
    
    config = get_config()
    
    try:
        import gunicorn.app.wsgiapp as wsgi
    except ImportError:
        print("❌ 生产模式需要安装 gunicorn")
        print("请运行: pip install gunicorn")
        sys.exit(1)
    
    # 根据推理模式选择应用
    if config.inference_mode == "load_balance":
        print("⚖️ 负载均衡模式")
        app_module = 'src.api.load_balanced_app:get_load_balanced_app()'
        print(f"⚖️ 负载均衡策略: {config.load_balance_strategy}")
        print(f"🔢 集群实例数: {config.cluster_instances}")
    else:
        print("🎯 单实例模式")
        app_module = 'src.api.app:get_app()'
    
    # 设置 Gunicorn 参数
    sys.argv = [
        'gunicorn',
        '--bind', f'{config.host}:{config.port}',
        '--workers', str(config.workers),
        '--worker-class', 'gevent',
        '--worker-connections', '1000',
        '--timeout', '300',
        '--keepalive', '5',
        '--max-requests', '1000',
        '--max-requests-jitter', '100',
        '--preload',
        '--access-logfile', '-',
        '--error-logfile', '-',
        app_module
    ]
    
    print(f"📍 服务地址: http://{config.host}:{config.port}")
    print(f"👥 Worker进程数: {config.workers}")
    print(f"🔧 推理引擎: {config.inference_engine}")
    print("\n启动 Gunicorn 服务器...")
    
    # 启动 Gunicorn
    wsgi.run()


def check_dependencies():
    """检查依赖"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要 Python 3.8 或更高版本")
        sys.exit(1)
    
    print(f"✅ Python版本: {sys.version}")
    
    # 检查必需的包
    required_packages = [
        ('flask', 'flask'), 
        ('flask-cors', 'flask_cors'), 
        ('loguru', 'loguru'), 
        ('pydantic', 'pydantic'),
        ('python-dotenv', 'dotenv'), 
        ('psutil', 'psutil'), 
        ('requests', 'requests')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ 缺少必需的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ 基础依赖检查通过")
    
    # 检查推理引擎
    from src.utils.platform_detector import PlatformDetector
    
    detector = PlatformDetector()
    platform_info = detector.get_platform_info()
    optimal_engine = detector.detect_best_engine()
    
    print(f"✅ 检测到平台: {platform_info['system']} ({platform_info['machine']})")
    print(f"✅ 推荐推理引擎: {optimal_engine}")
    
    # 检查推理引擎可用性
    if optimal_engine == 'vllm':
        try:
            import vllm
            print(f"✅ VLLM 可用 (版本: {vllm.__version__})")
        except ImportError:
            print("⚠️ VLLM 不可用，将回退到 llama.cpp")
    
    elif optimal_engine == 'mlx':
        try:
            import mlx.core as mx
            import mlx_lm
            print(f"✅ MLX 可用")
        except ImportError:
            print("⚠️ MLX 不可用，将回退到 llama.cpp")
    
    try:
        import llama_cpp
        print(f"✅ llama.cpp 可用 (版本: {llama_cpp.__version__})")
    except ImportError:
        print("❌ llama.cpp 不可用")
        print("请安装: pip install llama-cpp-python")
        sys.exit(1)


def setup_environment():
    """设置环境"""
    print("⚙️ 设置运行环境...")
    
    # 创建必要的目录
    directories = ['logs', 'models', 'cache']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ 确保目录存在: {directory}/")
    
    # 检查配置文件
    if not Path('.env').exists():
        print("⚠️ 未找到 .env 配置文件")
        print("将使用默认配置，建议复制 .env.example 并根据需要修改")
    else:
        print("✅ 找到 .env 配置文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VLLM 跨平台推理服务')
    parser.add_argument(
        '--mode', 
        choices=['dev', 'prod'], 
        default='dev',
        help='运行模式 (dev: 开发模式, prod: 生产模式)'
    )
    parser.add_argument(
        '--skip-check', 
        action='store_true',
        help='跳过依赖检查'
    )
    parser.add_argument(
        '--config', 
        type=str,
        help='指定配置文件路径'
    )
    
    args = parser.parse_args()
    
    print("🚀 VLLM 跨平台推理服务")
    print("=" * 50)
    
    # 设置配置文件
    if args.config:
        os.environ['CONFIG_FILE'] = args.config
    
    # 环境检查
    if not args.skip_check:
        check_dependencies()
        setup_environment()
        print("=" * 50)
    
    # 启动服务
    try:
        if args.mode == 'dev':
            run_development()
        else:
            run_production()
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()