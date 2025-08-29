#!/usr/bin/env python3
"""
依赖检查脚本
检查所有必需的依赖是否正确安装
"""

import sys
import platform
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


class DependencyChecker:
    """依赖检查器"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.python_version = platform.python_version()
        self.results = {
            'core': [],
            'platform_specific': [],
            'optional': [],
            'missing': [],
            'errors': []
        }
    
    def check_package(self, package_name: str, import_name: str = None, version_check: bool = True) -> Dict:
        """检查单个包"""
        if import_name is None:
            import_name = package_name.replace('-', '_')
            
        result = {
            'name': package_name,
            'import_name': import_name,
            'installed': False,
            'version': None,
            'error': None
        }
        
        try:
            # 尝试导入包
            module = importlib.import_module(import_name)
            result['installed'] = True
            
            # 获取版本信息
            if version_check:
                if hasattr(module, '__version__'):
                    result['version'] = module.__version__
                elif hasattr(module, 'VERSION'):
                    result['version'] = module.VERSION
                elif hasattr(module, 'version'):
                    result['version'] = module.version
                else:
                    # 尝试从 pip 获取版本
                    try:
                        import pkg_resources
                        result['version'] = pkg_resources.get_distribution(package_name).version
                    except:
                        result['version'] = 'unknown'
                        
        except ImportError as e:
            result['error'] = str(e)
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            
        return result
    
    def check_core_dependencies(self):
        """检查核心依赖"""
        core_packages = [
            ('flask', 'flask'),
            ('flask-cors', 'flask_cors'),
            ('python-dotenv', 'dotenv'),
            ('pydantic', 'pydantic'),
            ('pydantic-settings', 'pydantic_settings'),
            ('loguru', 'loguru'),
            ('psutil', 'psutil'),
            ('requests', 'requests'),
            ('numpy', 'numpy'),
            ('transformers', 'transformers'),
            ('torch', 'torch')
        ]
        
        print("🔍 检查核心依赖...")
        for package_name, import_name in core_packages:
            result = self.check_package(package_name, import_name)
            self.results['core'].append(result)
            
            status = "✅" if result['installed'] else "❌"
            version = f" (v{result['version']})" if result['version'] else ""
            print(f"  {status} {package_name}{version}")
            
            if not result['installed']:
                self.results['missing'].append(package_name)
    
    def check_platform_specific_dependencies(self):
        """检查平台特定依赖"""
        print(f"\n🔍 检查 {self.system} 平台特定依赖...")
        
        if self.system == 'darwin' and 'arm64' in self.machine:
            # macOS Apple Silicon
            platform_packages = [
                ('mlx', 'mlx'),
                ('mlx-lm', 'mlx_lm'),
            ]
        elif self.system == 'linux':
            # Linux
            platform_packages = [
                ('vllm', 'vllm'),
                ('nvidia-ml-py3', 'pynvml'),
            ]
        elif self.system == 'windows':
            # Windows
            platform_packages = [
                ('waitress', 'waitress'),
                ('pywin32', 'win32api'),
            ]
        else:
            platform_packages = []
        
        # 通用后备引擎
        platform_packages.append(('llama-cpp-python', 'llama_cpp'))
        
        for package_name, import_name in platform_packages:
            result = self.check_package(package_name, import_name)
            self.results['platform_specific'].append(result)
            
            status = "✅" if result['installed'] else "❌"
            version = f" (v{result['version']})" if result['version'] else ""
            print(f"  {status} {package_name}{version}")
            
            if not result['installed']:
                self.results['missing'].append(package_name)
    
    def check_optional_dependencies(self):
        """检查可选依赖"""
        optional_packages = [
            ('pytest', 'pytest'),
            ('pytest-asyncio', 'pytest_asyncio'),
            ('pytest-cov', 'pytest_cov'),
            ('modelscope', 'modelscope'),
            ('huggingface-hub', 'huggingface_hub'),
            ('safetensors', 'safetensors'),
            ('gunicorn', 'gunicorn'),
            ('gevent', 'gevent'),
        ]
        
        print("\n🔍 检查可选依赖...")
        for package_name, import_name in optional_packages:
            result = self.check_package(package_name, import_name)
            self.results['optional'].append(result)
            
            status = "✅" if result['installed'] else "⚠️"
            version = f" (v{result['version']})" if result['version'] else ""
            print(f"  {status} {package_name}{version}")
    
    def check_system_requirements(self):
        """检查系统需求"""
        print("\n🔍 检查系统环境...")
        
        # Python 版本检查
        python_major, python_minor = sys.version_info[:2]
        if python_major >= 3 and python_minor >= 8:
            print(f"  ✅ Python版本: {self.python_version}")
        else:
            print(f"  ❌ Python版本: {self.python_version} (需要 >= 3.8)")
            self.results['errors'].append("Python版本过低")
        
        # 系统信息
        print(f"  ✅ 操作系统: {platform.system()} {platform.release()}")
        print(f"  ✅ 架构: {platform.machine()}")
        
        # GPU 检查
        self.check_gpu_availability()
    
    def check_gpu_availability(self):
        """检查 GPU 可用性"""
        # NVIDIA GPU 检查
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"  ✅ NVIDIA GPU: {gpu_name} ({gpu_count} 个设备)")
            else:
                print(f"  ⚠️ NVIDIA GPU: 不可用或未安装CUDA")
        except ImportError:
            print(f"  ⚠️ PyTorch: 未安装，无法检查GPU")
        
        # Apple Metal 检查 (macOS)
        if self.system == 'darwin':
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    print(f"  ✅ Apple Metal (MPS): 可用")
                else:
                    print(f"  ⚠️ Apple Metal (MPS): 不可用")
            except:
                pass
    
    def generate_install_suggestions(self):
        """生成安装建议"""
        if not self.results['missing']:
            return
        
        print("\n💡 安装建议:")
        print("=" * 50)
        
        # 基础安装命令
        if self.system == 'darwin':
            requirements_file = 'requirements-mac.txt'
        elif self.system == 'linux':
            requirements_file = 'requirements-linux.txt'
        elif self.system == 'windows':
            requirements_file = 'requirements-windows.txt'
        else:
            requirements_file = 'requirements.txt'
        
        print(f"推荐使用平台特定的requirements文件:")
        print(f"pip install -r {requirements_file}")
        print()
        
        # 分别安装缺失的包
        if len(self.results['missing']) <= 5:
            print("或者单独安装缺失的包:")
            for package in self.results['missing']:
                print(f"pip install {package}")
        
        # 平台特定建议
        if self.system == 'windows':
            print("\nWindows 特定建议:")
            print("1. 安装 Visual Studio Build Tools")
            print("2. 考虑使用 Anaconda 管理环境")
            print("3. PyTorch CUDA版本: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        elif self.system == 'darwin' and 'arm64' in self.machine:
            print("\nmacOS Apple Silicon 特定建议:")
            print("1. 确保使用 Python 3.8+ for Apple Silicon")
            print("2. 某些包可能需要从源码编译")
            print("3. MLX仅支持 Apple Silicon")
        elif self.system == 'linux':
            print("\nLinux 特定建议:")
            print("1. 安装系统依赖: sudo apt install python3-dev build-essential")
            print("2. NVIDIA GPU: 安装 CUDA Toolkit")
            print("3. 考虑使用虚拟环境隔离依赖")
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "=" * 60)
        print("📊 依赖检查总结")
        print("=" * 60)
        
        total_core = len(self.results['core'])
        installed_core = sum(1 for pkg in self.results['core'] if pkg['installed'])
        
        total_platform = len(self.results['platform_specific'])
        installed_platform = sum(1 for pkg in self.results['platform_specific'] if pkg['installed'])
        
        total_optional = len(self.results['optional'])
        installed_optional = sum(1 for pkg in self.results['optional'] if pkg['installed'])
        
        print(f"核心依赖: {installed_core}/{total_core} 已安装")
        print(f"平台特定: {installed_platform}/{total_platform} 已安装")
        print(f"可选依赖: {installed_optional}/{total_optional} 已安装")
        
        if self.results['missing']:
            print(f"\n❌ 缺失 {len(self.results['missing'])} 个依赖包")
            print("建议运行安装命令来解决缺失依赖")
        else:
            print("\n✅ 所有必需依赖都已安装！")
        
        if self.results['errors']:
            print(f"\n⚠️ 发现 {len(self.results['errors'])} 个问题:")
            for error in self.results['errors']:
                print(f"  - {error}")
    
    def run_check(self):
        """运行完整检查"""
        print("🚀 开始依赖检查...")
        print(f"系统: {self.system} | 架构: {self.machine} | Python: {self.python_version}")
        print("=" * 60)
        
        self.check_system_requirements()
        self.check_core_dependencies()
        self.check_platform_specific_dependencies()
        self.check_optional_dependencies()
        
        self.print_summary()
        self.generate_install_suggestions()
        
        return len(self.results['missing']) == 0 and len(self.results['errors']) == 0


def main():
    """主函数"""
    checker = DependencyChecker()
    success = checker.run_check()
    
    if success:
        print("\n🎉 环境检查完成，可以开始使用服务！")
        return 0
    else:
        print("\n⚠️ 请先解决上述问题再启动服务")
        return 1


if __name__ == "__main__":
    sys.exit(main())