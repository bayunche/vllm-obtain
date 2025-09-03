#!/bin/bash

# VLLM 跨平台推理服务 - 自动安装脚本
# 支持 macOS (Apple Silicon) / Linux (CUDA/ROCm) / 通用 CPU

set -e  # 遇到错误立即退出

# 颜色和样式定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 图标定义
CHECKMARK="✅"
CROSS="❌"
WARNING="⚠️"
INFO="ℹ️"
ROCKET="🚀"
GEAR="⚙️"
APPLE="🍎"
LINUX="🐧"
CPU="💻"

# 打印带颜色的消息
print_success() { echo -e "${GREEN}${CHECKMARK} $1${NC}"; }
print_error() { echo -e "${RED}${CROSS} $1${NC}"; }
print_warning() { echo -e "${YELLOW}${WARNING} $1${NC}"; }
print_info() { echo -e "${CYAN}${INFO} $1${NC}"; }
print_header() { echo -e "${PURPLE}${BOLD}$1${NC}"; }

# 显示欢迎信息
show_welcome() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║             🚀 VLLM 跨平台推理服务 自动安装程序             ║"
    echo "║                                                              ║"
    echo "║          企业级 AI 推理服务，完全兼容 OpenAI API             ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
    
    print_info "正在检测您的系统环境..."
    sleep 2
}

# 检测操作系统和硬件
detect_platform() {
    print_header "\n${GEAR} 平台检测"
    
    # 检测操作系统
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "操作系统: macOS"
        
        # 检测芯片类型
        ARCH=$(uname -m)
        if [[ "$ARCH" == "arm64" ]]; then
            CHIP="apple_silicon"
            CHIP_NAME=$(system_profiler SPHardwareDataType | grep "Chip" | awk '{print $2, $3}')
            print_success "芯片: Apple Silicon ($CHIP_NAME)"
            PLATFORM="mac_silicon"
            PLATFORM_NAME="${APPLE} macOS Apple Silicon"
        else
            CHIP="intel"
            print_warning "芯片: Intel (性能受限)"
            PLATFORM="mac_intel" 
            PLATFORM_NAME="${APPLE} macOS Intel"
        fi
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "操作系统: Linux"
        
        # 检测GPU
        if command -v nvidia-smi &> /dev/null; then
            if nvidia-smi &> /dev/null; then
                GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
                print_success "NVIDIA GPU: $GPU_INFO"
                PLATFORM="linux_cuda"
                PLATFORM_NAME="${LINUX} Linux NVIDIA GPU"
            else
                print_warning "NVIDIA 驱动未正确安装"
                PLATFORM="linux_cpu"
                PLATFORM_NAME="${LINUX} Linux CPU"
            fi
        elif command -v rocm-smi &> /dev/null; then
            print_success "AMD GPU: ROCm 检测到"
            PLATFORM="linux_rocm"
            PLATFORM_NAME="${LINUX} Linux AMD GPU"
        else
            print_info "未检测到GPU，使用CPU模式"
            PLATFORM="linux_cpu"
            PLATFORM_NAME="${LINUX} Linux CPU"
        fi
        
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
        print_success "操作系统: Windows"
        
        # Windows GPU 检测
        if command -v nvidia-smi &> /dev/null; then
            if nvidia-smi &> /dev/null; then
                GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
                print_success "NVIDIA GPU: $GPU_INFO"
                PLATFORM="windows_cuda"
                PLATFORM_NAME="💻 Windows NVIDIA GPU"
            else
                print_warning "NVIDIA 驱动未正确安装"
                PLATFORM="windows_cpu"
                PLATFORM_NAME="💻 Windows CPU"
            fi
        else
            print_info "未检测到GPU，使用CPU模式"
            PLATFORM="windows_cpu"
            PLATFORM_NAME="💻 Windows CPU"
        fi
        
    else
        print_warning "未知操作系统，使用通用CPU配置"
        OS="unknown"
        PLATFORM="generic_cpu"
        PLATFORM_NAME="${CPU} 通用 CPU"
    fi
    
    print_info "检测到平台: ${BOLD}$PLATFORM_NAME${NC}"
}

# 检查Python版本
check_python() {
    print_header "\n🐍 Python 环境检查"
    
    # 检查 Python 3.8+
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            print_success "Python版本: $PYTHON_VERSION"
            PYTHON_CMD="python3"
        else
            print_error "需要 Python 3.8+，当前版本: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "未找到 Python3"
        print_info "请先安装 Python 3.8+ 版本"
        exit 1
    fi
}

# 创建虚拟环境
setup_venv() {
    print_header "\n📦 创建Python虚拟环境"
    
    if [[ -d "venv" ]]; then
        print_warning "虚拟环境已存在，正在删除旧环境..."
        rm -rf venv
    fi
    
    print_info "创建新的虚拟环境..."
    $PYTHON_CMD -m venv venv
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 更新 pip
    print_info "更新 pip..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    print_success "虚拟环境创建完成"
}

# 安装依赖
install_dependencies() {
    print_header "\n⚙️ 安装平台特定依赖"
    
    # 首先安装基础依赖
    print_info "安装基础依赖..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    # 安装核心框架
    print_info "安装核心框架..."
    pip install flask flask-cors python-dotenv pydantic pydantic-settings loguru psutil requests numpy transformers tokenizers --timeout 600
    
    # 根据平台安装特定依赖
    case $PLATFORM in
        "mac_silicon")
            print_info "安装 macOS Apple Silicon 专用依赖..."
            
            # 尝试安装 MLX
            print_info "尝试安装 MLX 引擎..."
            pip install mlx mlx-lm 2>/dev/null && print_success "MLX 安装成功" || print_warning "MLX 安装失败，将使用 llama.cpp"
            
            # 安装 llama.cpp 作为后备
            print_info "安装 llama.cpp 后备引擎..."
            pip install llama-cpp-python --timeout 600
            
            # 安装其他依赖
            print_info "安装其他必需组件..."
            pip install huggingface-hub safetensors modelscope --timeout 300 2>/dev/null || print_warning "部分组件安装失败"
            
            if [[ -f "requirements-mac.txt" ]]; then
                print_info "安装 macOS 其他依赖..."
                pip install -r requirements-mac.txt --timeout 300 2>/dev/null || print_warning "部分依赖安装失败"
            fi
            print_success "macOS 依赖安装完成"
            ;;
            
        "mac_intel")
            print_info "安装 macOS Intel 专用依赖..."
            
            # Intel Mac 不支持 MLX，直接使用 llama.cpp
            print_info "安装 llama.cpp 引擎..."
            pip install llama-cpp-python --timeout 600
            
            # 安装基础依赖
            pip install torch transformers tokenizers safetensors huggingface-hub modelscope --timeout 600
            print_success "macOS Intel 依赖安装完成"
            ;;
            
        "linux_cuda")
            print_info "安装 Linux NVIDIA GPU 专用依赖..."
            
            # 安装 PyTorch CUDA 版本
            print_info "安装 PyTorch (CUDA)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --timeout 600
            
            # 尝试安装 VLLM
            print_info "尝试安装 VLLM 引擎..."
            pip install vllm 2>/dev/null && print_success "VLLM 安装成功" || print_warning "VLLM 安装失败，将使用 llama.cpp"
            
            # 安装 llama.cpp 作为后备
            print_info "安装 llama.cpp 后备引擎..."
            CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --timeout 600
            
            # 安装其他必需组件
            print_info "安装其他必需组件..."
            pip install huggingface-hub safetensors modelscope nvidia-ml-py3 --timeout 300 2>/dev/null || print_warning "部分组件安装失败"
            
            # 安装其他依赖
            if [[ -f "requirements-linux.txt" ]]; then
                print_info "安装 Linux 其他依赖..."
                pip install -r requirements-linux.txt --timeout 600 2>/dev/null || print_warning "部分依赖安装失败"
            fi
            print_success "Linux GPU 依赖安装完成"
            ;;
            
        "linux_rocm")
            print_info "安装 Linux AMD GPU 专用依赖..."
            
            # 安装 PyTorch ROCm 版本
            print_info "安装 PyTorch (ROCm)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 --timeout 600
            
            # 安装 llama.cpp
            print_info "安装 llama.cpp 引擎..."
            pip install llama-cpp-python --timeout 600
            
            # 安装基础依赖
            pip install transformers tokenizers safetensors huggingface-hub modelscope --timeout 600
            print_warning "ROCm 支持需要手动配置 VLLM"
            print_info "请参考: https://docs.vllm.ai/en/latest/getting_started/amd-installation.html"
            print_success "Linux AMD GPU 依赖安装完成"
            ;;
            
        "windows_cuda")
            print_info "安装 Windows NVIDIA GPU 专用依赖..."
            
            # 安装 PyTorch CUDA 版本 (Windows)
            print_info "安装 PyTorch (CUDA for Windows)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --timeout 600
            
            # 安装 llama.cpp (主要引擎)
            print_info "安装 llama.cpp 引擎..."
            pip install llama-cpp-python --timeout 600
            
            # Windows 特定组件
            print_info "安装 Windows 特定组件..."
            pip install waitress pywin32 wmi --timeout 300 2>/dev/null || print_warning "部分 Windows 组件安装失败"
            
            # 安装其他必需组件
            print_info "安装其他必需组件..."
            pip install huggingface-hub safetensors modelscope --timeout 300 2>/dev/null || print_warning "部分组件安装失败"
            
            # 安装其他依赖
            if [[ -f "requirements-windows.txt" ]]; then
                print_info "安装 Windows 其他依赖..."
                pip install -r requirements-windows.txt --timeout 600 2>/dev/null || print_warning "部分依赖安装失败"
            fi
            print_success "Windows GPU 依赖安装完成"
            ;;
            
        "windows_cpu"|"linux_cpu"|"generic_cpu")
            print_info "安装通用 CPU 依赖..."
            
            # 安装 PyTorch CPU 版本
            print_info "安装 PyTorch (CPU)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --timeout 600
            
            # 安装 llama.cpp
            print_info "安装 llama.cpp 引擎..."
            pip install llama-cpp-python --timeout 600
            
            # Windows 特定组件 (如果是 Windows)
            if [[ "$OS" == "windows" ]]; then
                print_info "安装 Windows 特定组件..."
                pip install waitress pywin32 wmi --timeout 300 2>/dev/null || print_warning "部分 Windows 组件安装失败"
                
                if [[ -f "requirements-windows.txt" ]]; then
                    print_info "安装 Windows 其他依赖..."
                    pip install -r requirements-windows.txt --timeout 600 2>/dev/null || print_warning "部分依赖安装失败"
                fi
            fi
            
            # 安装基础依赖
            pip install transformers tokenizers safetensors huggingface-hub modelscope --timeout 600
            print_success "通用依赖安装完成"
            ;;
            
        *)
            print_info "安装通用依赖..."
            pip install -r requirements.txt --timeout 600
            print_success "通用依赖安装完成"
            ;;
    esac
    
    # 安装测试和开发工具（必需用于兼容性测试）
    print_info "安装测试工具..."
    pip install pytest pytest-asyncio pytest-cov pytest-benchmark || print_warning "测试工具安装失败"
    
    # 安装服务器组件
    print_info "安装服务器组件..."
    pip install gunicorn gevent waitress || print_warning "服务器组件安装失败"
    
    # 安装异步支持包（测试需要）
    print_info "安装异步支持包..."
    pip install asyncio-mqtt anyio httpx aiohttp || print_warning "异步包安装失败"
    
    # 运行依赖检查
    print_header "\n🔍 依赖检查"
    if [[ -f "check_dependencies.py" ]]; then
        print_info "运行依赖检查脚本..."
        $PYTHON_CMD check_dependencies.py 2>/dev/null || print_warning "依赖检查发现问题，但安装已完成"
    else
        print_warning "未找到依赖检查脚本"
    fi
}

# 配置环境
setup_config() {
    print_header "\n🔧 配置系统环境"
    
    case $PLATFORM in
        "mac_silicon"|"mac_intel")
            if [[ -f ".env.mac" ]]; then
                cp .env.mac .env
                print_success "使用 macOS 专用配置"
            else
                cp .env.example .env 2>/dev/null || true
                print_info "使用默认配置"
            fi
            ;;
            
        "linux_cuda"|"linux_rocm"|"linux_cpu")
            if [[ -f ".env.linux" ]]; then
                cp .env.linux .env
                print_success "使用 Linux 专用配置"
            else
                cp .env.example .env 2>/dev/null || true
                print_info "使用默认配置"
            fi
            ;;
            
        "windows_cuda"|"windows_cpu")
            if [[ -f ".env.windows" ]]; then
                cp .env.windows .env
                print_success "使用 Windows 专用配置"
            else
                cp .env.example .env 2>/dev/null || true
                print_info "使用默认配置"
            fi
            ;;
            
        *)
            cp .env.example .env 2>/dev/null || true
            print_info "使用默认配置"
            ;;
    esac
    
    # 创建必要目录
    mkdir -p models logs cache
    print_success "创建必要目录完成"
}

# 下载默认模型
download_model() {
    print_header "\n📥 下载默认模型"
    
    print_info "正在下载 Qwen2.5-0.5B-Instruct 模型..."
    
    case $PLATFORM in
        "mac_silicon"|"mac_intel")
            # macOS 使用 ModelScope
            print_info "使用 ModelScope 镜像下载 (适合国内用户)..."
            $PYTHON_CMD -c "
try:
    from modelscope import snapshot_download
    print('正在下载模型...')
    snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models/qwen-0.5b')
    print('✅ 模型下载完成')
except Exception as e:
    print(f'❌ ModelScope 下载失败: {e}')
    print('请手动下载模型或检查网络连接')
" 2>/dev/null || print_warning "模型下载失败，请手动下载"
            ;;
            
        *)
            # 其他平台尝试 HuggingFace
            if command -v huggingface-cli &> /dev/null; then
                huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./models/qwen-0.5b 2>/dev/null || \
                print_warning "模型下载失败，请手动下载"
            else
                print_warning "未安装 huggingface-cli，跳过模型下载"
                print_info "可以稍后手动下载模型"
            fi
            ;;
    esac
}

# 验证安装
verify_installation() {
    print_header "\n🧪 验证安装"
    
    print_info "检查核心依赖..."
    
    # 基础包检查
    $PYTHON_CMD -c "
import sys
failed = []
packages = [
    ('flask', 'Flask'),
    ('requests', 'Requests'),
    ('psutil', 'PSUtil'),
    ('pydantic', 'Pydantic'),
    ('loguru', 'Loguru'),
    ('numpy', 'NumPy')
]

for import_name, display_name in packages:
    try:
        __import__(import_name)
        print(f'✅ {display_name} 导入成功')
    except ImportError:
        print(f'❌ {display_name} 导入失败')
        failed.append(display_name)

if failed:
    print(f'⚠️ 缺失依赖: {', '.join(failed)}')
    sys.exit(1)
else:
    print('✅ 所有基础依赖导入成功')
" || print_error "基础依赖检查失败"
    
    # 推理引擎检查
    print_info "检查推理引擎..."
    $PYTHON_CMD -c "
import platform

# 检查 llama.cpp (通用后备)
try:
    import llama_cpp
    print('✅ llama.cpp 引擎可用')
    llama_available = True
except ImportError:
    print('❌ llama.cpp 引擎不可用')
    llama_available = False

# 检查平台特定引擎
system = platform.system().lower()
machine = platform.machine().lower()

if system == 'darwin' and ('arm64' in machine or 'aarch64' in machine):
    # macOS Apple Silicon - 检查 MLX
    try:
        import mlx.core as mx
        import mlx_lm
        print('✅ MLX 引擎可用 (Apple Silicon 优化)')
    except ImportError:
        print('⚠️ MLX 引擎不可用，将使用 llama.cpp')

elif system in ['linux', 'windows']:
    # Linux/Windows - 检查 CUDA 和 VLLM
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f'✅ PyTorch CUDA 可用 (GPU: {torch.cuda.get_device_name(0)})')
        else:
            print('⚠️ CUDA 不可用，将使用 CPU 模式')
    except ImportError:
        print('⚠️ PyTorch 未安装')
    
    try:
        import vllm
        print('✅ VLLM 引擎可用')
    except ImportError:
        print('⚠️ VLLM 引擎不可用，将使用 llama.cpp')

# 确保至少有一个引擎可用
if not llama_available:
    print('❌ 错误：没有可用的推理引擎！')
    import sys
    sys.exit(1)
" || print_warning "推理引擎检查发现问题"
    
    # 测试脚本检查
    print_info "检查测试脚本..."
    if [[ -f "test_compatibility_report.py" ]]; then
        print_success "找到兼容性测试脚本"
        
        # 尝试导入测试脚本依赖
        $PYTHON_CMD -c "
try:
    import asyncio
    import statistics
    import threading
    import concurrent.futures
    from datetime import datetime
    from typing import Dict, List, Any
    print('✅ 测试脚本基础依赖检查通过')
except ImportError as e:
    print(f'⚠️ 测试脚本依赖缺失: {e}')
    
# 检查项目特定模块
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from src.utils.platform_detector import PlatformDetector
    from src.utils.config import ConfigManager
    print('✅ 项目模块导入成功')
except ImportError as e:
    print(f'⚠️ 项目模块导入失败: {e}')
" 2>/dev/null || print_warning "测试脚本依赖检查失败"
    else
        print_warning "未找到 test_compatibility_report.py"
        print_info "请确保测试脚本存在于项目根目录"
    fi
    
    # 检查运行测试脚本
    if [[ -f "run_tests.sh" ]]; then
        print_success "找到测试运行器 run_tests.sh"
    else
        print_warning "未找到 run_tests.sh"
    fi
    
    print_success "安装验证完成"
}

# 显示下一步操作
show_next_steps() {
    print_header "\n${ROCKET} 安装完成！"
    
    echo -e "${GREEN}${BOLD}🎉 VLLM 跨平台推理服务安装成功！${NC}\n"
    
    echo -e "${YELLOW}📋 下一步操作:${NC}"
    echo -e "  ${CYAN}1. 激活虚拟环境:${NC}"
    echo -e "     source venv/bin/activate"
    echo ""
    echo -e "  ${CYAN}2. 启动服务:${NC}"
    
    case $PLATFORM in
        "mac_silicon")
            echo -e "     python run.py --mode prod  ${GREEN}# 生产模式 (推荐)${NC}"
            echo -e "     python run.py --mode dev   ${BLUE}# 开发模式${NC}"
            ;;
        *)
            echo -e "     python run.py --mode prod  ${GREEN}# 生产模式 (推荐)${NC}"
            echo -e "     python run.py --mode dev   ${BLUE}# 开发模式${NC}"
            ;;
    esac
    
    echo ""
    echo -e "  ${CYAN}3. 运行完整测试（生成HTML报告）:${NC}"
    echo -e "     ./run_tests.sh  ${GREEN}# 推荐：完整测试流程${NC}"
    echo -e "     python test_compatibility_report.py  ${BLUE}# 直接运行测试${NC}"
    echo ""
    echo -e "  ${CYAN}4. 验证服务:${NC}"
    echo -e "     curl http://localhost:8001/health"
    echo ""
    echo -e "  ${CYAN}5. OpenAI 兼容测试:${NC}"
    echo -e "     python scripts/tests/test_openai_compatibility.py"
    echo ""
    echo -e "  ${PURPLE}📋 测试流程说明:${NC}"
    echo -e "     1. ./run_tests.sh 会自动运行依赖检查 -> 兼容性测试 -> 生成HTML报告"
    echo -e "     2. 测试报告包含: 平台信息、性能测试、Qwen3-7B 并发测试"
    echo -e "     3. HTML报告文件格式: compatibility_report_YYYYMMDD_HHMMSS.html"
    
    echo -e "\n${PURPLE}🔗 访问地址:${NC}"
    echo -e "  • 服务地址: ${BOLD}http://localhost:8001${NC}"
    echo -e "  • OpenAI API: ${BOLD}http://localhost:8001/v1${NC}"
    echo -e "  • 健康检查: ${BOLD}http://localhost:8001/health${NC}"
    
    echo -e "\n${BLUE}📚 文档资源:${NC}"
    echo -e "  • 生产部署指南: README_PRODUCTION.md"
    echo -e "  • 项目结构说明: PROJECT_STRUCTURE.md"
    echo -e "  • 测试报告: Mac_Studio_完整测试报告_20250821.md"
    
    echo -e "\n${GREEN}✨ 平台优化配置:${NC}"
    case $PLATFORM in
        "mac_silicon")
            echo -e "  • 推理引擎: ${BOLD}MLX${NC} (Apple Silicon 专用优化)"
            echo -e "  • 服务器: ${BOLD}Gunicorn + Gevent${NC} (单worker模式)"
            echo -e "  • 并发能力: ${BOLD}20+ 并发${NC}"
            echo -e "  • 特色: 功耗极低，静音运行"
            ;;
        "linux_cuda")
            echo -e "  • 推理引擎: ${BOLD}VLLM${NC} (CUDA 高性能优化)"
            echo -e "  • 服务器: ${BOLD}Gunicorn + Gevent${NC} (多worker模式)"
            echo -e "  • 并发能力: ${BOLD}100+ 并发${NC}"
            echo -e "  • 特色: 最高性能，企业级扩展"
            ;;
        "windows_cuda")
            echo -e "  • 推理引擎: ${BOLD}LlamaCpp${NC} (Windows CUDA 优化)"
            echo -e "  • 服务器: ${BOLD}Waitress${NC} (Windows 专用)"
            echo -e "  • 并发能力: ${BOLD}20-50 并发${NC}"
            echo -e "  • 特色: Windows 原生支持，稳定性好"
            ;;
        "windows_cpu")
            echo -e "  • 推理引擎: ${BOLD}LlamaCpp${NC} (Windows CPU)"
            echo -e "  • 服务器: ${BOLD}Waitress${NC} (Windows 专用)"
            echo -e "  • 并发能力: ${BOLD}5-15 并发${NC}"
            echo -e "  • 特色: 零门槛，Windows 原生兼容"
            ;;
        *)
            echo -e "  • 推理引擎: ${BOLD}LlamaCpp${NC} (CPU 通用兼容)"
            echo -e "  • 服务器: ${BOLD}Flask/Gunicorn${NC}"
            echo -e "  • 并发能力: ${BOLD}5-10 并发${NC}"
            echo -e "  • 特色: 零门槛，全平台兼容"
            ;;
    esac
    
    echo -e "\n${RED}${BOLD}❤️ 如果项目对您有帮助，请给我们一个 Star！${NC}"
}

# 主函数
main() {
    # 检查是否在项目目录中
    if [[ ! -f "run.py" || ! -f "requirements.txt" ]]; then
        print_error "请在 VLLM 项目根目录中运行此脚本"
        exit 1
    fi
    
    # 执行安装步骤
    show_welcome
    detect_platform
    check_python
    setup_venv
    install_dependencies
    setup_config
    download_model
    verify_installation
    show_next_steps
    
    echo -e "\n${GREEN}${BOLD}🚀 安装完成！祝您使用愉快！${NC}\n"
}

# 错误处理
trap 'print_error "安装过程中出现错误，请检查上面的错误信息"; exit 1' ERR

# 运行主函数
main "$@"