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
    pip install --upgrade pip > /dev/null 2>&1
    
    print_success "虚拟环境创建完成"
}

# 安装依赖
install_dependencies() {
    print_header "\n⚙️ 安装平台特定依赖"
    
    case $PLATFORM in
        "mac_silicon")
            print_info "安装 macOS Apple Silicon 专用依赖..."
            if [[ -f "requirements-mac.txt" ]]; then
                pip install -r requirements-mac.txt
                print_success "macOS 依赖安装完成"
            else
                print_warning "未找到 requirements-mac.txt，使用通用依赖"
                pip install -r requirements.txt
            fi
            ;;
            
        "linux_cuda")
            print_info "安装 Linux NVIDIA GPU 专用依赖..."
            if [[ -f "requirements-linux.txt" ]]; then
                pip install -r requirements-linux.txt
                print_success "Linux GPU 依赖安装完成"
            else
                pip install -r requirements.txt
                print_success "通用依赖安装完成"
            fi
            ;;
            
        "linux_rocm")
            print_info "安装 Linux AMD GPU 专用依赖..."
            pip install -r requirements.txt
            print_warning "ROCm 支持需要手动配置"
            ;;
            
        *)
            print_info "安装通用CPU依赖..."
            pip install -r requirements.txt
            print_success "通用依赖安装完成"
            ;;
    esac
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
    
    print_info "检查 Python 包导入..."
    
    # 基础包检查
    $PYTHON_CMD -c "
import flask, requests, psutil
print('✅ 基础依赖导入成功')
" 2>/dev/null || print_error "基础依赖导入失败"
    
    # 平台特定包检查
    case $PLATFORM in
        "mac_silicon"|"mac_intel")
            $PYTHON_CMD -c "
try:
    import mlx.core as mx
    print('✅ MLX 引擎可用')
except ImportError:
    print('⚠️ MLX 引擎不可用，将使用 CPU 模式')
" 2>/dev/null
            ;;
            
        "linux_cuda")
            $PYTHON_CMD -c "
try:
    import torch
    print(f'✅ PyTorch 可用，CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU 数量: {torch.cuda.device_count()}')
except ImportError:
    print('⚠️ PyTorch 不可用')
" 2>/dev/null
            ;;
    esac
    
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
    echo -e "  ${CYAN}3. 验证服务:${NC}"
    echo -e "     curl http://localhost:8001/health"
    echo ""
    echo -e "  ${CYAN}4. OpenAI 兼容测试:${NC}"
    echo -e "     python test_openai_compatibility.py"
    
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