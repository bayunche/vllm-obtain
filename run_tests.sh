#!/bin/bash

# VLLM 跨平台推理服务 - 测试运行器
# 在安装完成后运行此脚本以执行完整的测试并生成报告

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

# 图标定义
CHECKMARK="✅"
CROSS="❌"
INFO="ℹ️"
ROCKET="🚀"

# 打印函数
print_success() { echo -e "${GREEN}${CHECKMARK} $1${NC}"; }
print_error() { echo -e "${RED}${CROSS} $1${NC}"; }
print_info() { echo -e "${CYAN}${INFO} $1${NC}"; }
print_header() { echo -e "${CYAN}${BOLD}$1${NC}"; }

# 检查虚拟环境
check_venv() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "虚拟环境已激活"
        PYTHON_CMD="$VIRTUAL_ENV/bin/python"
    elif [[ -d "venv" ]]; then
        print_info "激活虚拟环境..."
        source venv/bin/activate
        PYTHON_CMD="./venv/bin/python"
        print_success "虚拟环境已激活"
    else
        print_info "使用系统 Python"
        PYTHON_CMD="python3"
    fi
}

# 运行依赖检查
run_dependency_check() {
    print_header "\n${INFO} 依赖检查"
    
    if [[ -f "check_dependencies.py" ]]; then
        print_info "运行依赖检查..."
        $PYTHON_CMD check_dependencies.py
        
        if [ $? -eq 0 ]; then
            print_success "依赖检查通过"
        else
            print_error "依赖检查失败"
            echo -e "${YELLOW}建议运行 ./install.sh 重新安装依赖${NC}"
            exit 1
        fi
    else
        print_error "未找到 check_dependencies.py"
    fi
}

# 运行兼容性测试
run_compatibility_test() {
    print_header "\n${ROCKET} 运行兼容性和性能测试"
    
    if [[ -f "test_compatibility_report.py" ]]; then
        print_info "开始测试（包含性能测试，可能需要几分钟）..."
        
        # 运行测试
        $PYTHON_CMD test_compatibility_report.py
        
        if [ $? -eq 0 ]; then
            print_success "测试完成！"
            
            # 查找生成的报告文件
            REPORT_FILE=$(ls -t compatibility_report_*.html 2>/dev/null | head -n1)
            
            if [[ -n "$REPORT_FILE" ]]; then
                print_success "测试报告已生成: $REPORT_FILE"
                
                # 尝试在浏览器中打开
                if command -v xdg-open &> /dev/null; then
                    xdg-open "$REPORT_FILE" 2>/dev/null &
                    print_info "正在浏览器中打开报告..."
                elif command -v open &> /dev/null; then
                    open "$REPORT_FILE" 2>/dev/null &
                    print_info "正在浏览器中打开报告..."
                else
                    print_info "请手动打开报告文件: $REPORT_FILE"
                fi
            else
                print_error "未找到生成的报告文件"
            fi
        else
            print_error "测试运行失败"
            exit 1
        fi
    else
        print_error "未找到 test_compatibility_report.py"
        exit 1
    fi
}

# 运行其他测试（可选）
run_additional_tests() {
    print_header "\n${INFO} 其他测试（可选）"
    
    echo -e "${CYAN}是否运行额外的测试？(y/n)${NC}"
    read -p "> " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 平台检测测试
        if [[ -f "scripts/tests/test_platform_detection.py" ]]; then
            print_info "运行平台检测测试..."
            $PYTHON_CMD scripts/tests/test_platform_detection.py
        fi
        
        # OpenAI 兼容性测试（需要服务运行）
        if [[ -f "scripts/tests/test_openai_compatibility.py" ]]; then
            echo -e "${CYAN}运行 OpenAI 兼容性测试需要先启动服务，是否继续？(y/n)${NC}"
            read -p "> " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "请在另一个终端中运行: python run.py"
                print_info "等待服务启动后按任意键继续..."
                read -n 1 -s -r
                
                print_info "运行 OpenAI 兼容性测试..."
                $PYTHON_CMD scripts/tests/test_openai_compatibility.py
            fi
        fi
    fi
}

# 显示总结
show_summary() {
    print_header "\n📊 测试总结"
    echo "=" * 50
    
    if [[ -n "$REPORT_FILE" ]]; then
        print_success "HTML 测试报告: $REPORT_FILE"
    fi
    
    echo -e "\n${GREEN}${BOLD}✨ 所有测试完成！${NC}"
    echo -e "\n${CYAN}下一步操作：${NC}"
    echo -e "1. 查看测试报告了解系统兼容性和性能"
    echo -e "2. 运行 ${BOLD}python run.py${NC} 启动服务"
    echo -e "3. 访问 ${BOLD}http://localhost:8001/health${NC} 检查服务状态"
}

# 主函数
main() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║             🧪 VLLM 跨平台推理服务 测试运行器               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
    
    # 检查并激活虚拟环境
    check_venv
    
    # 运行依赖检查
    run_dependency_check
    
    # 运行兼容性测试
    run_compatibility_test
    
    # 可选的额外测试
    run_additional_tests
    
    # 显示总结
    show_summary
}

# 错误处理
trap 'print_error "测试过程中出现错误"; exit 1' ERR

# 运行主函数
main "$@"