#!/bin/bash
# 快速开始脚本：HTTP 接口版 Marvain M6 工作流
#
# 使用方法:
#   chmod +x workflows/quickstart.sh
#   ./workflows/quickstart.sh

set -e  # 遇到错误立即退出

echo "=================================================="
echo "Marvain M6 HTTP 接口工作流 - 快速开始"
echo "=================================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 已安装"
        return 0
    else
        echo -e "${RED}✗${NC} $1 未安装"
        return 1
    fi
}

check_http_server() {
    if curl -s --max-time 2 $1 > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} HTTP 服务器响应正常: $1"
        return 0
    else
        echo -e "${RED}✗${NC} HTTP 服务器无响应: $1"
        return 1
    fi
}

# 1. 环境检查
echo "步骤 1/5: 环境检查"
echo "-------------------"

all_ok=true

# 检查 Python
if check_command python3; then
    python_version=$(python3 --version)
    echo "  版本: $python_version"
else
    all_ok=false
fi

# 检查 curl（用于测试 HTTP）
check_command curl || all_ok=false

# 检查 pip
check_command pip || check_command pip3 || all_ok=false

echo ""

# 2. HTTP 服务器检查
echo "步骤 2/5: HTTP 服务器检查"
echo "-------------------------"

HTTP_URL="http://192.168.10.123:8010"
if check_http_server "$HTTP_URL/observation"; then
    echo -e "  ${GREEN}服务器就绪！${NC}"
else
    echo -e "  ${YELLOW}警告: 服务器未响应${NC}"
    echo "  请确保 HTTP 服务器正在运行"
    all_ok=false
fi

echo ""

# 3. 目录结构检查
echo "步骤 3/5: 目录结构检查"
echo "---------------------"

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 不存在"
        return 1
    fi
}

check_file "src/lerobot/robots/marvain_m6_http/__init__.py" || all_ok=false
check_file "src/lerobot/robots/marvain_m6_http/marvain_m6_http.py" || all_ok=false
check_file "workflows/_config_loader.py" || all_ok=false
check_file "workflows/_robot_home.py" || all_ok=false
check_file "workflows/robot_interaction/deploy.py" || all_ok=false
check_file "workflows/robot_interaction/replay.py" || all_ok=false
check_file "workflows/robot_interaction/deploy_config.yaml" || all_ok=false
check_file "workflows/robot_interaction/replay_config.yaml" || all_ok=false

echo ""

# 4. 依赖检查
echo "步骤 4/5: Python 依赖检查"
echo "------------------------"

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

check_import() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 未安装"
        return 1
    fi
}

deps_ok=true
check_import "requests" || deps_ok=false
check_import "numpy" || deps_ok=false
check_import "cv2" || deps_ok=false
check_import "yaml" || deps_ok=false

if [ "$deps_ok" = false ]; then
    echo ""
    echo -e "${YELLOW}部分依赖未安装。请运行:${NC}"
    echo "  pip install requests numpy opencv-python pyyaml"
fi

echo ""

# 5. 总结
echo "步骤 5/5: 检查总结"
echo "-----------------"

if [ "$all_ok" = true ]; then
    echo -e "${GREEN}✅ 所有检查通过！环境已就绪。${NC}"
    echo ""
    echo "下一步操作:"
    echo ""
    echo "1. 测试 HTTP 连接:"
    echo "   python workflows/robot_interaction/test_http_robot.py"
    echo ""
    echo "2. 回放数据集 episode:"
    echo "   python workflows/robot_interaction/replay.py --episode 0 --fps 10"
    echo ""
    echo "3. 部署策略:"
    echo "   python workflows/robot_interaction/deploy.py --fps 15"
    echo ""
    echo "详细文档:"
    echo "  - workflows/README.md - 完整使用指南"
    echo "  - workflows/CHECKLIST.md - 使用检查清单"
    echo "  - workflows/SUMMARY.md - 实施总结"
    echo ""
else
    echo -e "${RED}❌ 部分检查未通过，请解决上述问题后再继续。${NC}"
    echo ""
    echo "常见问题:"
    echo "  1. HTTP 服务器未响应 → 启动服务器或检查网络连接"
    echo "  2. Python 依赖缺失 → pip install -e . 或安装单个包"
    echo "  3. 文件缺失 → 重新运行实施脚本"
    echo ""
    exit 1
fi

echo "=================================================="
