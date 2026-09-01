#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 第一次运行 test1.py，成功后运行第二次
echo "=== 运行测试 ==="
bash ./test2.sh && \
bash ./test2.sh && \
echo "所有测试完成！"
