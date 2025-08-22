#!/bin/bash

# 第一次运行 test1.py，成功后运行第二次
echo "=== 运行测试 ==="
bash /home/Better-oft/RoboTwin/test2.sh && \
bash /home/Better-oft/RoboTwin/test2.sh && \
echo "所有测试完成！"