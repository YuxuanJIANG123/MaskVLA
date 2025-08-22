#!/usr/bin/env python3
"""
基于AIRBOT的RDT控制客户端 - 运行在笔记本B
使用实际的AIRBOT机器人与RDT推理服务器通信
"""

import time
import json
import base64
import numpy as np
import requests
from collections import deque
from PIL import Image
import io
import cv2

from airbot_robot import AIRBOTPlay

class AIRBOTRDTController:
    def __init__(self, server_url, max_steps=1000, chunk_size=64, control_freq=25):
        self.server_url = server_url
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.control_freq = control_freq
        self.action_buffer = None
        self.next_action_buffer = None  # 新增用于提前加载下一块动作
        self.prefetch_steps = 5  # 提前5步开始请求下一动作块
        # 初始化AIRBOT机器人
        print("初始化AIRBOT机器人...")
        self.robot = AIRBOTPlay()
        
        # 观测历史窗口 (存储两个时刻的观测)
        self.observation_window = deque(maxlen=2)
        
        # 动作缓冲区
        self.action_buffer = None
        
        print("AIRBOT RDT控制器初始化完成")
        
    def image_to_base64(self, image):
        """将numpy数组或PIL Image转换为base64字符串"""
        if isinstance(image, np.ndarray):
            # numpy数组转PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            # 确保是RGB格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            else:
                print(f"警告: 图像形状异常 {image.shape}")
                return None
        
        # PIL Image转base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def capture_observation(self):
        """捕获当前观测"""
        obs_data = self.robot.capture_observation()
        
        # 提取关节位置 (AIRBOT: 每个机械臂6关节+1夹爪 = 14维总共)
        state = obs_data['observation.state']
        # state格式: [arm0_joints(6) + arm0_gripper(1) + arm1_joints(6) + arm1_gripper(1)]
        qpos = state  # 直接使用，应该已经是14维
        
        if len(qpos) != 14:
            print(f"警告: qpos维度不正确，期望14维，实际{len(qpos)}维")
            # 如果维度不对，尝试填充或截断到14维
            if len(qpos) < 14:
                qpos = qpos + [0.0] * (14 - len(qpos))
            else:
                qpos = qpos[:14]
        
        # 提取图像并进行JPEG压缩对齐训练
        images = {}
        camera_mapping = {
            'cam_high': 'front',
            'cam_right_wrist': 'right', 
            'cam_left_wrist': 'left'
        }
        
        for cam_name, img_key in camera_mapping.items():
            img = obs_data[f'observation.images.{cam_name}']
            
            # JPEG压缩对齐训练过程
            if isinstance(img, np.ndarray):
                # 确保是BGR格式用于OpenCV
                if img.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.max() <= 1 else img
                    img_encoded = cv2.imencode('.jpg', img_bgr)[1].tobytes()
                    img_decoded = cv2.imdecode(np.frombuffer(img_encoded, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)  # 转回RGB
                else:
                    print(f"警告: {cam_name} 图像通道数异常: {img.shape}")
            
            images[img_key] = img
        
        return {
            'qpos': qpos,
            'images': images,
            'timestamp': time.time()
        }
    
    def update_observation_window(self):
        """更新观测窗口"""
        obs = self.capture_observation()
        self.observation_window.append(obs)
        
        # 如果只有一个观测，复制一份作为历史
        if len(self.observation_window) == 1:
            self.observation_window.append(obs.copy())
            
    def prepare_inference_data(self, use_relative=False):
        """准备发送给推理服务器的数据"""
        if len(self.observation_window) < 2:
            raise ValueError("观测窗口数据不足")
        
        prev_obs = self.observation_window[-2]
        curr_obs = self.observation_window[-1]

        # 计算 qpos（相对或绝对）
        if use_relative:
            qpos = (np.array(curr_obs['qpos']) - np.array(prev_obs['qpos'])).tolist()
        else:
            qpos = curr_obs['qpos']

        # 准备图像数据 (base64编码)
        images_b64 = {}
        for img_key in ['front', 'right', 'left']:
            # t-1 时刻的图像
            if prev_obs['images'].get(img_key) is not None:
                img_b64 = self.image_to_base64(prev_obs['images'][img_key])
                if img_b64:
                    images_b64[f'{img_key}_t_minus_1'] = img_b64

            # t 时刻的图像
            if curr_obs['images'].get(img_key) is not None:
                img_b64 = self.image_to_base64(curr_obs['images'][img_key])
                if img_b64:
                    images_b64[f'{img_key}_t'] = img_b64

        return {
            'qpos': qpos,
            'images': images_b64,
            'timestamp': curr_obs['timestamp']
        }

    # def prepare_inference_data(self):
        """准备发送给推理服务器的数据"""
        if len(self.observation_window) < 2:
            raise ValueError("观测窗口数据不足")
        
        prev_obs = self.observation_window[-2]
        curr_obs = self.observation_window[-1]
        
        # 准备图像数据 (base64编码)
        images_b64 = {}
        for img_key in ['front', 'right', 'left']:
            # t-1时刻的图像
            if prev_obs['images'][img_key] is not None:
                img_b64 = self.image_to_base64(prev_obs['images'][img_key])
                if img_b64:
                    images_b64[f'{img_key}_t_minus_1'] = img_b64
            
            # t时刻的图像  
            if curr_obs['images'][img_key] is not None:
                img_b64 = self.image_to_base64(curr_obs['images'][img_key])
                if img_b64:
                    images_b64[f'{img_key}_t'] = img_b64
        
        return {
            'qpos': curr_obs['qpos'],
            'images': images_b64,
            'timestamp': curr_obs['timestamp']
        }
    
    def request_inference(self, observation_data):
        """向推理服务器请求动作预测"""
        try:
            response = requests.post(
                f"{self.server_url}/act",
                json=observation_data,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    return np.array(result['actions'])
                else:
                    print(f"推理失败: {result['error']}")
                    return None
            else:
                print(f"HTTP请求失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"推理请求异常: {e}")
            return None
        
    def get_current_qpos_from_obs(self):
        """从最新观测窗口中获取当前 qpos"""
        if len(self.observation_window) == 0:
            raise RuntimeError("观测窗口为空，无法获取当前关节角")
        curr_obs = self.observation_window[-1]
        return np.array(curr_obs['qpos'])
    
    def execute_action(self, action, use_relative=False):
        """执行动作：支持绝对或相对关节控制"""
        try:
            if use_relative:
                # 使用相对增量控制
                current_qpos = self.get_current_qpos_from_obs()
                target_qpos = current_qpos + np.array(action)
                self.robot.send_action(target_qpos.tolist())
            else:
                # 使用绝对位置控制
                self.robot.send_action(action.tolist())
            return True
        except Exception as e:
            print(f"动作执行失败: {e}")
            return False

    # def execute_action(self, action):
        """执行动作"""
        try:
            # action是14维: [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
            # AIRBOT格式也是这样，直接发送
            self.robot.send_action(action.tolist())
            return True
        except Exception as e:
            print(f"动作执行失败: {e}")
            return False
    
    def test_connection(self):
        """测试与服务器的连接"""
        print("=== 测试服务器连接 ===")
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 服务器连接正常: {result}")
                return True
            else:
                print(f"❌ 服务器响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 服务器连接失败: {e}")
            return False
    
    def test_observation_capture(self):
        """测试观测捕获功能"""
        print("=== 测试观测捕获 ===")
        try:
            obs = self.capture_observation()
            print(f"✅ 关节位置维度: {len(obs['qpos'])}")
            print(f"✅ 图像数量: {len(obs['images'])}")
            
            for img_name, img in obs['images'].items():
                if isinstance(img, np.ndarray):
                    print(f"✅ {img_name}: {img.shape} {img.dtype}")
                else:
                    print(f"❌ {img_name}: 非numpy数组")
            
            # 测试base64编码
            for img_name, img in obs['images'].items():
                b64_str = self.image_to_base64(img)
                if b64_str:
                    print(f"✅ {img_name} base64编码成功, 长度: {len(b64_str)}")
                else:
                    print(f"❌ {img_name} base64编码失败")
            
            return True
            
        except Exception as e:
            print(f"❌ 观测捕获测试失败: {e}")
            return False
    
    def test_inference_request(self):
        """测试推理请求功能"""
        print("=== 测试推理请求 ===")
        try:
            # 更新观测窗口
            self.update_observation_window()
            
            # 准备推理数据
            obs_data = self.prepare_inference_data()
            print(f"✅ 准备推理数据成功")
            print(f"   - qpos维度: {len(obs_data['qpos'])}")
            print(f"   - 图像数量: {len(obs_data['images'])}")
            
            # 发送推理请求
            start_time = time.time()
            actions = self.request_inference(obs_data)
            inference_time = time.time() - start_time
            
            if actions is not None:
                print(f"✅ 推理成功, 耗时: {inference_time:.3f}s")
                print(f"   - 动作形状: {actions.shape}")
                print(f"   - 动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
                return True
            else:
                print(f"❌ 推理失败")
                return False
                
        except Exception as e:
            print(f"❌ 推理请求测试失败: {e}")
            return False
    
    def run_test_suite(self):
        """运行完整测试套件"""
        print("🚀 开始AIRBOT RDT完整测试")
        print("=" * 50)
        
        tests = [
            ("服务器连接测试", self.test_connection),
            ("观测捕获测试", self.test_observation_capture),
            ("推理请求测试", self.test_inference_request),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{test_name}...")
            try:
                result = test_func()
                results.append((test_name, result))
                status = "✅ 通过" if result else "❌ 失败"
                print(f"{test_name}: {status}")
            except Exception as e:
                print(f"❌ {test_name} 异常: {e}")
                results.append((test_name, False))
        
        # 输出结果汇总
        print("\n" + "=" * 50)
        print("📋 测试结果汇总:")
        all_passed = True
        for test_name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
            if not result:
                all_passed = False
        
        print(f"\n🎯 总体结果: {'✅ 全部通过' if all_passed else '❌ 存在问题'}")
        return all_passed
    
    def run_control_loop(self, use_relative=False):
        """运行主控制循环"""
        print("开始AIRBOT RDT控制循环...")
        print(f"控制模式: {'相对关节角' if use_relative else '绝对关节角'}")

        # 初始化观测窗口
        print("初始化观测窗口...")
        self.update_observation_window()
        print("观测窗口初始化完成")
        step = 0
        rate = 1.0 / self.control_freq  # 控制频率对应的时间间隔

        try:
            while step < self.max_steps:
                step_start = time.time()

                # 更新观测
                self.update_observation_window()
                
                # 如果是chunk起点且有next buffer，切换过去
                if step % self.chunk_size == 0 and self.next_action_buffer is not None:
                    self.action_buffer = self.next_action_buffer
                    self.next_action_buffer = None
                    print(f"步骤 {step}: 切换到提前加载的动作块")

                # 提前请求下一块动作（距离当前块末尾还有prefetch_steps步时）
                if (step % self.chunk_size) == (self.chunk_size - self.prefetch_steps):
                    print(f"步骤 {step}: 提前请求下一块动作...")
                    obs_data = self.prepare_inference_data(use_relative=use_relative)
                    inference_start = time.time()
                    # 这里同步调用，实际可改成异步线程
                    actions = self.request_inference(obs_data)
                    inference_time = time.time() - inference_start
                    if actions is not None:
                        self.next_action_buffer = actions
                        print(f"步骤 {step}: 提前请求完成")
                        print(f"推理完成，耗时: {inference_time:.3f}s")
                    else:
                        print(f"步骤 {step}: 提前请求失败，使用零动作")
                        self.action_buffer = np.zeros((self.chunk_size, 14))

                # # 当动作缓冲区用完时，请求新的动作序列
                # if step % self.chunk_size == 0:
                #     print(f"步骤 {step}: 请求新动作序列...")

                #     # 准备推理数据（根据use_relative选择关节角类型）
                #     obs_data = self.prepare_inference_data(use_relative=use_relative)

                #     # 请求推理
                #     inference_start = time.time()
                #     actions = self.request_inference(obs_data)
                #     inference_time = time.time() - inference_start

                #     if actions is not None:
                #         self.action_buffer = actions
                #         print(f"推理完成，耗时: {inference_time:.3f}s")
                #     else:
                #         print("推理失败，使用零动作")
                #         self.action_buffer = np.zeros((self.chunk_size, 14))

                # # 执行当前动作
                # if self.action_buffer is not None:
                #     action_idx = step % self.chunk_size
                #     action = self.action_buffer[action_idx]

                #     # 动作执行时也可以考虑是否使用相对控制
                #     if self.execute_action(action, use_relative=use_relative):
                #         print(f"步骤 {step}: 动作执行成功")
                #     else:
                #         print(f"步骤 {step}: 动作执行失败")
                overlap = self.prefetch_steps
                chunk_size = self.chunk_size

                if self.action_buffer is not None:
                    action_idx = step % chunk_size

                    # 融合区间：当前动作块的重叠部分
                    if self.next_action_buffer is not None and action_idx >= chunk_size - overlap:
                        t = action_idx - (chunk_size - overlap)
                        w = (overlap - t) / overlap  # 权重线性递减

                        action_A = self.action_buffer[action_idx]
                        action_B = self.next_action_buffer[t]

                        fused_action = w * action_A + (1 - w) * action_B
                        action_to_execute = fused_action
                        execute = True

                    # 跳过下一动作块重叠部分动作（重叠区间起点及之前动作）
                    elif (self.next_action_buffer is not None and
                        action_idx < overlap and
                        step >= chunk_size):  # 确保是在第二个动作块及以后
                        # 跳过这部分动作，不执行
                        print(f"步骤 {step}: 跳过下一动作块重叠区间动作索引 {action_idx}")
                        execute = False

                    else:
                        action_to_execute = self.action_buffer[action_idx]
                        execute = True

                    if execute:
                        success = self.execute_action(action_to_execute, use_relative=use_relative)
                        print(f"步骤 {step}: 动作执行{'成功' if success else '失败'}")

                step += 1

                # 控制循环频率
                elapsed = time.time() - step_start
                if elapsed < rate:
                    time.sleep(rate - elapsed)

        except KeyboardInterrupt:
            print("\n收到中断信号，停止控制循环")
        except Exception as e:
            print(f"控制循环异常: {e}")
        finally:
            print("机器人回到初始位置...")
            self.robot.back_home()
            time.sleep(2)

    # def run_control_loop(self):
        """运行主控制循环"""
        print("开始AIRBOT RDT控制循环...")
        
        # # 检查服务器连接
        # if not self.test_connection():
        #     print("服务器连接失败，退出控制循环")
        #     return
        
        # 初始化观测窗口
        print("初始化观测窗口...")
        self.update_observation_window()
        print("观测窗口初始化完成")
        step = 0
        rate = 1.0 / self.control_freq  # 控制频率对应的时间间隔
        
        try:
            while step < self.max_steps:
                step_start = time.time()
                
                # 更新观测
                self.update_observation_window()
                
                # 当动作缓冲区用完时，请求新的动作序列
                if step % self.chunk_size == 0:
                    print(f"步骤 {step}: 请求新动作序列...")
                    
                    # 准备推理数据
                    obs_data = self.prepare_inference_data()
                    
                    # 请求推理
                    inference_start = time.time()
                    actions = self.request_inference(obs_data)
                    inference_time = time.time() - inference_start
                    
                    if actions is not None:
                        self.action_buffer = actions
                        print(f"推理完成，耗时: {inference_time:.3f}s")
                    else:
                        print("推理失败，使用零动作")
                        self.action_buffer = np.zeros((self.chunk_size, 14))
                
                # 执行当前动作
                if self.action_buffer is not None:
                    action_idx = step % self.chunk_size
                    action = self.action_buffer[action_idx]
                    
                    if self.execute_action(action):
                        print(f"步骤 {step}: 动作执行成功")
                    else:
                        print(f"步骤 {step}: 动作执行失败")
                
                step += 1
                
                # 控制循环频率
                elapsed = time.time() - step_start
                if elapsed < rate:
                    time.sleep(rate - elapsed)
                
        except KeyboardInterrupt:
            print("\n收到中断信号，停止控制循环")
        except Exception as e:
            print(f"控制循环异常: {e}")
        finally:
            print("机器人回到初始位置...")
            self.robot.back_home()
            time.sleep(2)
    
    def __del__(self):
        if hasattr(self, 'robot'):
            self.robot.disconnect()

def main():
    # 配置参数
    SERVER_IP = "192.168.3.101"  # 替换为服务器A的实际IP地址
    SERVER_PORT = 8777
    SERVER_URL = f"http://{SERVER_IP}:{SERVER_PORT}"
    
    # 控制参数
    MAX_STEPS = 1000
    CHUNK_SIZE = 64
    CONTROL_FREQ = 25
    # 预取步数
    PREFETCH_STEPS = 5  # 提前5步开始请求下一动作块

    
    print("AIRBOT RDT控制器")
    print(f"目标服务器: {SERVER_URL}")
    print("请确保:")
    print("1. 服务器A已启动RDT推理服务")
    print("2. AIRBOT机器人已连接并启动")
    print("3. 网络连接正常")
    
    # 创建控制器
    controller = AIRBOTRDTController(
        server_url=SERVER_URL,
        max_steps=MAX_STEPS,
        chunk_size=CHUNK_SIZE,
        control_freq=CONTROL_FREQ,
        prefetch_steps=PREFETCH_STEPS 
    )
    
    # 交互式菜单
    while True:
        print("\n" + "=" * 40)
        print("请选择操作:")
        print("1. 运行完整测试套件")
        print("2. 测试服务器连接")
        print("3. 测试观测捕获")
        print("4. 测试推理请求")
        print("5. 运行控制循环")
        print("6. 机器人回初始位置")
        print("0. 退出")
        
        try:
            choice = input("\n请输入选择 (0-6): ").strip()
            
            if choice == '0':
                print("退出程序")
                break
            elif choice == '1':
                controller.run_test_suite()
            elif choice == '2':
                controller.test_connection()
            elif choice == '3':
                controller.test_observation_capture()
            elif choice == '4':
                controller.test_inference_request()
            elif choice == '5':
                print("⚠️  警告: 这将开始真实的机器人控制!")
                confirm = input("确认开始吗? (y/N): ").strip().lower()
                if confirm == 'y':
                    # 启动控制循环，并传入用户选择的模式
                    controller.run_control_loop(use_relative=False)
                else:
                    print("已取消")

            # elif choice == '5':
            #     print("⚠️  警告: 这将开始真实的机器人控制!")
            #     confirm = input("确认开始吗? (y/N): ").strip().lower()
            #     if confirm == 'y':
            #         controller.run_control_loop()
            #     else:
            #         print("已取消")
            elif choice == '6':
                controller.robot.back_home()
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，退出程序")
            break
        except Exception as e:
            print(f"操作错误: {e}")

if __name__ == "__main__":
    main()
