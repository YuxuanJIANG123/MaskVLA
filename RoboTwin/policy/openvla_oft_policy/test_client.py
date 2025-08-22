# test_client.py - 客户端测试程序

import requests
import numpy as np
import base64
import cv2
import json

def create_fake_obs():
    """创建模拟的观察数据"""
    # 创建假的图像数据
    fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 转换为base64
    def encode_image_to_base64(image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    # 构建观察数据
    obs = {
        "instruction": "",
        "images": {
            "front_t": encode_image_to_base64(fake_image),
            "right_t": encode_image_to_base64(fake_image),
            "left_t": encode_image_to_base64(fake_image),
        },
        "full_image": [fake_image.tolist(), fake_image.tolist(), fake_image.tolist()],
        "state": np.random.rand(14).tolist(),
        "qpos": np.random.rand(14).tolist()
    }
    return obs

def send_and_receive(obs, instruction, ckpt_path, unnorm_key, server_url="http://192.168.3.101:8080"):
    """发送观察数据到远程服务器，接收动作序列"""
    data = {"observation": obs, "instruction": instruction, "ckpt_path": ckpt_path, "unnorm_key": unnorm_key}
    response = requests.post(f"{server_url}/predict", json=data, timeout=300)
    result = response.json()
    
    if not result["success"]:
        raise Exception(f"推理失败: {result['error']}")
    
    # 转换回numpy数组
    return [np.array(action) for action in result["actions"]]

def reset_remote_model(ckpt_path, unnorm_key, server_url="http://192.168.3.101:8080"):
    """重置远程模型"""
    data = {"ckpt_path": ckpt_path, "unnorm_key": unnorm_key}
    response = requests.post(f"{server_url}/reset", json=data, timeout=100)
    result = response.json()
    
    if not result["success"]:
        raise Exception(f"重置失败: {result['error']}")
    
    print(f"✅ {result['message']}")

def test_communication():
    """测试通信"""
    server_url = "http://192.168.3.101:8080"
    
    # 测试参数
    ckpt_path = "/new_data/ckpt/openvla-oft/sim/adjust_bottle/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--50000_chkpt"
    unnorm_key = "my_aloha_sim_adjust_bottle"
    instruction = "pick up the bottle"

    print("🚀 开始测试客户端-服务器通信...")
    
    try:
        # 1. 测试健康检查
        print("\n1. 测试健康检查...")
        response = requests.get(f"{server_url}/health", timeout=50)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 服务器健康: {health_data}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return
        
        # 2. 测试模型推理
        print("\n2. 测试模型推理...")
        obs = create_fake_obs()
        print(f"📦 发送观察数据大小: {len(json.dumps(obs))} 字节")
        
        actions = send_and_receive(obs, instruction, ckpt_path, unnorm_key, server_url)
        print(f"✅ 推理成功!")
        print(f"📋 收到动作数量: {len(actions)}")
        print(f"📋 第一个动作形状: {actions[0].shape}")
        print(f"📋 第一个动作值: {actions[0][:5]}...")  # 只显示前5个值
        
        # 3. 测试模型重置
        print("\n3. 测试模型重置...")
        reset_remote_model(ckpt_path, unnorm_key, 2, server_url)
        
        # 4. 再次测试推理
        print("\n4. 重置后再次测试推理...")
        actions2 = send_and_receive(obs, instruction, ckpt_path, unnorm_key, server_url)
        print(f"✅ 重置后推理成功!")
        print(f"📋 收到动作数量: {len(actions2)}")
        
        print("\n🎉 所有测试通过!")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器: {server_url}")
        print("请确保服务器正在运行")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_communication()
