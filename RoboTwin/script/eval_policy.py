import sys
import os
import subprocess
import time
import json

os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
sys.path.append("./description/task_instruction")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
import json

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

# hjy add
def get_instruction_from_json(task_name):
    """从JSON获取任务指令"""
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # /home/Better-oft/RoboTwin/script/
    # 构建JSON文件的相对路径
    json_path = os.path.join(current_dir, "../description/task_instruction/all_tasks.json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            instructions = json.load(f)
        return instructions.get(task_name, "instruction get!")
    except:
        return "no!"


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main_remote(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    
    continue_evaling = usr_args.get("continue_evaling", False)
    
    save_dir = None
    video_save_dir = None
    video_size = None

    now_id = 0
    now_success = 0

    get_config = eval_function_decorator(policy_name, "get_config")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    checkpoint_path = usr_args["pretrained_checkpoint"]
    check_point_name=checkpoint_path.split('/')[-2] + '/' + '--'.join(checkpoint_path.split('--')[-2:]).replace('_chkpt', '')
    print(f'NOW CHECK POINT NAME:{check_point_name}')  # 输出: baseline-oft/472--40000
    # save_dir = Path(f"/media/abrain/Seagate/eval_results/{task_name}/{policy_name}/{task_config}/{check_point_name}/{current_time}")
    # save_dir = Path(f"/new_data/eval_result/{task_name}/{policy_name}/{check_point_name}/{current_time}")
    # 断点续训功能
    base_path = Path(f"/mnt/external_4tb/eval_result/{task_name}/{policy_name}/{task_config}/{check_point_name}")
    if continue_evaling and base_path.exists() and any(base_path.iterdir()):
        save_dir = max(base_path.iterdir(), key=lambda x: x.stat().st_mtime)
        file_path = os.path.join(save_dir, "_result.json")
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                now_id = data["latest_id"] + 1  # 从下一个id开始
                now_success = len(data["success_list"])
        except:
            now_id = 0  # 文件不存在时从0开始
    else:
        save_dir = base_path / current_time
        now_id = 0
    print(f"continue_evaling: {continue_evaling}")
    print(f"now_id: {now_id}")

    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    # hjy add this
    # seed = time.time() % 43

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 50
    topk = 1

    # hjy add a list to mark success eval_task id
    suc_list = []
   
    
    # model = get_model(usr_args)
    # st_seed, suc_num = eval_policy(task_name,
    #                                TASK_ENV,
    #                                args,
    #                                model,
    #                                st_seed,
    #                                test_num=test_num,
    #                                video_size=video_size,
    #                                instruction_type=instruction_type,
    #                                save_dir=save_dir)
    config = get_config(usr_args)
    # hjy changes
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                #    model,
                                   config,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   now_id=now_id,
                                   now_success=now_success,
                                   save_dir=save_dir)
    suc_nums.append(suc_num)
    # if (suc_id != -1):
    #     suc_list.append(suc_id)
    print(f"now success list:{suc_list}")
    
    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]


    file_path = os.path.join(save_dir, f"_result.json")
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except:
        data = {"latest_id": None, "success_list": []}

    # 更新最终结果
    data.update({
        "timestamp": current_time,
        "instruction_type": instruction_type,
        "success_rates": (np.array(suc_nums) / test_num).tolist()
    })

    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)

    print(f"Data has been saved to {file_path}")


def eval_policy(task_name,
                TASK_ENV,
                args,
                # model,
                config,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                now_id=0,
                now_success=0,
                save_dir=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = now_success
    TASK_ENV.test_num = now_id

    succ_seed = now_id
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval_remote")
    reset_func = eval_function_decorator(policy_name, "reset_remote_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                print("AFTER SETUP DEMOOOOOOOOO")
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        # hjy changes this
        # instruction = np.random.choice(results[0][instruction_type])
        instruction = get_instruction_from_json(task_name)
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{now_id}.mp4",
                    # test_num改成了now_id，不然断点续评的时候会重新覆盖
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(config.pretrained_checkpoint, config.unnorm_key)  # hjy changes
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, config.pretrained_checkpoint, config.unnorm_key, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        print(f"success num: \033[98m {TASK_ENV.eval_success} \033[0m")

        TASK_ENV.index += 1
        print(f"🍉TASK_INDEX IS :{TASK_ENV.index}")
        if succ:
            TASK_ENV.suc += 1

            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # 读取或初始化结果
        file_path = os.path.join(save_dir, f"_result.json")
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except:
            data = {"latest_id": None, "success_list": []}

        # 更新断点
        data["latest_id"] = now_id
        if succ and now_id not in data["success_list"]:
            data["success_list"].append(now_id)

        # 写回文件
        with open(file_path, "w") as file:
            json.dump(data, file, indent=2)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc  # suc_id


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main_remote(usr_args)
