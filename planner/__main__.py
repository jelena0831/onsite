# 导入内置库
import sys
import csv
import os
import time
import json
import numpy as np
import math
import importlib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from pathlib import Path as PathlibPath

# 导入第三方库
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from typing import Dict,List,Tuple,Optional,Union,Any

# 导入onsite-unstructured模块
## onsite-unstructured.dynamic_scenes模块
scenarioOrganizer = importlib.import_module("onsite-unstructured.dynamic_scenes.scenarioOrganizer1")
env = importlib.import_module("onsite-unstructured.dynamic_scenes.env")
controller = importlib.import_module("onsite-unstructured.dynamic_scenes.controller")
lookup = importlib.import_module("onsite-unstructured.dynamic_scenes.lookup")

# 导入本地模块
from predictor import Predictor
from planner import Planner
from simple_control import MotionController

# 检查对应文件夹是否存在
def check_dir(target_dir):
    """check path"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir,exist_ok=True)

# 输入高度矩阵,高度矩阵为二维列表，共计41*41个数据
def get_slope(observation, height):
    z1 = height[20][20]
    dx = observation['vehicle_info']['ego']['shape']['length']*np.cos(observation['vehicle_info']['ego']['yaw_rad'])
    dy = observation['vehicle_info']['ego']['shape']['length']*np.sin(observation['vehicle_info']['ego']['yaw_rad'])
    x_idx = math.floor(dx+20)
    y_idx = math.floor(dy+20)
    x_idx = max(0, min(x_idx, 40))
    y_idx = max(0, min(y_idx, 40))
    z2 = height[x_idx][y_idx]

    distance = math.sqrt(dx ** 2 + dy ** 2)
    return -math.atan2((z2-z1), distance)

dir_current_file = os.path.dirname(__file__)  # 'algorithm\planner'
dir_parent_1 = os.path.dirname(dir_current_file) # 'algorithm'


if __name__ == "__main__":

    dir_inputs = os.path.abspath(os.path.join(dir_parent_1, 'inputs'))  # 场景文件的位置
    dir_outputs = os.path.abspath(os.path.join(dir_parent_1, 'outputs'))  # 输出轨迹的位置
    dir_save_img = os.path.abspath(os.path.join(dir_parent_1, 'onsite_images_saved'))  # 图像保存位置
    tic = time.time()
    so = scenarioOrganizer.ScenarioOrganizer()  # 初始化场景管理模块，用于加载测试场景
    envi = env.Env()  # 初始化测试环境
    # 根据配置文件config.py装载场景,指定输入文件夹即可,会自动检索配置文件
    so.load(dir_inputs, dir_outputs)  # 根据配置文件config.py加载待测场景，指定输入文件夹即可，会自动检索配置文件
    formatted_so_config = json.dumps(so.config, indent=4, ensure_ascii=False)  # 清晰展示各种设置，没有特殊含义
    print(f"###log### <测试参数>\n{formatted_so_config}\n")

    # 初始化log，清除历史log
    
    log_file_path = os.path.abspath(os.path.join(dir_current_file, 'output.log'))
    log_file = open(log_file_path, "w")
    original_stdout = sys.stdout
    sys.stdout = log_file
    print("start testing...")
    sys.stdout = original_stdout
    log_file.close()

    while True:
        scenario_to_test = so.next()  # !使用场景管理模块给出下一个待测场景
        if scenario_to_test is None:
            break  # !如果场景管理模块给出None,意味着所有场景已测试完毕.
        print(f"###log### <scene-{scenario_to_test['data']['scene_name']}>\n")  # 输出待测场景信息
        try:
            scene_type = ""
            shovel_scene_type = "shovel"
            intersection_scene_type = "intersection"
            mission_scene_type = "B"
            if shovel_scene_type in scenario_to_test['data']['scene_name']:
                scene_type = "shovel"
            elif intersection_scene_type in scenario_to_test['data']['scene_name']:
                scene_type = "intersection"
            elif mission_scene_type in scenario_to_test['data']['scene_name']:
                scene_type = "B"
            # 使用env.make方法初始化当前测试场景
            # （1）observation包含当前帧环境信息；
            # （2）选择kinetics_mode=complex
            # （3）禁止以任何方式读取赛题中的背景车全局轨迹，通过直接或间接手段读取利用题目中的动态障碍车完整轨迹获利，将直接判作弊
            #      车辆观测范围内的动态障碍车的当前位姿、速度、加速度及转角会包含在observation中，请根据此信息进行障碍车未来轨迹预测
            observation, client, init_height = envi.make(scenario=scenario_to_test, dir_inputs=dir_inputs, save_img_path=dir_save_img,
                                                  kinetics_mode='complex', scene_type=scene_type)  
            
            slope = get_slope(observation, init_height)

            log_file = open(log_file_path, "a")
            original_stdout = sys.stdout
            sys.stdout = log_file
            print(f"###log### <scene-{scenario_to_test['data']['scene_name']}>\n")  # 输出待测场景信息
            print(f"init_observation:{observation}")
            sys.stdout = original_stdout
            log_file.close()

            ########### 算法 部分1:初始化 ###########
            # 交叉路口单车通行任务:构建整个场景的预测器,预测所有的车辆;构建规划器\控制器,控制ego车
            predictor = Predictor(time_horizon=3.0)
            planner = Planner(observation)
            collision_lookup = lookup.CollisionLookup()
            path_planned, spd_planned = planner.process(observation, scene_type, collision_lookup)

            if len(path_planned) == 0:
                print("###log### ERROR: 不存在全局参考路径，本场景结束")
                if client is not None:
                    client.close_sockets()
                continue            

            simple_controller = MotionController()

            # 逐帧进行仿真，触发仿真停止条件时结束
            # 当测试还未进行完毕,即观察值中test_setting['end']还是-1的时候
            while observation['test_setting']['end'] == -1:
                ########### 算法 部分2:执行 ###########
                t0=time.time()
                # 预测动态障碍车未来轨迹
                traj_future = predictor.predict(observation, predictor_type="CACV_PREDICTOR")

                # log_file = open(log_file_path, "a")
                # original_stdout = sys.stdout
                # sys.stdout = log_file
                # print(f"traj_future:{traj_future}")
                # sys.stdout = original_stdout
                # log_file.close()

                # steer为负的时候向右，正的时候向左
                action = simple_controller.process(observation['vehicle_info'], path_planned, 
                                                   spd_planned, traj_future, observation, scene_type, slope)
                
                # step函数与环境进行交互，返回新的观测值，action--控制指令，traj_future--预测轨迹(仅用于可视化，可以不给)，traj--背景车轨迹, path_planned--规划的全局参考路径(仅用于可视化，可以不给)
                # height是二维高度矩阵(41*41)，自车高度对应height[20][20]，自车左侧10m(x-10)的位置的高程对应height[10][20]，(y+10)对应height[20][30]
                # 高程为-1的时候代表这个位置是障碍物，vis_cir_open=ture时可视化会给出一个以自车为中心的20m半径的圆来表示观测范围
                observation, height = envi.step(action, traj_future, observation, path_planned=path_planned, vis_cir_open=True)  # 根据车辆的action,更新场景,并返回新的观测值.

                slope = get_slope(observation, height)
                log_file = open(log_file_path, "a")
                original_stdout = sys.stdout
                sys.stdout = log_file
                print(f"slope:{slope}")
                sys.stdout = original_stdout
                log_file.close()


        # except Exception as e:
        #     print(repr(e))
        #     if client is not None:
        #         client.close_sockets()
        finally:
            # 如果测试完毕，将测试结果传回场景管理模块（ScenarioOrganizer)
            so.add_result(scenario_to_test, observation['test_setting']['end'])
            # 在每一次测试最后都关闭可视化界面，避免同时存在多个可视化
            plt.close()

    toc = time.time()
    print("###log### 总用时:", toc - tic, "秒\n")
         


