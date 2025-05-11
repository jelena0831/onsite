  # 内置库 
import sys
import os
import math
import importlib


# 第三方库
import numpy as np
import bisect
from typing import Dict,List,Tuple,Optional,Union

observation = importlib.import_module("onsite-unstructured.dynamic_scenes.observation")
utils = importlib.import_module("onsite-unstructured.common.utils")



class Predictor:
    """轨迹模块，针对背景车(周围障碍物车辆)轨迹预测器;"""

    def __init__(self, time_horizon: float = 2.0) -> None:
        """默认构造 Predictor.
        time_horizon: 预测时域,默认为5s .
        """
        self.prediction_total_time_ = time_horizon
        self.veh_traj_predicted = {}
        self.veh_predictor_info = dict()

    def predict(
        self,
        observation: Dict = None,
        predictor_type="CACV_PREDICTOR",
    ) -> List[Dict]:
        """调用轨迹预测器,进行目标车辆轨迹预测.

        :traj:待预测的背景车的轨迹;
        :observation :场景实时的观察结果;
        :veh_traj_predicted : 返回值,目标车辆未来轨迹预测信息;
        """
        self.predictor_type_ = predictor_type
        t_now_ = round(float(observation["test_setting"]["t"]), 2)
        interval_ = observation["test_setting"]["dt"]
        max_t_ = observation["test_setting"]["max_t"]

        # 当前时刻所有背景车的状态信息
        self.veh_traj_predicted = {}  # 每一个step,所有的veh_traj_predicted都清空
        for id_veh in observation["vehicle_info"].keys():
            if id_veh == "ego":
                continue

            self.veh_traj_predicted.update({id_veh: {}})
            vehicle_info = observation["vehicle_info"][id_veh]
            for key, value in vehicle_info.items():
                self.veh_traj_predicted[id_veh][key] = value

        # 不同的背景车可以使用不同的预测器 ######################################
        if self.predictor_type_ == "CACV_PREDICTOR":
            # 预测多个车的轨迹,使用简单的预测器
            for id_veh in observation["vehicle_info"].keys():
                if id_veh == "ego":
                    continue
                CACV_Predictor_ = ConstantAngularVelocityPredictor(t_now_, interval_, max_t_, self.prediction_total_time_)
                traj_future_ = CACV_Predictor_.predict(self.veh_traj_predicted[id_veh], t_now_)  # 调用单个车辆预测器
                for key, value in traj_future_.items():
                    self.veh_traj_predicted[id_veh][key] = value
                self.veh_traj_predicted[id_veh][-1] = {}
                self.veh_traj_predicted[id_veh]["traj_future_for_vis"] = traj_future_

        else:
            print("### log ### error:predictor_type有问题!")

        return self.veh_traj_predicted


class ConstantAngularVelocityPredictor:
    """利用纵向速度和角速度从当前位置推算出一条曲线
    extrapolates a curved line from current position using linear and angular velocity
    """

    def __init__(
        self,
        t_now: float = 0.0,
        interval: float = 0.1,
        max_t: float = 18.0,
        time_horizon: float = 1.5,
    ) -> None:
        """默认构造 获取每个场景后,进行初始化
        默认时间间隔0.1, 还有一些时间间隔是0.04;
        """
        self.prediction_total_time_ = time_horizon
        self.dt_ = interval
        self.t_now_ = t_now
        self.max_t_ = max_t
        self.traj_predict_ = dict()

    def predict(self, observation: Dict, t_now_: float) -> Dict:
        self.t_now_ = t_now_  #! 切忌要更新该值
        t = t_now_
        dt = self.dt_
        x = observation["x"]
        y = observation["y"]
        v = observation["v_mps"]
        yaw = observation["yaw_rad"]
        acc = observation["acc_mpss"]
        yaw_rate = observation["yawrate_radps"]

        delta_t = self.max_t_ - self.t_now_
        if delta_t < self.prediction_total_time_:
            self.numOfTrajPoint_ = int(delta_t / self.dt_)
        else:
            self.numOfTrajPoint_ = int(self.prediction_total_time_ / self.dt_) 

        for i in range(self.numOfTrajPoint_):
            t += dt
            x += dt * (v * np.cos(yaw) + acc * np.cos(yaw) * 0.5 * dt + yaw_rate * v * np.sin(yaw) * 0.5 * dt)  # 确定yaw定义
            y += dt * (v * np.sin(yaw) + acc * np.sin(yaw) * 0.5 * dt + yaw_rate * v * np.cos(yaw) * 0.5 * dt)
            yaw += dt * yaw_rate
            v += dt * acc

            if self.dt_ < 0.1:
                str_time = str(round(t, 3))
            else:
                str_time = str(round(t, 2))
            self.traj_predict_[str_time] = {
                "x": round(x, 2),
                "y": round(y, 2),
                "v": round(v, 2),
                "a": round(acc, 2),
                "yaw": round(yaw, 3),
            }

        return self.traj_predict_

if __name__ == "__main__":
    import time
    # 调试代码