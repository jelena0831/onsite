# 内置库
import math
from queue import PriorityQueue
import statistics
import os
import csv

# 第三方库
import numpy as np
from shapely.geometry import Point, Polygon  #待安装
from typing import Dict, List, Tuple, Optional, Union
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning


class HA_Planner:
    """ego车的轨迹规划器.
    注:业界一般做法是 path planning + speed planning .
    """
    def __init__(self, observation):
        self._goal_x = statistics.mean(observation['test_setting']['goal']['x'])
        self._goal_y = statistics.mean(observation['test_setting']['goal']['y'])
        self._goal_yaw = observation['test_setting']['goal']['head'][0]
        self._observation = observation
    
    
    def process(self,collision_lookup,observation):
        """规划器主函数.
        注：该函数功能设计举例
        1) 进行实时轨迹规划;
        2) 路径、速度解耦方案:给出局部轨迹规划器的实时求解结果--待行驶路径、待行驶速度；
        
        输入:observation——环境信息;
        输出: 路径、速度解耦方案:给出局部轨迹规划器的实时求解--
            待行驶路径（离散点序列）、待行驶速度（与路径点序列对应的速度离散点序列）
        """
        # 设置目标速度
        target_speed = 8.0
        target_speed_backward = 10.0/3.6

        goal = [np.mean(observation['test_setting']['goal']['x']), 
                np.mean(observation['test_setting']['goal']['y']), 
                observation['test_setting']['goal']['head'][0]]
        spd_planned = []
        # path_planned = self.get_ego_reference_path_to_goal(observation)
        keypoint_ind = 0
        if observation["test_setting"]["scenario_type"] == "loading":
            path_planned = self.get_main_road(observation)
            print('len(main_road)',len(path_planned))
            nearest_point, point_ind, min_distance = self.find_nearest_point(path_planned, goal)
            print("min_distance:",min_distance)
            print("nearest_point",nearest_point)
            dist = 0
            while point_ind:
                dist += math.sqrt((path_planned[point_ind][0] - path_planned[point_ind-1][0])**2 + (path_planned[point_ind][1] - path_planned[point_ind-1][1])**2)
                point_ind -= 1
                if dist > 30:
                    break
            path_planned = path_planned[0:point_ind+1]
            print("len(path_planned):",len(path_planned))
            keypoint_ind = len(path_planned)
            # print("keypoint:",keypoint)
            for i in range(len(path_planned)):
                path_planned[i]=path_planned[i][0:3]
                path_planned[i].append(True)
            handover = path_planned[-1] # 取最后一个点作为handover点
            print("handover:",handover) #打印handover信息
            astar_path = hybrid_a_star_planning(handover, goal, collision_lookup, observation, 1.0, 5,True)
            if astar_path is not None:
                # 打印路径点数量
                print("len(astar_path):",len(astar_path.xlist))
                # 将混合A*规划的路径点转换为统一格式
                for j in range(1,len(astar_path.xlist)):
                    path_planned.append([astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                    # 检测前进/后退的切换点
                    if astar_path.directionlist[j] == True and astar_path.directionlist[j+1] == False:
                        print("keypoint:",[astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                        keypoint_ind += j-1
                        #print("keypoint:",keypoint)
                print("keypoint_ind:",keypoint_ind)
                print("len(path):",len(path_planned))
                path_planned[keypoint_ind][-1]=2
                #速度规划
                for k in range(len(path_planned)):
                    # print("k=",k)
                    if k < 10:
                        spd_planned.append(target_speed / 10 * k)
                    elif k >= 10 and k <= keypoint_ind-20:
                        spd_planned.append(target_speed)
                    elif (k > keypoint_ind-20) and (k <= keypoint_ind):
                        spd_planned.append(target_speed/ 20 * (keypoint_ind-k))
                    elif k > keypoint_ind and k <= keypoint_ind+20:
                        spd_planned.append(-target_speed_backward / 40 * (k-keypoint_ind+20))
                    elif k > keypoint_ind+20 and k < len(path_planned)-20:
                        spd_planned.append(-target_speed_backward)
                    elif k >= len(path_planned)-20:
                        spd_planned.append(-target_speed_backward / 20 * (len(path_planned)-k-1))
            else:
                return [],[] #规划不出路径
        elif observation["test_setting"]["scenario_type"] == "unloading":
            # astar_star=[self._ego_x,self._ego_y,self._ego_yaw]
            path_planned = self.get_main_road(observation,False)
            astar_star = [observation['vehicle_info']['ego']['x'],
                          observation['vehicle_info']['ego']['y'],
                          observation['vehicle_info']['ego']['yaw_rad']]
            nearest_point, point_ind, min_distance = self.find_nearest_point(path_planned, astar_star)
            dist = 0
            while point_ind < len(path_planned):
                dist += math.sqrt((path_planned[point_ind][0] - path_planned[point_ind-1][0])**2 + (path_planned[point_ind][1] - path_planned[point_ind-1][1])**2)
                point_ind += 1
                # print(path_planned[point_ind+1][0])
                if dist > 20:
                    break
            print("min_distance:",min_distance)
            print("nearest_point",nearest_point)
            # print(path_planned)
            path_planned = path_planned[point_ind-1:]
            # print(path_planned)
            handover = path_planned[0]
            print("handover:",handover)
            astar_path = hybrid_a_star_planning(astar_star, handover, collision_lookup, observation, 1.0, 5,False)
            if astar_path is not None:
                print("len(astar_path):",len(astar_path.xlist))
                for i in range(len(path_planned)):
                    path_planned[i]=path_planned[i][0:3]
                    path_planned[i].append(True)
                path_planned1 = []
                for j in range(1,len(astar_path.xlist)):
                    path_planned1.append([astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                path_planned = path_planned1 + path_planned
                print("len(path):",len(path_planned))
                for k in range(len(path_planned)):
                    if k < 50:
                        spd_planned.append(target_speed / 50 * k)
                    else:
                        spd_planned.append(target_speed)
            else:
                return [],[]
        # print(path_planned)
        for i in range(1,len(path_planned)):
            path_planned[i].append(self.calculate_curvature(path_planned[i-1][0],path_planned[i-1][1],path_planned[i-1][2],
                                                            path_planned[i][0],path_planned[i][1],path_planned[i][2]))
        return path_planned[1:],spd_planned[1:]
        
        
    def get_main_road(self,observation,bo_in=True):
        """获取HD Map中ego车到达铲装平台的参考路径.     
        输入:observation——环境信息;
        输出:ego车到达目标区域的参考路径(拼接后).
        """
        main_road={"polygon-14":[["path-2","path-3","path-4","path-48","path-50","path-43","path-44"],
                                 ["path-13","path-14","path-58","path-31","path-37","path-38","path-39"]],
                   "polygon-27":[["path-36","path-5","path-6","path-7","path-8","path-9","path-70","path-71","path-72","path-73"],
                                 ["path-17","path-18","path-19","path-20","path-21","path-65","path-66","path-67","path-64","path-80","path-81"]],
                   "polygon-29":[["path-36","path-5","path-6","path-7","path-8","path-9","path-84","path-85","path-74","path-75"],
                                 ["path-22","path-23","path-59","path-65","path-66","path-67","path-64","path-80","path-81"]],
                   "polygon-25":[["path-36","path-5","path-6","path-7","path-8","path-9","path-68","path-69"],
                                 ["path-15","path-16","path-65","path-66","path-67","path-64","path-80","path-81"]],
                   "polygon-10":[["path-2","path-45","path-46","path-47","path-40","path-41"],
                                 ["path-77","path-10","path-11","path-27","path-28","path-25","path-26","path-39"]]}
        #################更新参数#################
        self._ego_x = observation['vehicle_info']['ego']['x']
        self._ego_y = observation['vehicle_info']['ego']['y']
        self._ego_v = observation['vehicle_info']['ego']['v_mps']
        self._ego_yaw = observation['vehicle_info']['ego']['yaw_rad']
        #################定位主车和目标点所在几何#################
        if bo_in:
        #bo_in参数是一个布尔值（Boolean），用于控制路径规划的方向。当它等于True时，表示规划的路径是从起点到终点；当它等于False时，表示规划的路径是从终点到起点。
            ego_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._ego_x,self._ego_y)
            ego_polygon_id = int( ego_polygon_token.split('-')[1])
            print("ego_polygon_token:",ego_polygon_token)
            goal_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._goal_x,self._goal_y)
            print("goal_polygon_token:",goal_polygon_token)
            ego_dubinspose_token = self.get_dubinspose_token_from_polygon\
                                (observation,(self._ego_x,self._ego_y,self._ego_yaw),ego_polygon_token)
            print("ego_dubinspose_token:",ego_dubinspose_token)
        else:
            ego_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._goal_x,self._goal_y)
            ego_polygon_id = int( ego_polygon_token.split('-')[1])
            print("ego_polygon_token:",ego_polygon_token)
            goal_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._ego_x,self._ego_y)
            print("goal_polygon_token:",goal_polygon_token)
            ego_dubinspose_token = self.get_dubinspose_token_from_polygon\
                                (observation,(self._goal_x,self._goal_y,self._goal_yaw),ego_polygon_token)
            print("ego_dubinspose_token:",ego_dubinspose_token)
        #################获取目标车最匹配的dubinspose#################
        
        # ego_dubinspose_id = int( ego_dubinspose_token.split('-')[1])
        # ego_dubinspose_token 作为 起点、终点的path拿到
        link_referencepath_tokens_ego_polygon = observation['hdmaps_info']['tgsc_map'].polygon[ego_polygon_id]['link_referencepath_tokens']
        # 去除掉 不包含 ego_dubinspose_token 的 path
        for _,path_token in enumerate(link_referencepath_tokens_ego_polygon):
            path_id = int( path_token.split('-')[1])
            link_dubinspose_tokens = observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['link_dubinspose_tokens']
            if ego_dubinspose_token not in link_dubinspose_tokens:
                pass
            else:
                only_one_path_token = path_token 
                only_one_path_id= path_id 
        print("path_token:",path_token)
        path_connected = []
        for road in main_road[goal_polygon_token]:
            if only_one_path_token in road:
                ind = road.index(only_one_path_token)
                terminal_path_id = int(road[ind].split('-')[1])
                terminal_path = observation['hdmaps_info']['tgsc_map'].reference_path[terminal_path_id]['waypoints']
                if bo_in:
                    _, nearest_point_ind, _ = self.find_nearest_point(terminal_path, [self._ego_x,self._ego_y])
                    terminal_path = terminal_path[nearest_point_ind:]
                    road_1 = road[ind+1:len(road)] # 获取从索引ind+1到列表末尾的所有元素
                    for road_segment in road_1:
                        path_id = int(road_segment.split('-')[1])
                        path_connected += observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['waypoints']
                    path_connected = terminal_path + path_connected
                else:
                    _, nearest_point_ind, _ = self.find_nearest_point(terminal_path, [self._goal_x,self._goal_y])
                    terminal_path = terminal_path[0:nearest_point_ind+1]
                    road_1 = road[:ind]
                    for road_segment in road_1:
                        path_id = int(road_segment.split('-')[1])
                        path_connected += observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['waypoints']
                    path_connected =  path_connected + terminal_path
        return path_connected
    
    def get_dubinspose_token_from_polygon(self,observation,veh_pose:Tuple[float,float,float],polygon_token:str):
        id_polygon = int(polygon_token.split('-')[1])
        link_dubinspose_tokens = observation['hdmaps_info']['tgsc_map'].polygon[id_polygon]['link_dubinspose_tokens']
        dubinsposes_indicators = []
        for token in link_dubinspose_tokens:
            id_dubinspose = int(token.split('-')[1])
            dx = observation['hdmaps_info']['tgsc_map'].dubins_pose[id_dubinspose]['x'] - veh_pose[0]
            dy = observation['hdmaps_info']['tgsc_map'].dubins_pose[id_dubinspose]['y'] - veh_pose[1]
            dyaw = observation['hdmaps_info']['tgsc_map'].dubins_pose[id_dubinspose]['yaw'] - veh_pose[2]
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 3 and abs(dyaw) < 1e-1:
                return token
    
    def line_equation(self,x, y, angle):
            if angle != np.pi / 2 and angle != -np.pi / 2:  # 避免斜率为无限大的情况
                m = np.tan(angle)
                b = y - m * x
                return m, b
            else:
                return float('inf'), x  # 对于垂直线，斜率为无限大，返回x作为截距

    def calculate_curvature(self,x1, y1, angle1, x2, y2, angle2):
        # 中点坐标
        # mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # 如果两个方向角度相同，曲线不是圆弧，是直线
        if angle1 == angle2:
            return 0  # 直线的曲率为0，倒数为无穷大

        # 计算垂直于两点连线的直线的斜率（即中垂线）
        # 避免除以零的错误
        # if x2 != x1:
        #     slope_perpendicular = -(x2 - x1) / (y2 - y1)
        # else:
        #     slope_perpendicular = float('inf')
        
        # 使用点斜式方程计算两条直线的方程
        # y = mx + b，通过一个点和斜率求b
        # 计算两个方向的直线方程
        m1, b1 = self.line_equation(x1, y1, angle1)
        m2, b2 = self.line_equation(x2, y2, angle2)

        # 找到圆心（两条直线的交点）
        if m1 != float('inf') and m2 != float('inf'):
            cx = (b2 - b1) / (m1 - m2)
            cy = m1 * cx + b1
        elif m1 == float('inf'):
            cx = b1
            cy = m2 * cx + b2
        else:
            cx = b2
            cy = m1 * cx + b1

        # 计算圆心到任一点的距离（半径）
        radius = np.sqrt((cx - x1)**2 + (cy - y1)**2)

        # 计算曲率
        curvature = 1 / radius

        return curvature
        
    def find_nearest_point(self,path, point):
        """
        在路径上找到给定点的最近点。
        
        参数:
        - path: 路径点的列表，每个点是一个(x, y)元组。
        - point: 给定点，一个(x, y)元组。
        
        返回:
        - nearest_point: 路径上最近的点。
        - min_distance: 到给定点的最小距离。
        """
        
        # 初始化最小距离和最近点
        min_distance = float('inf')
        point_ind = 0
        # print(path)
        # 遍历路径上的每个点
        for i in range(len(path)):
            path_point = path[i]
            # 计算当前点与给定点之间的欧几里得距离
            distance = math.sqrt((path_point[0] - point[0])**2 + (path_point[1] - point[1])**2)
            
            # 更新最小距离和最近点
            if distance < min_distance:
                point_ind = i
                nearest_point = path_point
                min_distance = distance
        
        return nearest_point, point_ind,min_distance

class Planner:
    """
    ego车的规划器.
    包括：全局路径规划器；局部轨迹规划器；
    注:局部轨迹规划器业界一般做法是 path planning + speed planning .
    """

    def __init__(self, observation):
        self._goal_x = statistics.mean(observation["test_setting"]["goal"]["x"])
        self._goal_y = statistics.mean(observation["test_setting"]["goal"]["y"])
        self._goal_yaw = None
        self._observation = observation
    
    def line_equation(self,x, y, angle):
        if angle != np.pi / 2 and angle != -np.pi / 2:  # 避免斜率为无限大的情况
            m = np.tan(angle)
            b = y - m * x
            return m, b
        else:
            return float('inf'), x  # 对于垂直线，斜率为无限大，返回x作为截距


    def calculate_curvature(self,x1, y1, angle1, x2, y2, angle2):
        # 中点坐标
        # mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # 如果两个方向角度相同，曲线不是圆弧，是直线
        if angle1 == angle2:
            return 0  # 直线的曲率为0，倒数为无穷大

        # 计算垂直于两点连线的直线的斜率（即中垂线）
        # 避免除以零的错误
        # if x2 != x1:
        #     slope_perpendicular = -(x2 - x1) / (y2 - y1)
        # else:
        #     slope_perpendicular = float('inf')
        
        # 使用点斜式方程计算两条直线的方程
        # y = mx + b，通过一个点和斜率求b
        # 计算两个方向的直线方程
        m1, b1 = self.line_equation(x1, y1, angle1)
        m2, b2 = self.line_equation(x2, y2, angle2)

        # 找到圆心（两条直线的交点）
        if m1 != float('inf') and m2 != float('inf'):
            cx = (b2 - b1) / (m1 - m2)
            cy = m1 * cx + b1
        elif m1 == float('inf'):
            cx = b1
            cy = m2 * cx + b2
        else:
            cx = b2
            cy = m1 * cx + b1

        # 计算圆心到任一点的距离（半径）
        radius = np.sqrt((cx - x1)**2 + (cy - y1)**2)

        # 计算曲率
        curvature = 1 / radius

        return curvature

    def process(self, observation, scene_type, collision_lookup=None):
        """规划器主函数，提供不同场景下的全局路径规划和速度规划方案供参考，鼓励参与者重写具体路径规划逻辑以实现更好的效果

        注：该函数功能设计举例
        0) 全局路径寻优,route.
        1）进行实时轨迹规划;
        2) 路径、速度解耦方案:给出局部轨迹规划器的实时求解结果--待行驶路径、待行驶速度；

        输入:observation——环境信息;
        输出: 路径、速度解耦方案:给出局部轨迹规划器的实时求解--
            待行驶路径（离散点序列）、待行驶速度（与路径点序列对应的速度离散点序列）
        """
        # 交叉口场景采用地图中给的参考路径进行路网规划
        if scene_type == "intersection":
            route_path = self.get_ego_reference_path_to_goal(observation)
            path_planned = route_path
            spd_planned = None
        elif scene_type == "shovel":
            self._goal_yaw = observation['test_setting']['goal']['head'][0]
            shovel_Planner = HA_Planner(observation)
            path_planned, spd_planned = shovel_Planner.process(collision_lookup, observation)
            
            dir_current_file = os.path.dirname(__file__)  # 'algorithm\planner'
            filename = os.path.abspath(os.path.join(dir_current_file, 'temp_hybrid_astar_path.csv'))
            HA_path_headers = ['x', 'y', 'yaw', 'direction', 'curvature']
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(HA_path_headers)
                writer.writerows(path_planned)
            print(f"shovel场景混合A*全局路径数据已保存到 {filename}")
        elif scene_type == "B":
            target_speed = 10
            target_speed_backward = 8
            goal = [np.mean(observation['test_setting']['goal']['x']), 
                np.mean(observation['test_setting']['goal']['y']), 
                observation['test_setting']['goal']['heading'][0]]
            start_point = [observation['vehicle_info']['ego']['x'], observation['vehicle_info']['ego']['y'], observation['vehicle_info']['ego']['yaw_rad']]
            in_astar_path = hybrid_a_star_planning(start_point, goal, collision_lookup, observation, 1, 5,True)
            if not in_astar_path == None:
                astar_path = in_astar_path
            else:
                out_astar_path = hybrid_a_star_planning(start_point, goal, collision_lookup, observation, 1, 5,False)
                astar_path = out_astar_path
            path_planned = []
            spd_planned = []
            if astar_path is not None:
                print("len(astar_path):",len(astar_path.xlist))
                keypoint_ind = -1
                for j in range(0,len(astar_path.xlist)-1):
                    path_planned.append([astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                    if astar_path.directionlist[j] == True and astar_path.directionlist[j+1] == False:
                        print("keypoint:",[astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                        keypoint_ind = j
                print("keypoint_ind:",keypoint_ind)
                print("len(path):",len(path_planned))
                path_planned[keypoint_ind][-1]=2
                # -1代表没有找到人字尖
                if keypoint_ind == -1:
                    for k in range(len(path_planned)):
                        # print("k=",k)
                        if k < 10:
                            spd_planned.append(target_speed / 10 * k)
                        elif k >= 10 and k <= len(path_planned)-20:
                            spd_planned.append(target_speed)
                        elif k >= len(path_planned)-20:
                            spd_planned.append(-target_speed_backward / 20 * (len(path_planned)-k-1))
                else:
                    for k in range(len(path_planned)):
                        # print("k=",k)
                        if k < 10:
                            spd_planned.append(target_speed / 10 * k)
                        elif k >= 10 and k <= keypoint_ind-20:
                            spd_planned.append(target_speed)
                        elif (k > keypoint_ind-20) and (k <= keypoint_ind):
                            spd_planned.append(target_speed/ 20 * (keypoint_ind-k))
                        elif k > keypoint_ind and k <= keypoint_ind+20:
                            spd_planned.append(-target_speed_backward / 40 * (k-keypoint_ind+20))
                        elif k > keypoint_ind+20 and k < len(path_planned)-20:
                            spd_planned.append(-target_speed_backward)
                        elif k >= len(path_planned)-20:
                            spd_planned.append(-target_speed_backward / 20 * (len(path_planned)-k-1))
                for i in range(1,len(path_planned)):
                    path_planned[i].append(self.calculate_curvature(path_planned[i-1][0],path_planned[i-1][1],path_planned[i-1][2],
                                                            path_planned[i][0],path_planned[i][1],path_planned[i][2]))
                path_planned = path_planned[1:]
                spd_planned = spd_planned[1:]
                dir_current_file = os.path.dirname(__file__)  # 'algorithm\planner'
                filename = os.path.abspath(os.path.join(dir_current_file, 'temp_hybrid_astar_path.csv'))
                HA_path_headers = ['x', 'y', 'yaw', 'direction', 'curvature']
                with open(filename, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(HA_path_headers)
                    writer.writerows(path_planned)
                print(f"shovel场景混合A*全局路径数据已保存到 {filename}")
            else:
                return [],[]
        else:
            raise ValueError(f"未知的场景类型: {scene_type}")

        return path_planned, spd_planned

    def get_best_matching_path_token_from_polygon(self, veh_pose: Tuple[float, float, float], polygon_token: str, observation) -> str:
        """根据veh_pose(x,y,yaw)车辆定位位姿,从polygon_token所属的 link_referencepath_tokens 中匹配最佳参考路径.

        方法：
        1) 匹配2条path最近点;
        2）获取最佳path;

        Args:
            veh_pose (Tuple[float,float,float]):车辆的位姿.
            polygon_token (str):指定的polygon token.

        Returns:
            str:最佳匹配的 path_token,id_path.
        """
        semantic_map = observation["hdmaps_info"]["tgsc_map"]
        if not polygon_token.startswith("polygon-"):
            raise ValueError(f"Invalid polygon_token:{polygon_token}")

        id_polygon = int(polygon_token.split("-")[1])
        if id_polygon > len(semantic_map.polygon):
            raise IndexError(f"Polygon ID {id_polygon} out of bounds.请检查.")
        if semantic_map.polygon[id_polygon]["type"] == "intersection":
            raise IndexError(f"##log## Polygon ID = {id_polygon},目前未处理自车初始位置在交叉口区域的寻路逻辑.")

        link_referencepath_tokens = semantic_map.polygon[id_polygon]["link_referencepath_tokens"]

        candidate_paths = PriorityQueue()
        for _, path_token in enumerate(link_referencepath_tokens):
            id_path = int(path_token.split("-")[1])
            if semantic_map.reference_path[id_path]["type"] == "base_path":
                # 匹配最近点
                waypoint = self.find_nearest_waypoint(
                    waypoints=np.array(semantic_map.reference_path[id_path]["waypoints"]), veh_pose=veh_pose, downsampling_rate=5
                )
                yaw_diff = self.calc_yaw_diff_two_waypoints(waypoint1=(waypoint[0], waypoint[1], waypoint[2]), waypoint2=veh_pose)
                path_info = {"path_token": path_token, "id_path": id_path, "waypoint": waypoint, "yaw_diff": abs(yaw_diff)}
                candidate_paths.put((path_info["yaw_diff"], path_info))  # yaw_diff由小到大排序

        if candidate_paths.empty():
            raise ValueError(f"##log## Polygon ID = {id_polygon},所属路径均为connector_path,有问题.")
        # 得到同向最佳path的 token,id
        best_path_info = candidate_paths.get()  # 自动返回优先级最高的元素（优先级数值最小的元素）并从队列中移除它。

        return best_path_info[1]["path_token"], best_path_info[1]["id_path"]

    def find_nearest_waypoint(self, waypoints: np.array, downsampling_rate: int = 5, veh_pose: Tuple[float, float, float] = None):
        waypoints_downsampling = np.array(waypoints[::downsampling_rate])  # downsampling_rate,每5个路径点抽取一个点
        distances = np.sqrt((waypoints_downsampling[:, 0] - veh_pose[0]) ** 2 + (waypoints_downsampling[:, 1] - veh_pose[1]) ** 2)
        id_nearest = np.argmin(distances)
        return waypoints_downsampling[id_nearest]

    def calc_yaw_diff_two_waypoints(self, waypoint1: Tuple[float, float, float], waypoint2: Tuple[float, float, float]):
        """计算两个路径点之间的夹角,结果在[-pi,pi]范围内,"""
        angle1 = waypoint1[2]
        angle2 = waypoint2[2]
        yaw_diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
        return yaw_diff

    def get_ego_reference_path_to_goal(self, observation):
        """全局路径规划器.
        获取HD Map中ego车到达目标区域的参考路径.
        注1：该参考路径仅表示了对于道路通行规则的指示。
            数据处理来源:1)在地图中手动标记dubinspose;2)生成dubins curve;3)离散拼接.
        注2:显然该参考路径存在以下缺陷--
            1)在实际道路中使用该参考路径跟车行驶时不符合曲率约束;2)onsite_mine仿真中存在与边界发生碰撞的风险.
        注3:onsite_mine自动驾驶算法设计鼓励参与者设计符合道路场景及被控车辆特性的实时轨迹规划算法.

        输入:observation——环境信息;
        输出:ego车到达目标区域的参考路径(拼接后).
        """
        # if 0:  # 仅适用于 会车场景 jiangtong-9-1-1,不在比赛场景中
        #     ego_connected_path_tokens = ["path-107", "path-93"]
        #     return self.get_connected_waypoints_from_multi_path(observation, ego_connected_path_tokens)
        #################更新参数#################
        self._ego_x = observation["vehicle_info"]["ego"]["x"]
        self._ego_y = observation["vehicle_info"]["ego"]["y"]
        self._ego_v = observation["vehicle_info"]["ego"]["v_mps"]
        self._ego_yaw = observation["vehicle_info"]["ego"]["yaw_rad"]

        #################定位主车和目标点所在几何#################
        ego_polygon_token = observation["hdmaps_info"]["tgsc_map"].get_polygon_token_using_node(self._ego_x, self._ego_y)
        # ego_polygon_id = int(ego_polygon_token.split("-")[1])

        #################获取目标车最匹配的path-token #################
        # 获取当前所处的多边形区域中的最合适的参考路径
        ego_path_token, ego_path_id = self.get_best_matching_path_token_from_polygon(
            (self._ego_x, self._ego_y, self._ego_yaw), ego_polygon_token, observation
        )

        # 广度优先搜索,一般搜索不超过3层；
        # todo 可以使用标准的树搜索方式重写.
        path_connect_tree = {"layer_1": ego_path_token, "layer_2": {}}
        second_layer_path_tokens = observation["hdmaps_info"]["tgsc_map"].reference_path[ego_path_id]["outgoing_tokens"]

        # 第二层
        for _, token_2 in enumerate(second_layer_path_tokens):  # path_token_2
            id_2 = int(token_2.split("-")[1])  # path_id_2
            if token_2 not in path_connect_tree["layer_2"]:
                path_connect_tree["layer_2"][token_2] = {
                    "flag_inside_goal_area": False,
                    "layer_3": {},
                }
            ref_path = np.array(observation["hdmaps_info"]["tgsc_map"].reference_path[id_2]["waypoints"])
            # 搜索终止条件:参考路径的waypoints有点waypoint在目标区域内部
            flag_inside_goal_area = self.has_waypoint_inside_goal_area(
                ref_path,
                observation["test_setting"]["goal"]["x"],
                observation["test_setting"]["goal"]["y"],
            )
            if flag_inside_goal_area:
                path_connect_tree["layer_2"][token_2]["flag_inside_goal_area"] = True
                ego_connected_path_tokens = [path_connect_tree["layer_1"], token_2]
                return self.get_connected_waypoints_from_multi_path(observation, ego_connected_path_tokens)
            else:
                path_connect_tree["layer_2"][token_2]["flag_inside_goal_area"] = False
                outgoing_tokens = observation["hdmaps_info"]["tgsc_map"].reference_path[id_2]["outgoing_tokens"]
                for _, token in enumerate(outgoing_tokens):
                    if token not in path_connect_tree["layer_2"][token_2]["layer_3"]:
                        path_connect_tree["layer_2"][token_2]["layer_3"][token] = {}

        # 第三层
        for _, token_2 in enumerate(second_layer_path_tokens):
            thrid_layer_path_tokens = path_connect_tree["layer_2"][token_2]["layer_3"]
            for _, token_3 in enumerate(thrid_layer_path_tokens):
                id_3 = int(token_3.split("-")[1])
                if not path_connect_tree["layer_2"][token_2]["layer_3"][token_3]:  # 空
                    path_connect_tree["layer_2"][token_2]["layer_3"][token_3] = {
                        "flag_inside_goal_area": False,
                        "layer_4": {},
                    }
                ref_path = np.array(observation["hdmaps_info"]["tgsc_map"].reference_path[id_3]["waypoints"])
                flag_inside_goal_area = self.has_waypoint_inside_goal_area(
                    ref_path,
                    observation["test_setting"]["goal"]["x"],
                    observation["test_setting"]["goal"]["y"],
                )
                if flag_inside_goal_area:
                    path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["flag_inside_goal_area"] = True
                    ego_connected_path_tokens = [
                        path_connect_tree["layer_1"],
                        token_2,
                        token_3,
                    ]
                    return self.get_connected_waypoints_from_multi_path(observation, ego_connected_path_tokens)

                else:
                    path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["flag_inside_goal_area"] = False
                    outgoing_tokens = observation["hdmaps_info"]["tgsc_map"].reference_path[id_3]["outgoing_tokens"]
                    for _, token in enumerate(outgoing_tokens):
                        if token not in path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["layer_4"]:
                            # path_connect_tree['layer_2']['path-7']['layer_3']['path-70']['layer_4']
                            path_connect_tree["layer_2"][token_2]["layer_3"][token_3]["layer_4"][token] = {}

    @staticmethod
    def has_waypoint_inside_goal_area(
        ref_path_waypoints: np.array = None,
        goal_area_x: List = None,
        goal_area_y: List = None,
    ) -> bool:
        """计算参考路径的waypoints 是否 有点waypoint在目标区域内部.

        Args:
            ref_path_waypoints (np.array, optional): 参考路径的waypoints. Defaults to None.
            goal_area_x (List, optional): 目标区域x坐标列表. Defaults to None.
            goal_area_y (List, optional): 目标区域y坐标列表. Defaults to None.

        Returns:
            bool: 参考路径的waypoints 是否 有点waypoint在目标区域内部
        """
        if ref_path_waypoints is None or goal_area_x is None or goal_area_y is None:
            return False

        # Create Polygon object representing the goal area
        goal_area_coords = list(zip(goal_area_x, goal_area_y))
        goal_area_polygon = Polygon(goal_area_coords)

        # Check each waypoint
        for waypoint in ref_path_waypoints:
            x, y = waypoint[0], waypoint[1]
            if goal_area_polygon.contains(Point(x, y)):
                return True
        return False

    def get_connected_waypoints_from_multi_path(self, observation, connected_path_tokens: List = None):
        """获得多条路径拼接后的waypoints. waypoints(x,y,yaw,heigh,slope).

        Args:
            observation (_type_): 环境信息.
            connected_path_tokens (List, optional): _description_. Defaults to None.

        Returns:
            List: 多条路径拼接后的waypoints(x,y)
        """

        connected_waypoints = []

        connected_waypoints = [
            point
            for token_path in connected_path_tokens
            for point in observation["hdmaps_info"]["tgsc_map"].reference_path[int(token_path.split("-")[1])]["waypoints"]
        ]

        return connected_waypoints

    def get_connected_waypoints_from_multi_path_array(self, observation, connected_path_tokens: List = None):
        """获得多条路径拼接后的waypoints.使用np.array

        Args:
            connected_path_tokens (List, optional): _description_. Defaults to None.
        """
        # Initialize connected_waypoints as None
        connected_waypoints = None
        for token_path in connected_path_tokens:
            id_path = int(token_path.split("-")[1])

            # Get temp_waypoints from observation
            temp_waypoints = np.array(observation["hdmaps_info"]["tgsc_map"].reference_path[id_path]["waypoints"])

            # Check if connected_waypoints is None, if so assign temp_waypoints to it
            # otherwise concatenate temp_waypoints to connected_waypoints
            if connected_waypoints is None:
                connected_waypoints = temp_waypoints
            else:
                connected_waypoints = np.concatenate((connected_waypoints, temp_waypoints), axis=0)

        return connected_waypoints

