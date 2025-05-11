import math
import numpy as np
import scipy.linalg as la

# State 对象表示自车的状态，位置x、y，以及横摆角yaw、速度v
class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class LQRController:

    def __init__(self):
        # Basic parameters
        self.dt = 0.1  # time tick[s]
        self.L = 5.6  # Wheel base of the vehicle [m]
        self.max_steer = 1   # maximum steering angle[rad]
        
        # 调整控制权重
        self.lqr_Q = np.eye(5)
        self.lqr_Q[0][0] = 1.0   # 横向偏差权重
        self.lqr_Q[1][1] = 0.5   # 降低横向偏差变化率权重
        self.lqr_Q[2][2] = 2.0   # 航向偏差权重
        self.lqr_Q[3][3] = 2.0   # 降低航向偏差变化率权重
        self.lqr_Q[4][4] = 5.0   # 显著增加速度误差权重

        # 降低控制成本
        self.lqr_R = np.eye(2)
        self.lqr_R[0][0] = 0.5    # 转向控制成本
        self.lqr_R[1][1] = 1.0  # 加速度控制成本

        # 速度参数调整
        self.min_speed = 5.0        # 最小速度要求
        self.target_speed = 12.0    # 设置合理的目标速度
        self.speed_recovery_rate = 3.0

        # Initialize other parameters
        self.pe, self.pth_e = 0.0, 0.0
        self.keyind = 0 
        self.isback = False  
        # 添加曲率相关参数
        self.min_curve_speed = 3.0  # 弯道最小速度
        self.max_lateral_accel = 2.0  # 最大横向加速度
        self.curvature_factor = 0.8  # 曲率影响因子  
    
    def calc_curvature_based_speed(self, curvature):
        """计算基于曲率的安全速度"""
        if abs(curvature) < 1e-6:  # 直道
            return self.target_speed
            
        # 使用横向加速度公式: v = sqrt(a_lat / k)
        # 其中 a_lat 是允许的最大横向加速度，k 是曲率
        safe_speed = math.sqrt(self.max_lateral_accel / (abs(curvature) * self.curvature_factor))
        
        # 限制在合理范围内
        safe_speed = min(safe_speed, self.target_speed)
        safe_speed = max(safe_speed, self.min_curve_speed)
        
        return safe_speed

    def pi_2_pi(self,angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # 实现离散Riccati equation 的求解方法
    def solve_dare(self,A, B, Q, R):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        x = Q
        x_next = Q
        max_iter = 150
        eps = 0.01

        for i in range(max_iter):
            x_next = A.T @ x @ A - A.T @ x @ B @ \
                    la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
            if (abs(x_next - x)).max() < eps:
                break
            x = x_next

        return x_next

    # 返回值K 即为LQR 问题求解方法中系数K的解
    def dlqr(self,A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        X = self.solve_dare(A, B, Q, R)

        # compute the LQR gain
        K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

        eig_result = la.eig(A - B @ K)

        return K, X, eig_result[0]

    # 计算距离自车当前位置最近的参考点
    def calc_nearest_index(self,state, cx, cy, cyaw):
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind)

        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y
        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def process(self, vehicle_info, path_waypoints, speed_profile, mode, slope=0):
        # if self.keyind != 0 and self.isback:
        #     path_waypoints = path_waypoints[self.keyind+1:]
        #     speed_profile = speed_profile[self.keyind+1:]
        cx,cy,cyaw,ck,cgear=[],[],[],[],[]
        for i in path_waypoints:
            cx.append(i[0])
            cy.append(i[1])
            cyaw.append(i[2])
            cgear.append(i[3])
            ck.append(i[4])
        _ego_v = abs(vehicle_info['ego']['v_mps'])  # 使用绝对值
        _ego_x = vehicle_info['ego']['x']
        _ego_y = vehicle_info['ego']['y']
        _ego_yaw = vehicle_info['ego']['yaw_rad']
        state = State(_ego_x, _ego_y, _ego_yaw, _ego_v)

        ind, e = self.calc_nearest_index(state, cx, cy, cyaw)
        
        if cgear[ind]==True:
            gear = 1
        else:
            gear = 3
        # if (cgear[ind]==2 or cgear[ind]==False) and not self.isback:
        #     for i in range(len(path_waypoints)):
        #         if path_waypoints[i][-1]==2:
        #             self.keyind = i
        #             break
        #     self.isback = True
        sp = speed_profile
        tv = max(sp[ind], self.min_speed)  # 确保目标速度不低于最小速度
        k = ck[ind]
        v_state = max(state.v, self.min_speed)  # 确保速度不会太低
        th_e = self.pi_2_pi(state.yaw - cyaw[ind])

        # 构建LQR表达式，X(k+1) = A * X(k) + B * u(k), 使用Riccati equation 求解LQR问题
    #     dt表示采样周期，v表示当前自车的速度
    #     A = [1.0, dt, 0.0, 0.0, 0.0
    #          0.0, 0.0, v, 0.0, 0.0]
    #          0.0, 0.0, 1.0, dt, 0.0]
    #          0.0, 0.0, 0.0, 0.0, 0.0]
    #          0.0, 0.0, 0.0, 0.0, 1.0]
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = v_state
        A[2, 2] = 1.0
        A[2, 3] = self.dt
        A[4, 4] = 1.0

        # 构建B矩阵，L是自车的轴距
        # B = [0.0, 0.0
        #     0.0, 0.0
        #     0.0, 0.0
        #     v/L, 0.0
        #     0.0, dt]
        B = np.zeros((5, 2))
        B[3, 0] = v_state / self.L
        B[4, 1] = self.dt

        K, _, _ = self.dlqr(A, B, self.lqr_Q, self.lqr_R)

        # 状态向量构建 - 只处理一次速度误差
        X = np.zeros((5, 1))
        X[0, 0] = e
        X[1, 0] = (e - self.pe) / self.dt
        X[2, 0] = th_e
        X[3, 0] = (th_e - self.pth_e) / self.dt
        speed_error = v_state - tv
        X[4, 0] = speed_error

        # 计算基础控制输入
        ustar = -K @ X
        
        # 转向控制保持不变
        ff = math.atan2(self.L * k, 1)
        fb = self.pi_2_pi(ustar[0, 0])
        delta = ff + fb
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        # 简化的速度控制逻辑
        base_accel = ustar[1, 0]
        
        # 正向行驶时的速度控制
        if gear == 1:
            if v_state < self.min_speed:
                accel = 5.0  # 强制加速
            elif speed_error < 0:  # 速度低于目标
                accel = max(base_accel * 2.0, 3.0)  # 更强的加速
            else:
                accel = base_accel
        # 倒车时的速度控制
        else:
            if v_state < self.min_speed:
                accel = -5.0  # 强制倒车加速
            elif speed_error < 0:
                accel = min(base_accel * 2.0, -3.0)  # 更强的倒车加速
            else:
                accel = base_accel

        # 计算基于曲率的目标速度
        curvature_based_speed = self.calc_curvature_based_speed(k)
        tv = min(sp[ind], curvature_based_speed)
        
        # 添加预瞻减速
        look_ahead = 5  # 预瞻点数
        for i in range(ind, min(ind + look_ahead, len(ck))):
            future_safe_speed = self.calc_curvature_based_speed(ck[i])
            tv = min(tv, future_safe_speed)
        
        # 速度恢复逻辑
        if state.v < self.min_speed:
            recovery_boost = 2.0
        else:
            recovery_boost = 1.0   

        # 5. 调整加速度限制
        max_accel = 30.0  # 增加最大加速度限制
        # 修改加速度控制
        if gear == 1:
            if v_state < tv * 0.8:  # 速度过低时
                accel = min(5.0 * recovery_boost, max_accel)
            elif speed_error < 0:  # 速度低于目标
                accel = min(base_accel * 1.5, 3.0) * recovery_boost
            else:
                accel = base_accel
            
        # 6. 添加坡度补偿
        slope_compensation = (0.02) * (-9.8) * np.sin(slope)
        if gear == 1:
            accel += abs(slope_compensation)  # 前进补偿
        else:
            accel -= abs(slope_compensation)  # 倒车补偿
        
        # 更新历史状态
        self.pe = e
        self.pth_e = th_e
        
        return (accel, delta, gear)

    def adjust_speed_profile(self, speed_profile, current_speed):
        """Adjust speed profile to maintain momentum"""
        if current_speed < self.target_speed * 0.7:  # If speed drops below 70% of target
            return [max(s, self.target_speed) for s in speed_profile]
        return speed_profile
