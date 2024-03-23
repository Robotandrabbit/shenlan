import numpy as np
import math
from typing import List, Type, Optional, Tuple
from nuplan.planning.simulation.planner.project2.frame_transform import get_match_point, cal_project_point
from shapely.geometry import Point, LineString
from scipy.interpolate import interp1d

class DpDecider:

    def __init__(self, 
                 obs_trajectory: List[List[float]],
                 obs_radius: List[float],
                 path_idx2s: List[float], 
                 path_x: List[float], 
                 path_y: List[float], 
                 path_heading: List[float], 
                 path_kappa: List[float],
                 total_time: float, 
                 step: float,
                 max_v:float,
                 ego_half_width: float,
                 ego_length: float,
                 ego_v: float,
                 max_acc: float,
                 max_dec: float) -> None:
        self._obs_trajectory = obs_trajectory
        self._obs_radius = obs_radius
        self._path_idx2s = path_idx2s
        self._path_x = path_x
        self._path_y = path_y
        self._path_heading = path_heading
        self._path_kappa = path_kappa
        self._total_t = total_time
        self._delta_t = step
        self._ego_half_width = ego_half_width
        self._ego_length = ego_length
        self._max_v = max_v
        self._max_acc = max_acc
        self._max_dec = max_dec
        self._w_cost_ref_speed = 40
        self._reference_speed = max_v
        self._w_cost_accel = 10
        self._w_cost_obs = 200
        self._plan_start_s_dot = ego_v
        self._obs_ts: List[LineString] = []
        self._obs_interp_ts: List[interp1d] = []



    def dynamic_programming(self) ->Tuple[float]:
        # 根据障碍物轨迹和路径规划结果，将障碍物轨迹映射到ST图
        self._obs_ts = []

        import matplotlib.pyplot as plt
        # 绘制曲线
        plt.clf()

        for trajectory in self._obs_trajectory:
            # 每个agent的trajectory
            points_ts = []
            for state in trajectory:
                t = state[0]
                x = state[1]
                y = state[2]
                # 计算每个点在path上的s和l
                match_point_index_set = get_match_point([x], [y], self._path_x, self._path_y)
                proj_x_set, proj_y_set, proj_heading_set, _, proj_s_set = cal_project_point(\
                                    [x], [y], match_point_index_set, self._path_x, self._path_y, self._path_heading, self._path_kappa, self._path_idx2s)
                s = proj_s_set[0]
                n_r = np.array([-math.sin(proj_heading_set[0]), math.cos(proj_heading_set[0])])
                r_h = np.array([x, y])
                r_r = np.array([proj_x_set[0], proj_y_set[0]])
                l = np.dot((r_h - r_r), n_r)
                if s >= 0 and s <= self._path_idx2s[-1] and abs(l) <= self._ego_half_width:
                    points_ts.append((t, s))
                # if s <= self._path_idx2s[-1]: # and abs(l) <= self._ego_half_width:
                #     points_ts.append((t, s))
            if len(points_ts) > 1:
                line_ts = LineString(points_ts)
                interp_ts = interp1d([ts[0] for ts in points_ts], [ts[1] for ts in points_ts])
                self._obs_ts.append(line_ts)
                self._obs_interp_ts.append(interp_ts)

            t_points = [point[0] for point in points_ts]
            s_points = [point[1] for point in points_ts]
            plt.plot(t_points, s_points, color='red')  # 连接成连续曲线并绘制在曲线图上，使用红色线条

        # S T撒点，T撒点要和后续的速度规划保持一致，S的最大值也和后续的速度规划保持一致(max_v * total_time)
        t_list = np.arange(self._delta_t, self._total_t, self._delta_t) # t = 0不必搜索
        t_list = np.append(t_list, self._total_t)
        max_s = self._max_v * self._total_t
        delta_s = 2
        s_list = np.arange(0, max_s, delta_s)
        s_list = np.append(s_list, max_s)
        # 稀疏采样可以加快速度，但也容易导致找不到符合约束的dp_s
        # third = int(max_s / 3)
        # s_list1 = np.arange(0, third, delta_s)
        # s_list2 = np.arange(third, max_s, 3 * delta_s)
        # s_list = np.concatenate((s_list1, s_list2))
        # s_list = np.append(s_list, max_s)

        # 保存dp过程的数据
        dp_st_cost =  [[math.inf] * len(t_list) for _ in range(len(s_list))] # [[t]]
        dp_st_s_dot =  [[0] * len(t_list) for _ in range(len(s_list))] # [[t]] 表示从起点开始到(i,j)点的最优路径的末速度
        dp_st_node =  [[0] * len(t_list) for _ in range(len(s_list))]

        # 计算从dp起点到第一列的cost
        for i in range(len(s_list)):
            dp_st_cost[i][0] = self._CalcDpCost(-1, -1, i, 0, s_list, t_list,dp_st_s_dot)
            # 计算第一列所有节点的的s_dot，并存储到dp_st_s_dot中 第一列的前一个节点只有起点
            dp_st_s_dot[i][0] = (s_list[i] - self._ego_length/2) / t_list[0]

        # 动态规划主程序
        for i in np.arange(1, len(t_list)):
            # i 为列循环
            for j in range(len(s_list)):
                # j 为行循环
                # 当前行为 j 列为 i
                cur_row = j
                cur_col = i
                # 遍历前一列
                for k in range(len(s_list)):
                    pre_row = k
                    pre_col = i - 1
                    # 计算边的代价 其中起点为pre_row,pre_col 终点为cur_row cur_col
                    cost_temp = self._CalcDpCost(pre_row, pre_col, cur_row, cur_col, s_list, t_list, dp_st_s_dot)
                    if cost_temp + dp_st_cost[pre_row][pre_col] < dp_st_cost[cur_row][cur_col]:
                        dp_st_cost[cur_row][cur_col] = cost_temp + dp_st_cost[pre_row][pre_col]
                        # 计算最优的s_dot
                        s_start, t_start = self._CalcSTCoordinate(pre_row, pre_col, s_list, t_list)
                        s_end, t_end = self._CalcSTCoordinate(cur_row, cur_col, s_list, t_list)
                        dp_st_s_dot[cur_row][cur_col] = (s_end - s_start) / (t_end - t_start)
                        # 将最短路径的前一个节点的行号记录下来
                        dp_st_node[cur_row][cur_col] = pre_row
        
        # 输出dp结果
        # 输出初始化
        dp_speed_s = [-1] * len(t_list)
        dp_speed_t = [-1] * len(t_list)
        # 找到dp_node_cost 上边界和右边界代价最小的节点
        min_cost = math.inf
        min_row = math.inf
        min_col = len(t_list) -1
        for i in range(len(s_list)):
            # 遍历右边界
            if dp_st_cost[i][min_col] <= min_cost:
                min_cost = dp_st_cost[i][min_col]
                min_row = i

        # for j in range(len(t_list)):
        #     # 遍历上边界
        #     if dp_st_cost[-1][j] <= min_cost:
        #         min_cost = dp_st_cost[-1][j]
        #         min_row = -1
        #         min_col = j
        # 先把终点的ST输出出来
        s, t = self._CalcSTCoordinate(min_row, min_col, s_list, t_list)
        dp_speed_s[min_col] = s
        dp_speed_t[min_col] = t
        # 反向回溯
        while min_col != 0:
            pre_row = dp_st_node[min_row][min_col]
            pre_col = min_col - 1
            s, t = self._CalcSTCoordinate(pre_row, pre_col, s_list, t_list)
            dp_speed_s[pre_col] = s
            dp_speed_t[pre_col] = t
            min_row = pre_row
            min_col = pre_col
        
        s_lb = [0] * len(t_list)
        s_ub = [max_s] * len(t_list)
        # 根据dp结果和_obs_ts，计算s_lb, s_ub
        for i in range(len(self._obs_ts)):
            obs_ts = self._obs_ts[i]
            obs_interp_ts = self._obs_interp_ts[i]
            bounds = obs_ts.bounds
            obs_t_min = bounds[0]
            obs_t_max = bounds[2]
            for j in range(len(t_list)):
                t = t_list[j]
                if t < obs_t_min or t > obs_t_max:
                    continue
                s = dp_speed_s[j]
                # 计算obs在当前t的s，判断决策，输出边界
                obs_s = obs_interp_ts(t)
                if s < obs_s: # 避让改上界
                    s_ub[j] = min(s_ub[j], max(0, obs_s - self._ego_length - self._obs_radius[i]))
                    if s_ub[j] < s_lb[j] + 0.5:
                        s_ub[j] = s_lb[j] + 0.5
                    # s_ub[j] = min(s_ub[j], max(0, obs_s))
                else: # 超车改下界
                    s_lb[j] = max(s_lb[j], min(max_s, obs_s + self._ego_length + self._obs_radius[i]))
                    # s_lb[j] = max(s_lb[j], min(max_s, obs_s))
                    if s_lb[j] > s_ub[j] - 0.5:
                        s_lb[j] = s_ub[j] - 0.5


        plt.plot(dp_speed_t, dp_speed_s)

        # 添加标题和标签
        plt.title('S vs Time')
        plt.xlabel('T')
        plt.ylabel('S')
        plt.ylim((-5, max_s))

        # 显示网格
        plt.grid(True)

        # 显示图形
        # plt.show()
        import os
        from datetime import datetime
        # 创建一个文件夹用于保存图片
        if not os.path.exists("images"):
            os.mkdir("images")
        # 获取当前时间并格式化
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

        # 设置文件名前缀和扩展名
        file_name_prefix = "images/figure"
        file_extension = ".png"

        # 拼接完整的文件名
        file_name = f"{file_name_prefix}_{formatted_datetime}{file_extension}"

        plt.savefig(file_name)
        print('s_lb: ', s_lb)
        print('s_ub: ', s_ub)

        dp_s_out = [s - self._ego_length/2 for s in dp_speed_s]
        return s_lb, s_ub, dp_s_out, dp_st_s_dot


    def _CalcDpCost(self, row_start, col_start, row_end, col_end, s_list, t_list, dp_st_s_dot):
        """
        该函数将计算链接两个节点之间边的代价
        :param 边的起点的行列号row_start,col_start 边的终点行列号row_end,col_end
        :param s_list,t_list
        :dp_st_s_dot 用于计算加速度
        :return 边的代价
        """
        # 首先计算终点的st坐标
        s_end, t_end = self._CalcSTCoordinate(row_end, col_end, s_list, t_list)
        # 规定起点的行列号为-1 
        if row_start == -1:
            # 边的起点为dp的起点
            s_start = self._ego_length/2
            t_start = 0
            s_dot_start = self._plan_start_s_dot
        else:
            # 边的起点不是dp的起点
            s_start, t_start = self._CalcSTCoordinate(row_start, col_start, s_list, t_list)
            s_dot_start = dp_st_s_dot[row_start][col_start]
        cur_s_dot = (s_end - s_start) / (t_end - t_start)
        cur_s_dot2 = (cur_s_dot - s_dot_start)/(t_end - t_start)
        # 计算推荐速度代价
        # cost_ref_speed = self._w_cost_ref_speed * (cur_s_dot - self._reference_speed) ** 2
        if cur_s_dot <= self._reference_speed and cur_s_dot >= 0:
            cost_ref_speed = self._w_cost_ref_speed * (cur_s_dot - self._reference_speed)**2
        elif cur_s_dot > self._reference_speed:
            cost_ref_speed = 100 * self._w_cost_ref_speed * (cur_s_dot - self._reference_speed)**2 # 尽量让dp有解
        else:
            cost_ref_speed = math.inf
        # 计算加速度代价，这里注意，加速度不能超过车辆动力学上下限
        # cost_accel = self._w_cost_accel * cur_s_dot2 ** 2
        if cur_s_dot2 <= self._max_acc and cur_s_dot2 >= self._max_dec:
            cost_accel = self._w_cost_accel * cur_s_dot2**2
        else:
            # 超过车辆动力学限制，代价会增大很多倍
            cost_accel = 1000 * self._w_cost_accel * cur_s_dot2**2
            # cost_accel = math.inf
        cost_obs = self._CalcObsCost(s_start,t_start,s_end,t_end)
        cost = cost_obs + cost_accel + cost_ref_speed
        return cost

    def _CalcObsCost(self, s_start, t_start, s_end, t_end):
        """
        该函数将计算边的障碍物代价
        :param 边的起点终点s_start,t_start,s_end,t_end
        :return 边的障碍物代价obs_cost
        """
        # 输出初始化
        obs_cost = 0
        # 边的采样点的个数
        n = 4
        # 采样时间间隔
        dt = (t_end - t_start)/(n - 1)
        # 边的斜率
        k = (s_end - s_start)/(t_end - t_start)
        for i in range(n+1):
            # 计算采样点的坐标
            t = t_start + i * dt
            s = s_start + k * i * dt
            point_ts = Point(t, s)
            # 遍历所有障碍物
            for j in range(len(self._obs_ts)):
                obs_ts = self._obs_ts[j]
                bounds = obs_ts.bounds
                obs_t_min = bounds[0]
                obs_t_max = bounds[2]
                if t < obs_t_min or t > obs_t_max:
                    continue
                obs_interp_ts = self._obs_interp_ts[j]
                obs_s = obs_interp_ts(t)
                dis = s - obs_s
                obs_cost = obs_cost + self._CalcCollisionCost(self._w_cost_obs, dis, j)
                # # 计算点到st折线的最短距离
                # min_dis = point_ts.distance(self._obs_ts[j])
                # obs_cost = obs_cost + self._CalcCollisionCost(self._w_cost_obs, min_dis, j)
        return obs_cost
    
    def _CalcCollisionCost(self, w_cost_obs, min_dis, obs_idx):
        collision_cost = 0
        obs_radius = self._obs_radius[obs_idx]
        buffer = obs_radius + self._ego_length/2
        if abs(min_dis) < buffer:
            # collision_cost = w_cost_obs
            collision_cost = math.inf
        elif abs(min_dis) > buffer and abs(min_dis) < buffer*3:
            collision_cost = w_cost_obs**((buffer - abs(min_dis))/buffer + 2)
        else:
            collision_cost = 0
        return collision_cost
    
    def _CalcSTCoordinate(self, row, col, s_list, t_list):
        s = s_list[row]
        t = t_list[col]
        return s, t







        
        