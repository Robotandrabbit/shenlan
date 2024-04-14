import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.quintic_polynominal import QuinticPolynomial
from nuplan.planning.simulation.planner.project2.quartic_polynominal import QuarticPolynominal
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.planning.simulation.planner.project2.frame_transform import cartesian2frenet, local2global_vector

WEIGHT_PROGRESS = 50.0
WEIGHT_OFFSET = 5.0
WEIGHT_SMOOTH = 5.0

MAXIMUM_JERK = 2.0
MINIMUM_PROGRESS = 15.0
MAXIMUM_OFFSET = 4.0

MAXIMUM_DECELERATION = 5.0
MAXIMUM_ACCELERATION = 5.0
class LatticePathPlanning:
    def __init__(self,
                 ego_state: EgoState,
                 reference_line_provider: ReferenceLineProvider,
                 horizon_time: TimePoint,
                 sampling_time: TimePoint) -> None:
        self.ego_state = ego_state
        self.reference_line_provider = reference_line_provider
        self.horizon_time = horizon_time
        self.sampling_time = sampling_time
    
    # 横向上：在 d-s 维度采样。相比较 d-t 采样，d-s 可以明确自车局部的终点，也可以根据 s 获取车道宽度
    # 约束采样的范围
    def sample_lateral_end_state_ds(self):
        end_d_candidates = np.arange(-1.0, 1.2, 0.5)
        end_s_candidates = np.array([10.0, 20.0, 40.0])
        lb_set, rb_set = self.reference_line_provider.get_boundary(end_s_candidates)
        
        sampled_states = []
        for idx in range(len(end_s_candidates)):
            for d in end_d_candidates:
                # 检查超出道路边界
                if (d > lb_set[idx] or d < rb_set[idx]):
                    continue
                # 基于自车当前位置采样
                state = np.array([d, 0.0, 0.0, end_s_candidates[idx]])
                sampled_states.append(state)
        return sampled_states
    
    # 纵向上：在 v, t 维度采样
    def sample_lon_end_state(self, init_frenet_state, target_speed: float):
        end_states = []
        time_samples = []
        # 插入当前位置
        time_samples.append(0.01)
        delta_t = 1.0
        for i in np.arange(1, self.horizon_time.time_s + delta_t, delta_t):
            time_samples.append(i)
        
        for time in time_samples:
            v_upper = min(init_frenet_state[2][0] + MAXIMUM_DECELERATION * time, target_speed)
            v_lower = max(init_frenet_state[2][0] - MAXIMUM_ACCELERATION * time, 0.0)
            end_states.append([0.0, v_upper, 0.0, time])
            end_states.append([0.0, v_lower, 0.0, time])
            v_range = v_upper - v_lower
            num_of_mid_points = int(min(4, v_range / 1.0))
            if (num_of_mid_points > 0):
                velocity_seg = v_range / (num_of_mid_points + 1)
                for i in range(num_of_mid_points):
                    end_states.append([0.0, v_lower + velocity_seg * i, 0.0, time])
        
        return end_states
            

    def evaluate_trajectory(self, init_frenet_state, lon_trajectory:QuarticPolynominal, lat_trajectory:QuinticPolynomial) -> float:
        proress_item = 0.0
        lon_jerk_item = 0.0
        lat_jerk_item = 0.0
        offset_item = 0.0
        # 1. longitudinal progress cost
        progress = lon_trajectory.get_point(lon_trajectory.get_time()) - init_frenet_state[0][0]
        # progress = lat_trajectory.get_param()
        if (progress < MINIMUM_PROGRESS):
            proress_item += (MINIMUM_PROGRESS - progress) / MINIMUM_PROGRESS
        # longitudinal jerk
        for t in np.arange(0.0, lon_trajectory.get_time(), 0.2):
            lon_jerk_item += abs(lon_trajectory.get_third_derivative(t)) / MAXIMUM_JERK

        for t in np.arange(0.0, self.horizon_time.time_s, self.sampling_time.time_s):
            if (t > lon_trajectory.get_time()):
                break
            # 2. lateral smooth cost
            local_s = lon_trajectory.get_point(t) - init_frenet_state[0][0]
            lateral_jerk = lat_trajectory.get_third_derivative(local_s)
            lat_jerk_item += abs(lateral_jerk) / MAXIMUM_JERK
            
            # 3. offset cost
            lateral_offset = lat_trajectory.get_point(local_s)
            offset_item += abs(lateral_offset)/ MAXIMUM_OFFSET
            
            # TODO(wanghao): add more cost.
            # 4. collision cost:static agent.
            # 5. flicker cost
        print("progress, smooth, offset:", proress_item * WEIGHT_PROGRESS, (lat_jerk_item + lon_jerk_item) * WEIGHT_SMOOTH, offset_item * WEIGHT_OFFSET)
        return proress_item * WEIGHT_PROGRESS + \
               (lat_jerk_item + lon_jerk_item) * WEIGHT_SMOOTH + \
               offset_item * WEIGHT_OFFSET

    def is_valid_lon_trajectory(self, lon_trajectory:QuarticPolynominal) -> bool:
        t = 0.0
        while (t < lon_trajectory.get_time()):
            velocity = lon_trajectory.get_first_derivative(t)
            accleration = lon_trajectory.get_second_derivative(t)
            if (velocity > 10.0 and velocity < 0.0):
                return False
            
            if (accleration > 10.0 and accleration < -10.0):
                return False
            
            t += 0.1
        return True

    def get_optimal_trajectory(self, init_frenet_state, lat_trajectories, lon_trajectories) -> Tuple[QuinticPolynomial, QuarticPolynominal]:
        min_score = float('+inf')
        best_lon_trajectory = None
        best_lat_trajectory = None
        has_valid_traj = False
        print("lat_traj, lon_traj:", len(lat_trajectories), len(lon_trajectories))
        for lon_trajectory in lon_trajectories:
            # Check if lon_trajectory is valid.
            if (not self.is_valid_lon_trajectory(lon_trajectory)):
                continue
            for lat_trajectory in lat_trajectories:
                score = self.evaluate_trajectory(init_frenet_state, lon_trajectory, lat_trajectory)
                if (score < min_score):
                    min_score = score
                    best_lon_trajectory = lon_trajectory
                    best_lat_trajectory = lat_trajectory
                    has_valid_traj = True
        # Combine two 1d trajectories to one 2d trajectory
        if (not has_valid_traj):    
            print("Failed to sample trajectory")
        return best_lat_trajectory, best_lon_trajectory

    def get_init_cartesian_state(self):
        velocity_global_x, velocity_global_y = local2global_vector(self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                                                                   self.ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
                                                                   self.ego_state.car_footprint.oriented_box.center.heading)

        acce_global_x, acce_global_y = local2global_vector(self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                                                           self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
                                                           self.ego_state.car_footprint.oriented_box.center.heading)

        return np.array([self.ego_state.car_footprint.oriented_box.center.x,
                         self.ego_state.car_footprint.oriented_box.center.y,
                         velocity_global_x, velocity_global_y,
                         acce_global_x, acce_global_y])

    # 规划主函数
    def path_planning(self) -> tuple[float, float, float, float]:
        plt.figure()
        # 提取自车规划起始点笛卡尔坐标系下的状态
        # ego_state 中的位置基于全局坐标系，但是速度是自车坐标系下的，我们应该全部转换到全局系。
        init_cartesian_state = self.get_init_cartesian_state()
        # 构造参考线
        reference_line = np.array([self.reference_line_provider._x_of_reference_line,
                                   self.reference_line_provider._y_of_reference_line,
                                   self.reference_line_provider._heading_of_reference_line,
                                   self.reference_line_provider._kappa_of_reference_line,
                                   self.reference_line_provider._s_of_reference_line])
        '''
        # s = []
        # l = [0 for _ in range(len(reference_line[4]))] 
        # for sampled_s in reference_line[4]:
        #     s.append(sampled_s)
        # plt.plot(s, l, '-', color='yellow', linewidth=2)
        '''
        # 获取基于参考线下的规划起点状态
        init_frenet_state = cartesian2frenet([init_cartesian_state[0]],
                                             [init_cartesian_state[1]],
                                             [init_cartesian_state[2]],
                                             [init_cartesian_state[3]],
                                             [init_cartesian_state[4]],
                                             [init_cartesian_state[5]],
                                             reference_line[0],
                                             reference_line[1],
                                             reference_line[2],
                                             reference_line[3],
                                             reference_line[4])

        print("init s, l, kappa:", f"{init_frenet_state[0][0]:.3f}", f"{init_frenet_state[1][0]:.3f}", f"{self.reference_line_provider.get_kappa()(30):.3f}")
        lat_trajectory, lon_trajectory = [], []
        # lateral(l + s) path planning (l_s, dl_s ,ddl_s, l_e, dl_e, ddl_e, s)
        end_lat_states = self.sample_lateral_end_state_ds()
        # 可视化横向采样的path: 
        for end_lat_state in end_lat_states:
            # 横向上是从s = 0开始推导的，因此是基于自车为原点的坐标系
            lateral_curve = QuinticPolynomial(init_frenet_state[1][0], init_frenet_state[4][0], init_frenet_state[7][0],
                                              end_lat_state[0], end_lat_state[1], end_lat_state[2], end_lat_state[3])
            lat_trajectory.append(lateral_curve)
            # '''
            s, l = [], []
            print()
            for s_sample in np.arange(0, end_lat_state[3], 0.2):
                s.append(s_sample)
                l.append(lateral_curve.get_point(s_sample))
            plt.axis("equal")
            plt.plot(s, l)
            plt.xlabel("S[m]")
            plt.ylabel("L[m]")
            # '''

        # longitudinal(v + t) path planning (s_s, ds_s, dds_s, ds_e, dds_e, time)
        target_speed = 5.0
        end_lon_frenet_states = self.sample_lon_end_state(init_frenet_state, target_speed)
        s_time, vel = [], []
        for idx in range(len(end_lon_frenet_states)):
            s_time.append(end_lon_frenet_states[idx][3])
            vel.append(end_lon_frenet_states[idx][1])
        # plt.figure()
        # plt.plot(s_time, vel, 'g.')
        # plt.figure()
        for end_lon_frenet_state in end_lon_frenet_states:
            # 纵向上从自车的s处开始累计，因此是参考线起点的坐标系
            longitudinal_curve = QuarticPolynominal(init_frenet_state[0][0], init_frenet_state[2][0], init_frenet_state[6][0],
                                                    end_lon_frenet_state[1], end_lon_frenet_state[2], end_lon_frenet_state[3])
            lon_trajectory.append(longitudinal_curve)
            '''
            t, v = [], []
            for t_sample in np.arange(0, end_lon_frenet_state[3], 0.2):
                t.append(t_sample)
                v.append(longitudinal_curve.get_first_derivative(t_sample))
            plt.axis("equal")
            plt.plot(t, v)
            plt.xlabel("T[m]")
            plt.ylabel("V[m/s]")
            '''
        # get the optimal trajectory
        optimal_lat_trajectory, optimal_lon_trajectory = self.get_optimal_trajectory(init_frenet_state, lat_trajectory, lon_trajectory)

        # visualize V-T figure.
        '''
        v, time = [], []
        for t in np.arange(0.0, optimal_lon_trajectory.get_time(), 0.1):
            v.append(optimal_lon_trajectory.get_first_derivative(t))
            time.append(t)
        plt.figure()
        plt.axis("equal")
        plt.plot(time, v, 'r.')
        plt.xlabel("T[s]")
        plt.ylabel("V[m/s]")
        '''

        # get the optimal trajectory in frenet
        l = []
        dl = []
        ddl = []
        s = []
        s_local = []
        # 方法1：根据间隔时间，由纵向轨迹得到s，再由s得到l
        print("optimal time:", optimal_lon_trajectory.get_time())
        for t in np.arange(0, self.horizon_time.time_s, self.sampling_time.time_s):
            if (t > optimal_lon_trajectory.get_time()):
                break
            # 由速度曲线得到 s, 再由 s 得到 l.
            sampled_gloabl_s = optimal_lon_trajectory.get_point(t)
            sampled_local_s = sampled_gloabl_s - init_frenet_state[0][0]
            l.append(optimal_lat_trajectory.get_point(sampled_local_s))
            dl.append(optimal_lat_trajectory.get_first_derivative(sampled_local_s))
            ddl.append(optimal_lat_trajectory.get_second_derivative(sampled_local_s))
            s.append(sampled_gloabl_s) # 输出frenet下基于reference line 的纵向位置
            s_local.append(sampled_local_s)
        # 方法2：离散 s，直接由横向轨迹得到
        # for sampled_s in np.arange(0, optimal_lat_trajectory.get_param(), 0.2):
        #     # 由速度曲线得到 s, 再由 s 得到 l.
        #     l.append(optimal_lat_trajectory.get_point(sampled_s))
        #     dl.append(optimal_lat_trajectory.get_first_derivative(sampled_s))
        #     ddl.append(optimal_lat_trajectory.get_second_derivative(sampled_s))
        #     s.append(sampled_s)
        plt.plot(s_local, l, '-', color='black', linewidth=2.0)
        return l, dl, ddl, s