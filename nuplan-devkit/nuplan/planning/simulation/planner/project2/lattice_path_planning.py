import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.quintic_polynominal import QuinticPolynomial
from nuplan.planning.simulation.planner.project2.quartic_polynominal import QuarticPolynominal
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.planning.simulation.planner.project2.frame_transform import cartesian2frenet

WEIGHT_PROGRESS = 1.0
WEIGHT_OFFSET = 5.0
WEIGHT_SMOOTH = 10.0

MAXIMUM_JERK = 1.5
MAXIMUM_PROGRESS = 120
MAXIMUM_OFFSET = 1.5

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
    def sample_lateral_end_state_ds(self, init_frenet_state):
        end_d_candidates = np.arange(-1.0, 1.2, 0.5)
        end_s_candidates = np.array([init_frenet_state[0][0] + 10.0,
                                     init_frenet_state[0][0] + 20.0,
                                     init_frenet_state[0][0] + 40.0])
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
        # time_samples.append(0.01)
        for i in range(1, 9):
            time_samples.append(i)
        
        for time in time_samples:
            v_upper = min(init_frenet_state[2][0] + MAXIMUM_DECELERATION * 1.0, target_speed) # 间隔时间 = 1.0s
            v_lower = max(init_frenet_state[2][0] - MAXIMUM_ACCELERATION * 1.0, 0.0)
            end_states.append([0.0, v_upper, 0.0, time])
            end_states.append([0.0, v_lower, 0.0, time])
            v_range = v_upper - v_lower
            num_of_mid_points = int(min(4, v_range / 1.0))
            if (num_of_mid_points > 0):
                velocity_seg = v_range / (num_of_mid_points + 1)
                for i in range(num_of_mid_points):
                    end_states.append([0.0, v_lower + velocity_seg * i, 0.0, time])
        
        return end_states
            

    def evaluate_trajectory(self, lon_trajectory:QuarticPolynominal, lat_trajectory:QuinticPolynomial) -> float:
        cost = 0.0
        # 1. longitudinal progress cost
        progress = lon_trajectory.get_point(lon_trajectory.get_time())
        if (progress < MAXIMUM_PROGRESS):
            cost += WEIGHT_PROGRESS * (MAXIMUM_PROGRESS - progress) / MAXIMUM_PROGRESS
        
        for t in np.arange(0.0, self.horizon_time.time_s, self.sampling_time.time_s):
            # 2. lateral smooth cost
            lateral_jerk = lat_trajectory.get_third_derivative(t)
            if (lateral_jerk > 1.0):
                cost += WEIGHT_SMOOTH * (lateral_jerk / MAXIMUM_JERK)
            
            # 3. offset cost
            lateral_offset = lat_trajectory.get_point(t)
            if (lateral_offset > 0.5):
                cost += WEIGHT_OFFSET * (lateral_offset - 0.5) / MAXIMUM_OFFSET
            
            # TODO(wanghao): add more cost.
            # 4. collision cost:static agent.
            # 5. flicker cost
        return cost

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

    def get_optimal_trajectory(self, lat_trajectories, lon_trajectories) -> Tuple[QuinticPolynomial, QuarticPolynominal]:
        min_score = float('+inf')
        best_lon_trajectory = None
        best_lat_trajectory = None
        has_valid_traj = False
        for lon_trajectory in lon_trajectories:
            # Check if lon_trajectory is valid.
            if (not self.is_valid_lon_trajectory(lon_trajectory)):
                continue
            for lat_trajectory in lat_trajectories:
                score = self.evaluate_trajectory(lon_trajectory, lat_trajectory)
                if (score < min_score):
                    min_score = score
                    best_lon_trajectory = lon_trajectory
                    best_lat_trajectory = lat_trajectory
                    has_valid_traj = True
        # Combine two 1d trajectories to one 2d trajectory
        if (not has_valid_traj):    
            print("Failed to sample trajectory")
        return best_lat_trajectory, best_lon_trajectory


    def path_planning(self) -> tuple[float, float, float, float]:
        # calculate ego state lateral state
        cos_h = math.cos(self.ego_state.car_footprint.oriented_box.center.heading)
        sin_h = math.sin(self.ego_state.car_footprint.oriented_box.center.heading)

        init_cartesian_state = np.array([self.ego_state.car_footprint.oriented_box.center.x,
                                        self.ego_state.car_footprint.oriented_box.center.y,
                                        self.ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude() * cos_h,
                                        self.ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude() * sin_h,
                                        self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.magnitude() * cos_h,
                                        self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.magnitude() * sin_h])

        reference_line = np.array([self.reference_line_provider._x_of_reference_line,
                                   self.reference_line_provider._y_of_reference_line,
                                   self.reference_line_provider._heading_of_reference_line,
                                   self.reference_line_provider._kappa_of_reference_line,
                                   self.reference_line_provider._s_of_reference_line])
        
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

               
        lat_trajectory, lon_trajectory = [], []
        # lateral(l + s) path planning (l_s, dl_s ,ddl_s, l_e, dl_e, ddl_e, s)
        end_lat_states = self.sample_lateral_end_state_ds(init_frenet_state)
        # 可视化横向采样的path
        # plt.figure()
        for end_lat_state in end_lat_states:
            lateral_curve = QuinticPolynomial(init_frenet_state[1][0], init_frenet_state[3][0], init_frenet_state[5][0],
                                              end_lat_state[0], end_lat_state[1], end_lat_state[2], end_lat_state[3])
            lat_trajectory.append(lateral_curve)
            # s, l = [], []
            # for s_sample in np.arange(init_frenet_state[1][0], end_lat_state[3], 0.2):
            #     s.append(s_sample)
            #     l.append(lateral_curve.get_point(s_sample))
            # plt.axis("equal")
            # plt.plot(s, l)
            # plt.xlabel("S[m]")
            # plt.ylabel("L[m]")
            

        # longitudinal(v + t) path planning (s_s, ds_s, dds_s, ds_e, dds_e, time)
        target_speed = 5.0
        end_lon_frenet_states = self.sample_lon_end_state(init_frenet_state, target_speed)
        for end_lon_frenet_state in end_lon_frenet_states:
            longitudinal_curve = QuarticPolynominal(init_frenet_state[0][0], init_frenet_state[2][0], init_frenet_state[6][0],
                                                    end_lon_frenet_state[1], end_lon_frenet_state[2], end_lon_frenet_state[3])
            lon_trajectory.append(longitudinal_curve)

        # get the optimal trajectory
        optimal_lat_trajectory, optimal_lon_trajectory = self.get_optimal_trajectory(lat_trajectory, lon_trajectory)

        # s, v, l, time = [], [], [], []
        # print("horizon_time.time_s: ", self.horizon_time.time_s)
        # for t in np.arange(0.0, self.horizon_time.time_s, 0.1):
        #     sampled_s = optimal_lon_trajectory.get_point(t)
        #     s.append(sampled_s)
        #     v.append(optimal_lon_trajectory.get_first_derivative(t))
        #     l.append(optimal_lat_trajectory.get_point(sampled_s))
        #     time.append(t)
        # plt.figure()
        # plt.axis("equal")
        # plt.plot(time, v)
        # plt.xlabel("T[s]")
        # plt.ylabel("V[m/s]")
        
        # plt.figure()
        # plt.axis("equal")
        # plt.plot(s, l)
        # plt.xlabel("S[m]")
        # plt.ylabel("L[m]")

        # get the optimal trajectory in frenet
        l = []
        dl = []
        ddl = []
        s = []
        for t in np.arange(0, self.horizon_time.time_s, self.sampling_time.time_s):
            if (t > optimal_lon_trajectory.get_time()):
                break
            # 由速度曲线得到 s, 再由 s 得到 l.
            sampled_s = optimal_lon_trajectory.get_point(t)
            l.append(optimal_lat_trajectory.get_point(sampled_s))
            dl.append(optimal_lat_trajectory.get_first_derivative(sampled_s))
            ddl.append(optimal_lat_trajectory.get_second_derivative(sampled_s))
            s.append(sampled_s)
            # print("t, l, s:", t, l[-1], s[-1])
        
        return l, dl, ddl, s