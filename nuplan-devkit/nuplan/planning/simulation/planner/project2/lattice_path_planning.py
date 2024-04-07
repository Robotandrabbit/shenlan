import math
import numpy as np

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.quintic_polynominal import QuinticPolynomial
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.planning.simulation.planner.project2.frame_transform import cartesian2frenet

WEIGHT_PROGRESS = 1.0
WEIGHT_OFFSET = 5.0
WEIGHT_SMOOTH = 10.0

MAXIMUM_JERK = 1.5
MAXIMUM_PROGRESS = 120
MAXIMUM_OFFSET = 1.5

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

    def sample_lateral_end_state(self):
        end_d_candidates = np.array([0.0, -0.5, 0.5])
        end_s_candidates = np.array([10.0, 20.0, 40.0])
        
        sampled_states = []
        for s in end_s_candidates:
            for d in end_d_candidates:
                state = np.array([d, 0.0, 0.0, s])
                sampled_states.append(state)
        return sampled_states

    def evaluate_trajectory(self, trajectory: list[QuinticPolynomial, QuinticPolynomial]) -> float:
        cost = 0.0
        # 1. longitudinal progress cost
        progress = trajectory[1].get_point(self.horizon_time.time_s)
        if (progress < MAXIMUM_PROGRESS):
            cost += WEIGHT_PROGRESS * (MAXIMUM_PROGRESS - progress) / MAXIMUM_PROGRESS
        
        for t in np.arange(0.0, self.horizon_time.time_s, self.sampling_time.time_s):
            # 2. lateral smooth cost
            lateral_jerk = trajectory[0].get_third_derivative(t)
            if (lateral_jerk > 1.0):
                cost += WEIGHT_SMOOTH * (lateral_jerk / MAXIMUM_JERK)

            # 3. longidinal smooth cost
            longitudinal_jerk = trajectory[1].get_third_derivative(t)
            if (longitudinal_jerk > 1.0):
                cost += WEIGHT_SMOOTH * (longitudinal_jerk - 1.0) / MAXIMUM_JERK
            
            # 4. offset cost
            lateral_offset = trajectory[0].get_point(t)
            if (lateral_offset > 0.5):
                cost += WEIGHT_OFFSET * (lateral_offset - 0.5) / MAXIMUM_OFFSET
            
            # 5. TODO(wanghao): collision cost
        return cost
        

    def get_optimal_trajectory(self, trajectories) -> list[QuinticPolynomial, QuinticPolynomial]:
        min_score = float('+inf')
        best_trajectory = None
        for trajectory in trajectories:
            score = self.evaluate_trajectory(trajectory)
            if (score < min_score):
                min_score = score
                best_trajectory = trajectory
        
        return best_trajectory


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

               
        end_frenet_states = self.sample_lateral_end_state()
        trajectories = []
        # 对每个采样的终点
        for end_frenet_state in end_frenet_states:
            # lateral(d/l) path planning
            lateral_curve = QuinticPolynomial(init_frenet_state[1][0], init_frenet_state[4][0], init_frenet_state[7][0],
                                              end_frenet_state[0], end_frenet_state[1], end_frenet_state[2], self.horizon_time.time_s)
            # longitudinal path planning: velocity = 15m/s; accelaration = 0.0;
            longitudinal_curve = QuinticPolynomial(init_frenet_state[0][0], init_frenet_state[2][0], init_frenet_state[6][0],
                                                   end_frenet_state[3], 15.0, 0.0, self.horizon_time.time_s)
            trajectories.append([lateral_curve, longitudinal_curve])
        
        # get the optimal trajectory
        optimal_trajectory = self.get_optimal_trajectory(trajectories)

        # get the optimal trajectory in frenet
        l = []
        dl = []
        ddl = []
        s = []
        for t in np.arange(0, self.horizon_time.time_s, self.sampling_time.time_s):
            l.append(optimal_trajectory[0].get_point(t))
            dl.append(optimal_trajectory[0].get_first_derivative(t))
            ddl.append(optimal_trajectory[0].get_second_derivative(t))
            s.append(optimal_trajectory[1].get_point(t))
        
        return l, dl, ddl, s