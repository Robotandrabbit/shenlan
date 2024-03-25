import math
import numpy as np

from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.quintic_polynominal import QuinticPolynomial
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.planning.simulation.planner.project2.frame_transform import cartesian2frenet

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
        
        sampled_state = []
        for s in end_s_candidates:
            for d in end_d_candidates:
                state = np.array([d, 0.0, 0.0, s])
                sampled_state.append(state)

    def path_planning(self) -> tuple[float, float, float, float]:
        # calculate ego state lateral state
        cos_h = math.cos(self.ego_state.car_footprint.oriented_box.center.heading)
        sin_h = math.sin(self.ego_state.car_footprint.oriented_box.center.heading)

        init_cartesian_state = np.array([self.ego_state.car_footprint.oriented_box.center.x,
                                        self.ego_state.car_footprint.oriented_box.center.y,
                                        self.ego_state.dynamic_car_state.rear_axle_velocity_2d * cos_h,
                                        self.ego_state.dynamic_car_state.rear_axle_velocity_2d * sin_h,
                                        self.ego_state.dynamic_car_state.rear_axle_acceleration_2d * cos_h,
                                        self.ego_state.dynamic_car_state.rear_axle_acceleration_2d * sin_h])

        reference_line = np.array([self.reference_line_provider._x_of_reference_line,
                                   self.reference_line_provider._y_of_reference_line,
                                   self.reference_line_provider._heading_of_reference_line,
                                   self.reference_line_provider._kappa_of_reference_line,
                                   self.reference_line_provider._s_of_reference_line])
        
        init_frenet_state = cartesian2frenet(init_cartesian_state[0],
                                             init_cartesian_state[1],
                                             init_cartesian_state[2],
                                             init_cartesian_state[3],
                                             init_cartesian_state[4],
                                             init_cartesian_state[5],
                                             reference_line[0],
                                             reference_line[1],
                                             reference_line[2],
                                             reference_line[3],
                                             reference_line[4])

        # lateral(d/l) path planning        
        end_frenet_state = self.sample_lateral_end_state()
        lateral_curve = QuinticPolynomial(init_frenet_state[1], init_frenet_state[4], init_frenet_state[7],
                                          end_frenet_state[0], end_frenet_state[1], end_frenet_state[2], self.horizon_time)
        l = []
        dl = []
        ddl = []
        for t in range(0, self.horizon_time, self.sampling_time):
            l.append(lateral_curve.get_point(t))
            dl.append(lateral_curve.get_first_derivative(t))
            ddl.append(lateral_curve.get_second_derivative(t))
        
        # longitudinal path planning
        longitudinal_curve = QuinticPolynomial(init_frenet_state[0], init_frenet_state[2], init_frenet_state[6],
                                               end_frenet_state[3], 15.0, 0.0, self.horizon_time)
        s = []
        for t in range(0, self.horizon_time, self.sampling_time):
            l.append(longitudinal_curve.get_point(t))
        
        return l, dl, ddl, s
