import math
import logging
import time
from typing import List, Type, Optional, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.planning.simulation.planner.project2.simple_predictor import SimplePredictor
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.planning.simulation.planner.project2.dp_decider import DpDecider

from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects

from nuplan.planning.simulation.planner.project2.lattice_path_planning import LatticePathPlanning

logger = logging.getLogger(__name__)


class MyPlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
    ):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """

        # 1. Routing
        ego_state, observations = current_input.history.current_state
        if not self._routing_complete:
            self._router._initialize_ego_path(ego_state, self.max_velocity)
            self._routing_complete = True

        # 2. Generate reference line
        self._reference_path_provider = ReferenceLineProvider(self._router)
        self._reference_path_provider._reference_line_generate(ego_state)

        # 3. Objects prediction
        self._predictor = SimplePredictor(ego_state, observations, self.horizon_time.time_s, self.sampling_time.time_s)
        objects = self._predictor.predict()

        # 4. Planning
        trajectory: List[EgoState] = self.planning(ego_state, self._reference_path_provider, objects,
                                                    self.horizon_time, self.sampling_time, self.max_velocity)

        return InterpolatedTrajectory(trajectory)
    
    def get_constant_speed_profile(self, horizon_time:float, sampling_time:float, default_velocity:float):
        optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t = [], [], [], []
        t = 0.0
        while t < horizon_time /sampling_time:
            optimal_speed_s.append(5.0 * t)
            optimal_speed_s_dot.append(5.0)
            optimal_speed_s_2dot.append(0.0)
            optimal_speed_t.append(t)
            t += sampling_time
        return optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t

    # TODO: 2. Please implement your own trajectory planning.
    def planning(self,
                 ego_state: EgoState,
                 reference_path_provider: ReferenceLineProvider,
                 objects: TrackedObjects,
                 horizon_time: TimePoint,
                 sampling_time: TimePoint,
                 max_velocity: float) -> List[EgoState]:
        """
        Implement trajectory planning based on input and output, recommend using lattice planner or piecewise jerk planner.
        param: ego_state Initial state of the ego vehicle
        param: reference_path_provider Information about the reference path
        param: objects Information about dynamic obstacles
        param: horizon_time Total planning time
        param: sampling_time Planning sampling time
        param: max_velocity Planning speed limit (adjustable according to road speed limits during planning process)
        return: trajectory Planning result
        """


        # 可以实现基于采样的planer或者横纵向解耦的planner,此处给出planner的示例,仅提供实现思路供参考
        # 1.Path planning
        lattice_path_planning = LatticePathPlanning (ego_state, reference_path_provider, horizon_time, sampling_time)
        optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s = lattice_path_planning.path_planning()
        '''
        print(optimal_path_s[0], optimal_path_s[-1])
        plt.figure()
        plt.plot(optimal_path_s, optimal_path_l)
        plt.plot(optimal_path_s[0], optimal_path_l[0], 'go')
        plt.plot(optimal_path_s[-1], optimal_path_l[-1], 'ro')
        plt.axis("equal")
        plt.show()
        '''
        # 2.Transform path planning result to cartesian frame
        path_idx2s, path_x, path_y, path_heading, path_kappa = transform_path_planning(optimal_path_s, optimal_path_l, \
                                                                                       optimal_path_dl,
                                                                                       optimal_path_ddl, \
                                                                                       reference_path_provider)

        # 3.Speed planning
        # optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t = speed_planning( \
        #     ego_state, horizon_time.time_s, max_velocity, object, \
        #     path_idx2s, path_x, path_y, path_heading, path_kappa)
        # 1）简单的匀速运动 5m/s
        optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t = \
            self.get_constant_speed_profile(horizon_time.time_s, sampling_time.time_s, max_velocity)
        '''

        # 2) DpDecider. TODO(wanghao): 3) TreeSearch
        predicted_trajectoris = [] # 存储所有动态障碍物轨迹
        obs_radius = []
        for agent in objects.get_agents():
            temp_traj = []
            # 跳过，若无预测轨迹
            if (len(agent.predictions) == 0):
                continue
            for waypoint in agent.predictions[0].valid_waypoints:
                temp_traj.append([waypoint.time_point * 1e-6, waypoint.x, waypoint.y])
            predicted_trajectoris.append(temp_traj)
            obs_radius.append(0.5 * math.sqrt(agent.box.length ** 2 + agent.box.width ** 2))
        

        start_time = time.time()
        dp_decider = DpDecider(predicted_trajectoris, obs_radius, path_idx2s, path_x, path_y, path_heading, path_kappa, \
                                self.horizon_time.time_s, self.sampling_time.time_s, max_velocity, \
                                ego_state.agent.box.half_width, ego_state.agent.box.length, ego_state.agent.velocity.magnitude(), \
                                5.0, 5.0)
        _, _, optimal_speed_s, grid_speed_v, optimal_speed_t = dp_decider.dynamic_programming()
        print("dp time:", time.time() - start_time)
        # 利用 s 的微分求出 v, 也可以通过查找最优(t, s)对应的 grid_speed_v 求出 v
        optimal_speed_s_dot = [ego_state.dynamic_car_state.speed]
        for idx in np.arange(1, len(optimal_speed_s), 1):
            optimal_speed_s_dot.append((optimal_speed_s[idx] - optimal_speed_s[idx - 1]) / self.sampling_time.time_s)

        # 利用 v 的微分求出 a
        optimal_speed_s_2dot = [ego_state.dynamic_car_state.acceleration]
        for idx in np.arange(1, len(optimal_speed_s_dot), 1):
            optimal_speed_s_2dot.append((optimal_speed_s_dot[idx] - optimal_speed_s_dot[idx - 1]) / self.sampling_time.time_s)
        
        '''   
        # 4.Produce ego trajectory
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            ),
            tire_steering_angle=ego_state.dynamic_car_state.tire_steering_rate,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for iter in range(int(horizon_time.time_us / sampling_time.time_us)):
            relative_time = (iter + 1) * sampling_time.time_s
            # 根据 relative_time 和 speed planning 计算 velocity accelerate （三次多项式）
            s, velocity, accelerate = cal_dynamic_state(relative_time, optimal_speed_t, optimal_speed_s,
                                                        optimal_speed_s_dot, optimal_speed_s_2dot)
            # 根据当前时间下的s 和 路径规划结果 计算 x y heading kappa （线形插值）
            x, y, heading, _ = cal_pose(s, path_idx2s, path_x, path_y, path_heading, path_kappa)

            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(velocity, 0),
                rear_axle_acceleration_2d=StateVector2D(accelerate, 0),
                tire_steering_angle=heading,
                time_point=state.time_point + sampling_time,
                vehicle_parameters=state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=0,
                angular_accel=0,
            )

            trajectory.append(state)
        
        return trajectory