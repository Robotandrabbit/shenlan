import numpy as np
from typing import List, Type, Optional, Tuple
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.waypoint import Waypoint

class SimplePredictor(AbstractPredictor):
    def __init__(self, ego_state: EgoState, observations: Observation, duration: float, sample_time: float) -> None:
        self._ego_state = ego_state
        self._observations = observations
        self._duration = duration
        self._sample_time = sample_time
        self._occupancy_map_radius = 40

    def constant_velocity(self, object: TrackedObject) -> PredictedTrajectory:
        cv_probability = 1.0
        cv_waypoints = []
        if isinstance(object, Agent):
          for time in np.arange(self._ego_state.time_seconds, self._ego_state.time_seconds + self._duration, self._sample_time):
              x = object.center.x + time * object.velocity.magnitude() * np.cos(object.center.heading)
              y = object.center.y + time * object.velocity.magnitude() * np.sin(object.center.heading)
              cv_waypoints.append(Waypoint(time_point=time*1e6,
                                           oriented_box=OrientedBox.from_new_pose(object.box, StateSE2(x, y, object.center.heading)), 
                                           velocity=object.velocity))
        return PredictedTrajectory(waypoints=cv_waypoints, probability=cv_probability)

    def predict(self) -> TrackedObjects:
        """Inherited, see superclass."""
        if isinstance(self._observations, DetectionsTracks):
            objects_init = self._observations.tracked_objects.tracked_objects
            # 挑出一定范围内的动态障碍物
            objects = [
                object
                for object in objects_init
                if np.linalg.norm(self._ego_state.center.array - object.center.array) < self._occupancy_map_radius \
                   and isinstance(object, Agent)
            ]

            # TODO：1.Predicted the Trajectory of object
            for object in objects:
                predicted_trajectories = [self.constant_velocity(object)]  # predicted_trajectories : List[PredictedTrajectory]
                object.predictions = predicted_trajectories

            return TrackedObjects(objects)

        else:
            raise ValueError(
                f"SimplePredictor only supports DetectionsTracks. Got {self._observations.detection_type()}")
