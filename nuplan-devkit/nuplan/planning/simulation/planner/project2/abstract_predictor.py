import numpy as np
from abc import ABC
from typing import List, Type, Optional, Tuple
from abc import abstractmethod
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState

class AbstractPredictor(ABC):
    @abstractmethod
    def predict(self) -> Tuple[List[List[float]], List[float]]:
        """
        :return list of agent trajectory, every agent trajectory include left_front, right_front, left_rear, right_rear
        """
        pass