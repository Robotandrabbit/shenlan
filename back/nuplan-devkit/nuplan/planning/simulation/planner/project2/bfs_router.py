import numpy as np
import logging, math
from typing import List, Type, Optional, Tuple
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D


logger = logging.getLogger(__name__)


class BFSRouter:

    def __init__(
        self,
        map_api:AbstractMap):
        self._route_roadblocks: List[RoadBlockGraphEdgeMapObject] = []
        self._candidate_lane_edge_ids: Optional[List[str]] = None
        self._map_api: AbstractMap = map_api
        self._discrete_path: List[StateSE2] = []
        self._lb_of_path: List[float] = []
        self._max_v_of_path: List[float] = []
        self._rb_of_path: List[float] = []
        self._s_of_path: List[float] = []
        self._edge_of_path: List[LaneGraphEdgeMapObject] = []

    def _initialize_route_plan(self, route_roadblock_ids: List[str]) -> None:
        """
        Initializes the route plan with roadblocks.
        :param route_roadblock_ids: A list of roadblock ids that make up the ego's route
        """
        self._route_roadblocks = []
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

        assert (
            self._route_roadblocks
        ), "Cannot create route plan. No roadblocks were extracted from the given route_roadblock_ids!"

    def _initialize_ego_path(self, ego_state: EgoState, max_velocity):
        """
        Initializes the ego path from the ground truth driven trajectory
        :param ego_state: The ego state at the start of the scenario.
        :param max_velocity: max velocity of ego
        """
        route_plan, _ = self._breadth_first_search(ego_state)
        self._edge_of_path = route_plan
        ego_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        speed_limit = route_plan[0].speed_limit_mps or max_velocity
        max_velocity = speed_limit if speed_limit > ego_speed else ego_speed
        
        self._discrete_path = []
        self._lb_of_path = []
        self._rb_of_path = []
        self._s_of_path = []
        for idx in range(len(route_plan)):
            edge = route_plan[idx]
            self._discrete_path.extend(edge.baseline_path.discrete_path)
            lb = edge.left_boundary
            rb = edge.right_boundary
            point_in_baseline = edge.baseline_path.discrete_path[0].point
            nearest_pose_of_lb = lb.get_nearest_pose_from_position(point_in_baseline)
            distance_to_lb = np.linalg.norm(point_in_baseline.array - nearest_pose_of_lb.array)
            nearest_pose_of_rb = rb.get_nearest_pose_from_position(point_in_baseline)
            distance_to_rb = np.linalg.norm(point_in_baseline.array - nearest_pose_of_rb.array)
            # distance_to_lb, distance_to_rb = edge.get_width_left_right(point_in_baseline) # function of get_width_left_right is not implemented...
            # if idx == 0: # changing lane to left in the first edge
            #     distance_to_rb += (distance_to_lb + distance_to_rb)
            lb_of_path = [distance_to_lb] * len(edge.baseline_path.discrete_path)
            self._lb_of_path.extend(lb_of_path)
            rb_of_path = [distance_to_rb] * len(edge.baseline_path.discrete_path)
            self._rb_of_path.extend(rb_of_path)
            max_v_of_path = [edge.speed_limit_mps] * len(edge.baseline_path.discrete_path)
            self._max_v_of_path.extend(max_v_of_path)

        self._s_of_path = []
        s = 0
        self._s_of_path.append(s)
        for idx in range(len(self._discrete_path)-1):
            dis = np.linalg.norm(self._discrete_path[idx+1].point.array - self._discrete_path[idx].point.array)
            s += dis
            self._s_of_path.append(s)
        return max_velocity

    def _breadth_first_search(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs iterative breath first search to find a route to the target roadblock.
        :param ego_state: Current ego state.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock. If unsuccessful a longest route is given.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"

        starting_edge = self._get_starting_edge(ego_state)
        graph_search = BreadthFirstSearch(starting_edge, self._candidate_lane_edge_ids)
        # Target depth needs to be offset by one if the starting edge belongs to the second roadblock in the list
        # offset = 1 if starting_edge.get_roadblock_id() == self._route_roadblocks[1].id else 0
        # route_plan, path_found = graph_search.search(self._route_roadblocks[-1], len(self._route_roadblocks[offset:]))
        offset = 0
        for i in range(5):
            if starting_edge.get_roadblock_id() == self._route_roadblocks[i].id:
                offset = i
                break
        route_plan, path_found = graph_search.search(self._route_roadblocks[-1], len(self._route_roadblocks[offset:]))

        if not path_found:
            logger.warning(
                "IDMPlanner could not find valid path to the target roadblock. Using longest route found instead"
            )
        return route_plan, path_found

    def _get_starting_edge(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Get the starting edge based on ego state. If a lane graph object does not contain the ego state then
        the closest one is taken instead.
        :param ego_state: Current ego state.
        :return: The starting LaneGraphEdgeMapObject.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert len(self._route_roadblocks) >= 2, "_route_roadblocks should have at least 2 elements!"

        starting_edge = None
        closest_distance = math.inf

        # Check for edges in about the first 5 roadblocks
        for idx in range(5):
            for edge in self._route_roadblocks[idx].interior_edges:
                if edge.contains_point(ego_state.center):
                    starting_edge = edge
                    # Use the nearest left edge as starting edge, simulate changing lane to left
                    # starting_edge = self.get_nearest_left_edge(edge, self._route_roadblocks[idx].interior_edges)
                    break

            if starting_edge != None:
                break

        if starting_edge == None:
            for inx in range(5):
                if len(self._route_roadblocks[0].incoming_edges) > 0:
                    self._route_roadblocks.insert(0, self._route_roadblocks[0].incoming_edges[0]);
                    if self._route_roadblocks[0].contains_point(ego_state.center):
                        break
                else:
                    break

            # Check for edges in about the first 5 roadblocks
            for idx in range(5):
                for edge in self._route_roadblocks[idx].interior_edges:
                    if edge.contains_point(ego_state.center):
                        starting_edge = edge
                        # Use the nearest left edge as starting edge, simulate changing lane to left
                        # starting_edge = self.get_nearest_left_edge(edge, self._route_roadblocks[idx].interior_edges)
                        break

                if starting_edge != None:
                    break

        assert starting_edge, "Starting edge for BFS Router could not be found!, planning failed"
        return starting_edge
    
    def get_nearest_left_edge(self, edge: LaneGraphEdgeMapObject, 
                              parallel_edges: List[LaneGraphEdgeMapObject]) -> LaneGraphEdgeMapObject:
        l0 = edge.baseline_path.discrete_path[-1].array - edge.baseline_path.discrete_path[0].array
        # candidate_edges = edge.parallel_edges # function of parallel_edges is not implemented....
        rtn = edge
        cross_product_max = math.inf
        for target in parallel_edges:
            l1 = target.baseline_path.discrete_path[-1].array - edge.baseline_path.discrete_path[0].array
            cross_product = np.cross(l0, l1)
            if cross_product > 1e-6 and cross_product < cross_product_max:
                rtn = target
                cross_product_max = cross_product
        return rtn
