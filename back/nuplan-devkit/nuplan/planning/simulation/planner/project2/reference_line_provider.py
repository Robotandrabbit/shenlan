import numpy as np
import math
import logging
from typing import List, Type, Optional, Tuple
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from scipy.interpolate import interp1d
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState

logger = logging.getLogger(__name__)

class ReferenceLineProvider():

    def __init__(
        self,
        router: BFSRouter):
        self._discrete_path: List[StateSE2] = router._discrete_path
        self._lb_of_path: List[float] = router._lb_of_path
        self._rb_of_path: List[float] = router._rb_of_path
        self._s_of_path: List[float] = router._s_of_path
        self._max_v_of_path: List[float] = router._max_v_of_path
        self._lb_of_reference_line: List[float] = []
        self._rb_of_reference_line: List[float] = []
        self._max_v_of_reference_line: List[float] = []
        self._s_of_reference_line: List[float] = []
        self._x_of_reference_line: List[float] = []
        self._y_of_reference_line: List[float] = []
        self._heading_of_reference_line: List[float] = []
        self._kappa_of_reference_line: List[float] = []
        self._interp1d_x = None
        self._interp1d_y = None
        self._interp1d_heading = None
        self._interp1d_kappa = None
        self._ego_idx_of_reference_line = 0
        self._edge_of_path = router._edge_of_path

    def _reference_line_generate(self, ego_state: EgoState):
        """
        According delta_s to sample path, get reference line.
        """
        self._lb_of_reference_line = []
        self._rb_of_reference_line = []
        self._max_v_of_reference_line = []
        self._s_of_reference_line = []
        self._x_of_reference_line = []
        self._y_of_reference_line = []
        delta_s = 1
        length_forward = 200
        length_backward = 30 # 导航路径不一定能满足

        # find the nearest point of ego on path
        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_pos = np.array([ego_x, ego_y])
        i = 0
        j = 1
        a = self._edge_of_path
        for _ in range(len(self._discrete_path)-1):
            di = np.sum((self._discrete_path[i].array - ego_pos)**2)
            dj = np.sum((self._discrete_path[j].array - ego_pos)**2)
            if dj > di:
                break
            i += 1
            j += 1
            if j==len(self._discrete_path):
                logger.error('Can not find the nearest point of ego on path, ego_x: %f, ego_y: %f,\
                             path_start_x:%f, path_start_y:%f, path_end_x:%f, path_end_y:%f, \
                             path_length:%f', ego_x, ego_y, self._discrete_path[0].x, self._discrete_path[0].y, \
                             self._discrete_path[-1].x, self._discrete_path[-1].y, self._s_of_path[-1])
        ego_idx_of_path = i # discrete_path离ego最近点的索引
        self._s_of_path = [s-self._s_of_path[ego_idx_of_path] for s in self._s_of_path]

        # 前向采样
        i = ego_idx_of_path
        j = i + 1
        for s in range(0, length_forward, delta_s):
            while not(self._s_of_path[i] <= s and self._s_of_path[j] >= s):
                i += 1
                j += 1
                if j==len(self._s_of_path):
                    logger.error('Can not continue sample forward, s: %f, path_start_s: %f, path_end_s:%f', \
                                  s, self._s_of_path[0], self._s_of_path[-1])
            x = self.cal_point_in_line(self._s_of_path[i], self._discrete_path[i].x, self._s_of_path[j],\
                                  self._discrete_path[j].x, s)
            y = self.cal_point_in_line(self._s_of_path[i], self._discrete_path[i].y, self._s_of_path[j],\
                                  self._discrete_path[j].y, s)
            self._s_of_reference_line.append(s)
            self._x_of_reference_line.append(x)
            self._y_of_reference_line.append(y)
            self._lb_of_reference_line.append(self._lb_of_path[i])
            self._rb_of_reference_line.append(self._rb_of_path[i])
            self._max_v_of_reference_line.append(self._max_v_of_path[i])

        # 后向采样
        self._ego_idx_of_reference_line = 0 # 维护s = 0在self._s_of_reference_line中的索引
        j = ego_idx_of_path
        i = j - 1
        for s in range(-delta_s, -length_backward, -delta_s):
            while not(self._s_of_path[i] <= s and self._s_of_path[j] >= s):
                i -= 1
                j -= 1
                if i < 0: # 如果导航路径不满足就停止
                    break
            if i < 0: # 如果导航路径不满足就停止
                break
            x = self.cal_point_in_line(self._s_of_path[i], self._discrete_path[i].x, self._s_of_path[j],\
                                  self._discrete_path[j].x, s)
            y = self.cal_point_in_line(self._s_of_path[i], self._discrete_path[i].y, self._s_of_path[j],\
                                  self._discrete_path[j].y, s)
            self._s_of_reference_line.insert(0, s)
            self._x_of_reference_line.insert(0, x)
            self._y_of_reference_line.insert(0, y)
            self._lb_of_reference_line.insert(0, self._lb_of_path[i])
            self._rb_of_reference_line.insert(0, self._rb_of_path[i])
            self._max_v_of_reference_line.insert(0, self._max_v_of_path[i])
            self._ego_idx_of_reference_line += 1

        self._s_of_reference_line = [s+delta_s*self._ego_idx_of_reference_line for s in self._s_of_reference_line]  
        self._interp1d_x =  interp1d(self._s_of_reference_line, self._x_of_reference_line)
        self._interp1d_y =  interp1d(self._s_of_reference_line, self._y_of_reference_line)
        self.calculate_heading()
        self._interp1d_heading =  interp1d(self._s_of_reference_line, self._heading_of_reference_line)
        self.calculate_kappa()
        self._interp1d_kappa =  interp1d(self._s_of_reference_line, self._kappa_of_reference_line)


    def cal_point_in_line(self, x1, y1, x2, y2, x3):
        if abs(x2 - x1) < 1e-6:
            return y1
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        y3 = k * x3 + b
        return y3

    def calculate_heading(self):
        self._heading_of_reference_line = [0]
        for i in range(len(self._x_of_reference_line) - 1):
            x1 = self._x_of_reference_line[i]
            y1 = self._y_of_reference_line[i]
            x2 = self._x_of_reference_line[i+1]
            y2 = self._y_of_reference_line[i+1]
            dx = x2 - x1
            dy = y2 - y1
            heading = math.atan2(dy, dx)
            heading = (heading + np.pi) % (2 * np.pi) -np.pi # Wrap heading angle in to [-pi, pi]
            self._heading_of_reference_line.append(heading)
        self._heading_of_reference_line[0] = self._heading_of_reference_line[1]

    def calculate_kappa(self):
        dx = np.diff(self._x_of_reference_line)
        dx = np.insert(dx, 0, dx[0])
        dy = np.diff(self._y_of_reference_line)
        dy = np.insert(dy, 0, dy[0])
        ds = np.sqrt(dx**2 + dy**2)
        dheading = np.diff(self._heading_of_reference_line)
        dheading = np.insert(dheading, 0, dheading[0])
        self._kappa_of_reference_line = np.divide(np.sin(dheading), ds).tolist()

    def get_boundary(self, s_set: List[float]):
        """
        根据参考线的s，返回对应的左右边界。此处依据参考线s间隔为1m进行简化
        左边界为正，右边界为负
        :param s_set
        :return lb_set, rb_set
        """
        lb_set = []
        rb_set = []
        for idx in range(len(s_set)):
            s = s_set[idx]
            lb = self._lb_of_reference_line[math.floor(s)]
            rb = -self._rb_of_reference_line[math.floor(s)]
            lb_set.append(lb)
            rb_set.append(rb)
        return lb_set, rb_set

        



        