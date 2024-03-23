import numpy as np
from typing import List, Tuple
from nuplan.planning.simulation.planner.project2.frame_transform import frenet2cartesian
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from scipy.interpolate import interp1d


def transform_path_planning(
    path_s: List[float], 
    path_l: List[float], 
    path_dl: List[float], 
    path_ddl: List[float], 
    reference_path_provider: ReferenceLineProvider) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    根据s和路径规划，计算s对应的x y heading kappa
    
    """
    # 将path从frenet转换到cartesian
    path_x, path_y, path_heading, path_kappa = \
                    frenet2cartesian(path_s, path_l, path_dl, path_ddl,\
                                    reference_path_provider._x_of_reference_line,
                                    reference_path_provider._y_of_reference_line,
                                    reference_path_provider._heading_of_reference_line,
                                    reference_path_provider._kappa_of_reference_line,
                                    reference_path_provider._s_of_reference_line)
    
    # 根据路径规划结果（optimal_path_x, optimal_path_y），建立optimal path中s和x y的对应关系
    path_idx2s = []
    s = 0
    path_idx2s.append(s)
    for idx in np.arange(1, len(path_x), 1):
        pos_cur = np.array([path_x[idx], path_y[idx]])
        pos_pre = np.array([path_x[idx-1], path_y[idx-1]])
        dis = np.linalg.norm(pos_cur - pos_pre)
        s = s + dis
        path_idx2s.append(s)
    return path_idx2s, path_x, path_y, path_heading, path_kappa

def cal_dynamic_state(
    relative_time: float, 
    t_set: List[float], 
    s_set: List[float], 
    s_dot_set: List[float], 
    s_2dot_set: List[float]):
    """
    采用三次多项式关系，根据relative_time 和 speed planning 计算 velocity accelerate TODO:有空仔细推导一下
    :param relative_time
    :param t_set
    :param s_set
    :param s_dot_set
    :param s_2dot_set
    :return s
    :return velocity
    :return accelerate
    """
    idx_l = 0
    idx_r = 0
    for idx in range(len(t_set)-1):
        if t_set[idx+1] > relative_time:
            idx_l = idx
            idx_r = idx + 1
            break
    if idx_l == idx_r: # relative_time maybe equals t_set[-1]
        idx_l = len(t_set)-2
        idx_r = len(t_set)-1

    delta_t = relative_time - t_set[idx_l]
    s = s_set[idx_l] + s_dot_set[idx_l]*delta_t + (1/3)*s_2dot_set[idx_l]*(delta_t**2) + (1/6)*s_2dot_set[idx_r]*(delta_t**2)
    s_dot = s_dot_set[idx_l] + 0.5*s_2dot_set[idx_l]*delta_t + 0.5*s_2dot_set[idx_r]*delta_t
    s_dot2 = s_2dot_set[idx_l] + (s_2dot_set[idx_r] - s_2dot_set[idx_l])*delta_t/(t_set[idx_r] - t_set[idx_l])

    return s, s_dot, s_dot2

def cal_pose(
    s: float, 
    path_idx2s: List[float], 
    path_x: List[float], 
    path_y: List[float], 
    path_heading: List[float], 
    path_kappa: List[float]) -> Tuple[float, float, float, float]:
    """
    采用一维插值，根据当前时间下的s 和 规划融合的结果 计算 x y heading kappa
    """
    f_x = interp1d(path_idx2s, path_x)
    f_y = interp1d(path_idx2s, path_y)
    f_heading = interp1d(path_idx2s, path_heading)
    f_kappa = interp1d(path_idx2s, path_kappa)
    x = f_x(s)
    y = f_y(s)
    heading = f_heading(s)
    kappa = f_kappa(s)


    return x, y, heading, kappa
