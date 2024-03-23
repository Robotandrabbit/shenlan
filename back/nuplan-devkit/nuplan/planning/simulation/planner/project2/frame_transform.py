from typing import List, Tuple
import numpy as np
import math


def get_match_point(
    x_set: List[float],
    y_set: List[float],
    frenet_path_x: List[float],
    frenet_path_y: List[float]):
    """
    该函数用于获得给定点的匹配点（最近点）
    :param x_set,y_set 待坐标转换的点
    :param frenet_path_x,frenet_path_y   frenet坐标轴(参考线)
    :return match_point_index 匹配点在参考线的索引
    """
    match_point_index_set = []
    for idx in range(len(x_set)):
        cur_pos = np.array([x_set[idx], y_set[idx]])
        i = 0
        j = 1
        for _ in range(len(frenet_path_x)-1):
            di = np.sum((np.array([frenet_path_x[i], frenet_path_y[i]]) - cur_pos)**2)
            dj = np.sum((np.array([frenet_path_x[j], frenet_path_y[j]]) - cur_pos)**2)
            if dj >= di:
                break
            i += 1
            j += 1
        match_point_index_set.append(i)
    return match_point_index_set

def cal_project_point(
    x_set: List[float],
    y_set: List[float],
    match_point_index_set: List[int],
    frenet_path_x: List[float],
    frenet_path_y: List[float],
    frenet_path_heading: List[float],
    frenet_path_kappa: List[float],
    frenet_path_s: List[float]
):
    """
    cartesian2frenet的辅助函数 通过match point计算project point
    :param
    :return proj_x_set 投影点的x
    :return proj_y_set 投影点的y
    :return proj_heading_set 投影点的heading
    :return proj_kappa_set 投影点的kappa
    :return proj_s_set 投影点的s
    """
    proj_x_set = []
    proj_y_set = []
    proj_heading_set = []
    proj_kappa_set = []
    proj_s_set = []
    for idx in range(len(x_set)):
        match_point_index = match_point_index_set[idx]
        match_point_x = frenet_path_x[match_point_index]
        match_point_y = frenet_path_y[match_point_index]
        match_point_heading = frenet_path_heading[match_point_index]
        match_point_kappa = frenet_path_kappa[match_point_index]
        vector_match_point = np.array([match_point_x, match_point_y])
        vector_match_point_direction = np.array([math.cos(match_point_heading), math.sin(match_point_heading)])
        vector_r = np.array([x_set[idx], y_set[idx]])
        vector_d = vector_r - vector_match_point
        ds = np.dot(vector_d, vector_match_point_direction)
        vector_proj_point = vector_match_point + ds * vector_match_point_direction
        proj_heading = match_point_heading + match_point_kappa * ds
        proj_kappa = match_point_kappa
        proj_x = vector_proj_point[0]
        proj_x_set.append(proj_x)
        proj_y = vector_proj_point[1]
        proj_y_set.append(proj_y)
        proj_heading_set.append(proj_heading)
        proj_kappa_set.append(proj_kappa)
        s = cal_proj_s(frenet_path_x, frenet_path_y, frenet_path_s, proj_x, proj_y, match_point_index)
        proj_s_set.append(s)
    return proj_x_set, proj_y_set, proj_heading_set, proj_kappa_set, proj_s_set

def cal_proj_s(
    frenet_path_x: List[float],
    frenet_path_y: List[float],
    frenet_path_s: List[float],
    proj_x: float,
    proj_y: float,
    match_point_index_of_proj: int
    ):
    proj_point = np.array([proj_x, proj_y])
    match_point = np.array([frenet_path_x[match_point_index_of_proj], frenet_path_y[match_point_index_of_proj]])
    
    vector_1 =  proj_point - match_point
    if match_point_index_of_proj < len(frenet_path_x)-1: # 匹配点不是最后一个点
        match_point_next = np.array([frenet_path_x[match_point_index_of_proj+1], frenet_path_y[match_point_index_of_proj+1]])
        vector_2 = match_point_next - match_point
    else:
        match_point_pre = np.array([frenet_path_x[match_point_index_of_proj-1], frenet_path_y[match_point_index_of_proj-1]])
        vector_2 = match_point - match_point_pre
    
    s = 0
    if np.dot(vector_1, vector_2) > 0:
        s = frenet_path_s[match_point_index_of_proj] + math.sqrt(np.dot(vector_1, vector_1))
    else:
        s = frenet_path_s[match_point_index_of_proj] - math.sqrt(np.dot(vector_1, vector_1))
    return s

    




def cartesian2frenet(
    x_set: List[float],
    y_set: List[float],
    vx_set: List[float],
    vy_set: List[float],
    ax_set: List[float],
    ay_set: List[float],
    frenet_path_x: List[float],
    frenet_path_y: List[float],
    frenet_path_heading: List[float],
    frenet_path_kappa: List[float],
    frenet_path_s: List[float]) -> Tuple[List[float], List[float], List[float], \
                                         List[float], List[float], List[float], List[float], List[float]]:
    """
    该函数将计算世界坐标系下的x_set,y_set上的点在frenet_path下的坐标s l (用匹配点近似投影点)
    :param x_set,y_set,vx_set,vy_set,ax_set,ay_set 待坐标转换点的位置、速度、加速度 （均定义在全局坐标系）
    :param frenet_path_x,frenet_path_y,frenet_path_heading,frenet_path_kappa,frenet_path_s  frenet坐标轴的信息(参考线)
    :return s_set
    :return l_set
    :return s_dot_set
    :return l_dot_set
    :return dl_set
    :return l_dot2_set
    :return s_dot2_set
    :return ddl_set
    """

    s_set = []
    l_set = []
    s_dot_set = []
    l_dot_set = []
    dl_set = []
    l_dot2_set = []
    s_dot2_set = []
    ddl_set = []
    # 计算待转换点的match point
    match_point_index_set = get_match_point(x_set, y_set, frenet_path_x, frenet_path_y)

    # 计算待转换点的project point
    proj_x_set, proj_y_set, proj_heading_set, proj_kappa_set, proj_s_set = cal_project_point(\
        x_set, y_set, match_point_index_set, frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s)
    s_set = proj_s_set

    # 根据投影点信息和自车速度矢量，计算frenet frame下的坐标
    for idx in range(len(x_set)):
        n_r = np.array([-math.sin(proj_heading_set[idx]), math.cos(proj_heading_set[idx])])
        r_h = np.array([x_set[idx], y_set[idx]])
        r_r = np.array([proj_x_set[idx], proj_y_set[idx]])
        l = np.dot((r_h - r_r), n_r)
        l_set.append(l)

        # 根据l、自车速度矢量、投影点，计算l_dot、s_dot
        v_h = np.array([vx_set[idx], vy_set[idx]])
        t_r = np.array([math.cos(proj_heading_set[idx]), math.sin(proj_heading_set[idx])])
        l_dot = np.dot(v_h, n_r)
        l_dot_set.append(l_dot)
        s_dot = np.dot(v_h, t_r)/(1 - proj_kappa_set[idx]*l)
        s_dot_set.append(s_dot) 

        # 根据l_dot、s_dot计算dl
        dl = 0
        if abs(vx_set[idx]) > 1e-1 or abs(vy_set[idx]) > 1e-1: # 低速数据有误差
            theta_e = math.atan2(vy_set[idx], vx_set[idx])
            theta_r = proj_heading_set[idx]
            delta_theta = theta_e - theta_r
            delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta)) # normalize to [-pi, pi]
            kappa_r = proj_kappa_set[idx]
            one_minus_cur_l = 1 - l*kappa_r
            dl = math.tan(delta_theta) * one_minus_cur_l
        # if abs(s_dot) >= 1e-1:
        #     dl = l_dot/s_dot
        dl_set.append(dl)

        a_h = np.array([ax_set[idx], ay_set[idx]])
        # 近似认为dkr/ds 为0 简化计算
        l_dot2 = np.dot(a_h, n_r) - proj_kappa_set[idx] * (1 - proj_kappa_set[idx] * l) * (s_dot**2)
        l_dot2_set.append(l_dot2)
        s_dot2 = (1/(1 - proj_kappa_set[idx] * l)) * (np.dot(a_h,  t_r) + 2 * proj_kappa_set[idx] * dl * (s_dot**2))
        s_dot2_set.append(s_dot2)
        ddl = 0
        if abs(vx_set[idx]) > 1e-1 or abs(vy_set[idx]) > 1e-1:
            kappa_e = abs((ay_set[idx]*vx_set[idx] - ax_set[idx]*vy_set[idx])/((vx_set[idx]**2 + vy_set[idx]**2)**1.5))
            ddl = -(kappa_r * dl) * math.tan(delta_theta) + one_minus_cur_l / math.cos(delta_theta) / math.cos(delta_theta) * \
            (kappa_e * one_minus_cur_l / math.cos(delta_theta) - kappa_r)
        # if s_dot**2 >= 1e-2:
        #     ddl = (l_dot2 - dl*s_dot2)/(s_dot**2)
        ddl_set.append(ddl)

    return s_set, l_set, s_dot_set, l_dot_set, dl_set, l_dot2_set, s_dot2_set, ddl_set

def local2global_vector(
    local_x: float,
    local_y: float,
    local_frame_heading: float) -> Tuple[float, float]:
    """
    根据local坐标系信息，将local坐标下的二维矢量转换到全局坐标系下
    param: local_x 待转换矢量x坐标
    param: local_y 待转换矢量y坐标
    param: local_frame_heading 全局坐标系原点heading
    """
    global_x = math.cos(local_frame_heading) * local_x - math.sin(local_frame_heading) * local_y
    global_y = math.sin(local_frame_heading) * local_x + math.cos(local_frame_heading) * local_y
    return global_x, global_y

def frenet2cartesian(
    s_set: List[float],
    l_set: List[float],
    dl_set: List[float],
    ddl_set: List[float],
    frenet_path_x: List[float],
    frenet_path_y: List[float],
    frenet_path_heading: List[float],
    frenet_path_kappa: List[float],
    frenet_path_s: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    该函数实现frenet frame到cartesian frame的转换
    :param s_set,l_set,dl_set,ddl_set 待坐标转换点在frenet frame中的状态
    :param frenet_path_x,frenet_path_y,frenet_path_heading,frenet_path_kappa,frenet_path_s  frenet坐标轴的信息(参考线)
    :return x_set
    :return y_set
    :return heading_set
    :return kappa_set
    """
    x_set = []
    y_set = []
    heading_set = []
    kappa_set = []
    for idx in range(len(s_set)):
        # 计算(s,l)在frenet坐标轴上的投影
        proj_x,proj_y,proj_heading,proj_kappa = cal_project_point_by_s(\
            s_set[idx], frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, frenet_path_s)
        nor = np.array([-math.sin(proj_heading), math.cos(proj_heading)])
        point = np.array([proj_x, proj_y]) + l_set[idx] * nor
        x_set.append(point[0])
        y_set.append(point[1])
        # heading = proj_heading + math.atan(dl_set[idx]/(1 - proj_kappa * l_set[idx]))
        heading = proj_heading + math.atan2(dl_set[idx], (1 - proj_kappa * l_set[idx]))
        heading = math.atan2(math.sin(heading), math.cos(heading)) # normalize to [-pi, pi]
        heading_set.append(heading)
        # 近似认为 kappa' = 0
        kappa = ((ddl_set[idx] + proj_kappa * dl_set[idx] * math.tan(heading_set[idx] - proj_heading)) * \
            (math.cos(heading_set[idx] - proj_heading)**2)/(1 - proj_kappa * l_set[idx]) + proj_kappa) * \
            math.cos(heading_set[idx] - proj_heading)/(1 - proj_kappa * l_set[idx])
        kappa_set.append(kappa)
    return x_set, y_set, heading_set, kappa_set

def cal_project_point_by_s(
    s: float,
    frenet_path_x: float,
    frenet_path_y : float,
    frenet_path_heading: float,
    frenet_path_kappa: float,
    frenet_path_s: float) -> Tuple[float, float, float, float]:
    """
    frenet2cartesian2辅助函数 通过frenet frame下的s坐标和frenent frame的坐标轴信息，计算frenet frame下 点在坐标轴上的投影点
    :param
    :return proj_x 投影点的x
    :return proj_y 投影点的y
    :return proj_heading 投影点的heading
    :return proj_kappa 投影点的kappa
    """
    # 先找匹配点的编号
    match_index = 0
    while frenet_path_s[match_index] < s:
        match_index = match_index + 1
    match_point = np.array([frenet_path_x[match_index], frenet_path_y[match_index]])
    match_point_heading = frenet_path_heading[match_index]
    match_point_kappa = frenet_path_kappa[match_index]
    ds = s - frenet_path_s[match_index]
    match_tor = np.array([math.cos(match_point_heading), math.sin(match_point_heading)])
    proj_point = match_point + ds * match_tor
    proj_heading = match_point_heading + ds * match_point_kappa
    proj_kappa = match_point_kappa
    proj_x = proj_point[0]
    proj_y = proj_point[1] 
    return proj_x, proj_y, proj_heading, proj_kappa

    
