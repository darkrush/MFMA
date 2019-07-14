# gym-MFMAs

## basic.py
### Class AgentState
* 定义了Agent的所有状态包括
  * __float__ x, y 车的中心点坐标
  * __float__ vel_b 车后点线速度
  * __float__ theta 车头方向
  * __float__ phi 车前轮偏角
  * __boolean__ crash,reach 车是否已碰撞,已到达
  * __boolean__ movable 车是否可以移动
  * __float__ target_x,target_y 目标点的坐标

### Class Action
* 定义了Agent的动作量
  * __float__ ctrl_vel 后轮电机的控制量
  * __float__ ctrl_phi 前轮偏角的控制量
### Class Observation
* 定义了观测量Observation
  * __list__ pos 列表内元素依次为x,y,theta,target_x,target_y 
  * __list__ laser_data 列表内为 N_laser个\[0,R_laser\]内的浮点数
### Class Agent
* 定义了Agent的所有属性
  * __float__ R_safe 车辆安全半径
  * __float__ R_reach 车辆到达判定距离
  * __float__ L_car   车辆外壳长
  * __float__ W_car   车辆外壳宽
  * __float__ L_axis  车辆外壳长
  * __float__ R_laser 激光雷达半径
  * __int__ N_laser 激光雷达分辨率（360°范围内均匀分布N根雷达线）
  * __float__ K_vel 后轮速度控制系数
  * __float__ K_phi 前轮偏角控制系数     
  * __float__ init_x 初始x位置 
  * __float__ init_y 初始y位置
  * __float__ init_theta 初始车头方向
  * __float__ init_vel_b 初始车后点线速度
  * __float__ init_phi   初始车前轮偏角
  * __boolean__ init_movable 初始车可移动状态
  * __float__ init_target_x,init_target_y 初始车目标点坐标
******

## make_env.py
### parse_senario(senario_file)
* 输入:senario_file为文件路径,会将指定文件解析成字典类型并返回
* 返回:解析后的scenario字典
* 如果键值以 'deg_' 开头,会去掉'deg_'前缀并将值从角度值转化为弧度制
* 返回值主要包含'common','agent_groups'两个键值。
  * agent_groups下包括多个group键值,每个group中包含多个agent的信息。
  * common的内容用来定义环境的其他内容
### make_env(name,senario,cfg)
* 输入:scenario为解析后的senario字典,cfg为可能需要的配置信息。
* 返回:构造好的环境
******

## MKF_env.py
### class MultiFidelityEnv(Env)
### \_\_init__(self,senario_dict,backend)
* 输入: senario_dict为解析后的scenario字典,backend为构造好的仿真器后端,构造一个用于强化学习的环境
### get_state(self)
* 返回:当前系统状态,\[AgentState * n\]
### set_state(self,state,enable_list = None):
* 输入:state:\[ AgentState * n\] 需要设置的状态,enable_list\[boolean * n\]
* 当enable_list[i] == True 时,对应的 i号 agent 状态被设置为state\[i\]
* enable_list == None 时，全部有效
### bankend_pause(self)
* 暂停后端的运行
### bankend_go_on(self)
* 继续后端的运行
### step(self,action)
* 输入:action: \[\[vel,phi\] * n\]
* 将第i个agent的运动控制量设置为action\[i\]
### reset(self)
* 重置环境到初始状态，并重置统计数据
### get_result(self)
* 返回统计数据
### close(self):
关闭环境


## MSE/MSE_backend.py
### class MSE_backend(object)
### \_\_init__(self,scenario)
* 输入:scenario：与 make_env.py 中 parse_senario(senario_file)的返回值
### get_state(self)
* 返回：所有agent的状态\[ AgentState * n\]
### set_state(self,state,enable_list = None)
* 输入：state：所有agent的状态\[ AgentState * n\]，enable_list\[boolean * n\] 
* 如果enable_list = None，则全部有效
### get_obs(self)
* 返回：当前所有agent的观测
* 返回值是一个字典{'time':time,'obs_data':\[Observation * n\]}
### set_action(self,actions,enable_list= None)
* 输入：actions：所有agent的动作\[ Action * n \]，enable_list\[boolean * n\]
* 当enable_list[i] == True 时,对应的 i号 agent 动作被设置为actions\[i\]
* 如果enable_list = None，则全部有效
### pause(self)
* 暂停仿真器后端
### go_on(self)
* 继续仿真器后端
### close(self)
* 关闭仿真器后端