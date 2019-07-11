from  gym import Env
import copy
import numpy as np
import math
import time
from .basic import Agent,Action,AgentState

class MultiFidelityEnv(Env):
    def __init__(self,senario_dict,backend):
        self.backend = backend
        self.senario_dict = senario_dict
        self.time_limit = senario_dict['common']['time_limit']
        self.reward_coef = senario_dict['common']['reward_coef']
        self.reset_mode = senario_dict['common']['reset_mode']
        self.field_range = senario_dict['common']['field_range']
        self.ref_state_list = []
        self.total_time = 0
        for (_,grop) in self.senario_dict['agent_groups'].items():
            for agent_prop in grop:
                agent = Agent(agent_prop)
                state = AgentState()
                state.x = agent.init_x
                state.y = agent.init_y
                state.theta = agent.init_theta
                state.vel_b = agent.init_vel_b
                state.movable = agent.init_movable
                state.phi = agent.init_phi
                state.target_x = agent.init_target_x
                state.target_y = agent.init_target_y
                state.crash = False
                state.reach = False    
                self.ref_state_list.append(state)

    def _random_state(self):
        state = AgentState()
        state.x = np.random.uniform(self.field_range[0],self.field_range[1])
        state.y = np.random.uniform(self.field_range[2],self.field_range[3])
        state.target_x = np.random.uniform(self.field_range[0],self.field_range[1])
        state.target_y = np.random.uniform(self.field_range[2],self.field_range[3])
        state.theta = np.random.uniform(0,math.pi*2)
        state.vel_b = 0
        state.phi = 0
        return state

    def _reset_state(self):
        state_list=copy.deepcopy(self.ref_state_list)
        if self.reset_mode == 'random':
            for idx in range(len(state_list)):
                state_list[idx] = self._random_state()
        return state_list

    def _calc_reward(self):
        reward_list = []
        for new_state,old_state in zip(self.new_state,self.old_state):
            re = self.reward_coef['time_penalty']
            if new_state.crash:
                re += self.reward_coef['crash']
            if new_state.reach:
                re += self.reward_coef['reach']
            new_dist = ((new_state.x-new_state.target_x)**2+(new_state.y-new_state.target_y)**2)**0.5
            old_dist = ((old_state.x-old_state.target_x)**2+(old_state.y-old_state.target_y)**2)**0.5
            re += self.reward_coef['potential'] * (old_dist-new_dist)
            reward_list.append(re)
        return reward_list
    
    def get_state(self):
        return self.backend.get_state()

    def set_state(self,state,enable_list = None):
        if enable_list is None:
            enable_list = [True] * len(state)
        self.backend.set_state(state,enable_list)
    
    def bankend_pause(self):
        self.backend.pause()

    def bankend_go_on(self):
        self.backend.go_on()

    def step(self,action):
        self.backend.set_action(action)
        self.backend.go_on()
        time.sleep(0.01)
        self.backend.pause()
        self.new_state = self.backend.get_state()

        obs = self.backend.get_obs()
        reward = self._calc_reward()
        done = [not state.movable for state in self.new_state]
        if self.reset_mode == 'random':
            state_list  = copy.deepcopy(self.new_state)
            enable_list = [False] * len(state_list)
            enable_tmp = False
            for idx,state in enumerate(self.new_state):
                if state.reach :
                    state_list[idx].reach = False
                    state_list[idx].movable = True
                    state_list[idx].target_x = np.random.uniform(self.field_range[0],self.field_range[1])
                    state_list[idx].target_y = np.random.uniform(self.field_range[2],self.field_range[3])
                    enable_list[idx] = True
                    enable_tmp = True
                if state.crash :
                    state_list[idx].crash = False
                    state_list[idx].movable = True
                    state_list[idx].x = np.random.uniform(self.field_range[0],self.field_range[1])
                    state_list[idx].y = np.random.uniform(self.field_range[2],self.field_range[3])
                    enable_list[idx] = True
                    enable_tmp = True
            if enable_tmp :
                self.backend.set_state(state_list,enable_list)
        info = {'time_end': self.total_time>self.time_limit}
        self.old_state = copy.deepcopy(self.new_state)
        self.total_time+=1
        return obs,reward,done,info

    def reset(self):
        self.backend.pause()
        self.backend.reset()
        obs = self.backend.get_obs()
        self.old_state = self.backend.get_state()
        self.total_time = 0
        return obs

    def close(self):
        self.backend.close()