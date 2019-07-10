from  gym import Env
import copy
class MultiFidelityEnv(Env):
    def __init__(self,senario_dict,backend):
        self.backend = backend
        self.time_limit = senario_dict['common']['time_limit']
        self.reward_coef = senario_dict['common']['reward_coef']
    
    def _calc_reward(self):
        reward_list = []
        for new_state,old_state in zip(self.new_state,self.old_state):
            re = 0 + self.reward_coef['time_penalty']
            if new_state.crash:
                re += self.reward_coef['crash']
            if new_state.reach:
                re += self.reward_coef['reach']
            new_dist = ((new_state.x-new_state.target_x)**2+(new_state.y-new_state.target_y)**2)**0.5
            old_dist = ((old_state.x-old_state.target_x)**2+(old_state.y-old_state.target_y)**2)**0.5
            re += self.reward_coef['potential'] * (old_dist-new_dist)
            reward_list.append(re)
        return 0
    
    def get_state(self):
        return self.backend.get_state()

    def set_state(self,state):
        self.backend.set_state(state)


    def step(self,action):
        self.backend.set_action(action)
        obs = self.backend.get_obs()
        self.new_time,self.new_state = self.backend.get_state()
        reward = self._calc_reward()
        done = []
        for state in self.new_state:
            done.append(not state.movable)
        info = {}
        self.old_state = copy.deepcopy(self.new_state)
        self.old_time = self.new_time
        return obs,reward,done,info

    def reset(self):
        self.backend.reset()
        self.backend.go_on()
        obs = self.backend.get_obs()
        self.old_time ,self.old_state = self.backend.get_state()
        return obs

    def close(self):
        self.backend.close()