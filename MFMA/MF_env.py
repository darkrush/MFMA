from  gym import Env

class Backend(object):
    def __init__(self,agent_groups,cfg):
        
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self,state = None):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def set_action(self,action = None):
        raise NotImplementedError

    def pause(self):
        raise NotImplementedError

    def go_on(self):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError

class MultiFidelityEnv(Env):
    def __init__(self,common,backend):
        self.backend = backend
        self.time_limit = common['time_limit']
        self.reward_coef = common['reward_coef']
    
    def _calc_reward(self):
        self.old_obs
        self.new_obs
        return 0

    def step(self,action):
        self.backend.set_action(action)
        self.new_obs = self.backend.get_obs()
        state_list = self.backend.get_state()
        reward = self._calc_reward()
        done = []
        for state in state_list:
            done.append(not state.movable)
        info = {}
        self.old_obs = self.new_obs
        return self.new_obs,reward,done,info

    def reset(self):
        self.backend.reset()
        self.backend.go_on()
        self.old_obs = self.backend.get_obs()
        return self.old_obs

    def close(self):
        self.backend.close()