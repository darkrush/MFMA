from . import core
from multiprocessing import Process, Queue, Manager
import threading
import time

class MSE_backend(object):
    def __init__(self,scenario,cfg=None):
        self.agent_groups = scenario['agent_groups']
        self.cfg = cfg
        self.fps = 100.0
        self.manager = Manager()
        self.manager_dict = self.manager.dict()
        self.manager_dict['run'] = 'pause'
        self.common_queue = Queue(maxsize = 5)
        self.data_queue = Queue(maxsize = 5)
        self.sub_process = Process(target = self.start_process,args = (self.manager_dict,self.common_queue,self.data_queue))
        self.sub_process.start()

    def close(self):
        self.common_queue.put(['kill',None])
        self.sub_process.join()

    def _lock(self,t,lock):
        lock.acquire(0)
        time.sleep(t)
        lock.release()

    def start_process(self,manager_dict,common_queue,data_queue):
        self.world = core.World(self.agent_groups)
        fps_lock = threading.Lock()
        while True:
            if not common_queue.empty():
                try:
                    item = common_queue.get_nowait()
                except :
                    print('this should not happend')
                if item[0] == 'reset':
                    self.world.reset()
                elif item[0] == 'kill':
                    break
                elif item[0] == 'get_state':
                    data_queue.put(self.world.get_state())
                elif item[0] == 'get_obs':
                    data_queue.put(self.world.get_obs())
                elif item[0] == 'set_state':
                    self.world.set_state(item[1])
                elif item[0] == 'set_action':
                    self.world.set_action(item[1])
            if manager_dict['run'] == 'pause':
                continue
            elif manager_dict['run'] == 'run':
                if fps_lock.acquire(0):
                    fps_lock.release()
                    t = threading.Thread(target=self._lock,args=(1.0/self.fps,fps_lock))
                    t.start()
                    self.world.step()
                    self.world.render()

    def reset(self):
        self.common_queue.put(['reset',None])

    def get_state(self):
        self.common_queue.put(['get_state',None])
        state = self.data_queue.get()
        return state

    def set_state(self,state):
        self.common_queue.put(['set_state',state])

    def get_obs(self):
        self.common_queue.put(['get_obs',None])
        obs = self.data_queue.get()
        return obs

    def set_action(self,action):
        self.common_queue.put(['set_action',action])

    def pause(self):
        self.manager_dict['run'] = 'pause'

    def go_on(self):
        self.manager_dict['run'] = 'run'

    def get_result(self):
        raise NotImplementedError