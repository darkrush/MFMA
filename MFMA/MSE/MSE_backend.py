from . import core
from multiprocessing import Process, Queue, Manager
import threading
import time

class MSE_backend(object):
    def __init__(self,scenario):
        self.agent_groups = scenario['agent_groups']
        self.cfg = {'dt':scenario['common']['dt'],}
        self.use_gui = scenario['common']['use_gui']
        self.fps = 300.0
        self.manager = Manager()
        self.manager_dict = self.manager.dict()
        self.manager_dict['run'] = 'pause'
        self.common_queue = Queue(maxsize = 5)
        self.data_queue = Queue(maxsize = 5)
        self.sub_process = Process(target = self._start_process,args = (self.manager_dict,self.common_queue,self.data_queue))
        self.sub_process.start()

    def _fps_func(self,t,event):
        event.set()
        time.sleep(t)
        t = threading.Thread(target=self._fps_func,args=(t,event))
        t.start()
    def _render_func(self,event):
        event.set()
        time.sleep(1.0/30.0)
        t = threading.Thread(target=self._render_func,args=(event,))
        t.start()  

    def _start_process(self,manager_dict,common_queue,data_queue):
        self.world = core.World(self.agent_groups,self.cfg)
        if self.fps != 0:
            fps_event = threading.Event()
            fps_event.set()
            fps_t = threading.Thread(target=self._fps_func,args=(1.0/self.fps,fps_event))
            fps_t.start()
        if self.use_gui:
            render_event = threading.Event()
            render_event.set()
            render_t = threading.Thread(target=self._render_func,args=(render_event,))
            render_t.start()
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
                    self.world.set_state(item[1][0],item[1][1])
                elif item[0] == 'set_action':
                    self.world.set_action(item[1][0],item[1][1])
            if self.use_gui:
                if render_event.is_set():
                    render_event.clear()
                    self.world.render()
            if manager_dict['run'] == 'pause':
                continue
            elif manager_dict['run'] == 'run':
                if self.fps !=0:
                    if fps_event.is_set():
                        fps_event.clear()
                        self.world.step()
                else:
                    self.world.step()
    #def reset(self):
    #    self.common_queue.put(['reset',None])

    def get_state(self):
        self.common_queue.put(['get_state',None])
        state = self.data_queue.get()
        return state

    def set_state(self,state,enable_list = None):
        if enable_list is None:
            enable_list = [True]* len(state)
        self.common_queue.put(['set_state',(enable_list,state)])

    def get_obs(self):
        self.common_queue.put(['get_obs',None])
        obs = self.data_queue.get()
        return obs

    def set_action(self,actions,enable_list= None):
        if enable_list is None:
            enable_list = [True]* len(actions)
        self.common_queue.put(['set_action',(enable_list,actions)])

    def pause(self):
        self.manager_dict['run'] = 'pause'

    def go_on(self):
        self.manager_dict['run'] = 'run'

    def close(self):
        self.common_queue.put(['kill',None])
        self.sub_process.join()