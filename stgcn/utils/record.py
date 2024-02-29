import pdb
import time
import os
from torch.utils.tensorboard import SummaryWriter

class Recorder(object):
    def __init__(self, work_dir, use_tb, is_main_process = True):
        self.cur_time = time.time()
        self.log_path = '{}/log.txt'.format(work_dir)
        self.is_main_process = is_main_process
        self.use_tb = use_tb
        if self.use_tb and self.is_main_process:
            self.tbwriter = SummaryWriter(log_dir=work_dir)

    def print_log(self, str):
        if not self.is_main_process:
            return
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
        print(str)
        with open(self.log_path, 'a') as f:
            f.writelines(str)
            f.writelines("\n")
    
    def print(self, str):
        if not self.is_main_process:
            return
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
        print(str)
    def add_scalar(self, tag, scalar_value, step):
        if not self.is_main_process or not self.use_tb:
            return
        self.tbwriter.add_scalar(tag=tag, scalar_value = scalar_value, global_step = step)