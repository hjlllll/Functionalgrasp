'''
使用说明：
定位想要查看的抓取文件，包括网络生成和标签xml文件（一般同时存放在result文件夹下的训练文件中，比如：./results/A-pre-[gen_fea-loss123-ang-kp-c-a]-[new]/xml/）
copy至文件夹'/home/lm/graspit/worlds'下，然后运行该程序即可
如果想要不同的显示模式，可参阅代码中注释掉的部分修改
'''
import os
import sys
import time
import signal
import psutil
from os.path import join
from multiprocessing import Process
from rewrite_xml import rewrite_xml

os.environ['GRASPIT'] = '/home/lm/graspit'

def get_pid(name):
    '''
     作用：根据进程名获取进程pid
    '''
    pids = psutil.process_iter()
    now_pid = []
    print("[" + name + "]'s pid is:")
    for pid in pids:
        if(pid.name() == name):
            print(pid.pid)
            now_pid.append(pid.pid)
    return now_pid

def run(name, xml_name):
    os.system('/home/lm/graspit/build/graspit_simulator --world {}'.format(xml_name))


worlds_path = '/home/lm/graspit/worlds'

for sub_name in sorted(os.listdir(worlds_path)):#Sorted()函数：使用数据中进行排序，而非使用前
    if sub_name.endswith('.xml'):
        sub_path = join(worlds_path, sub_name)
        print(sub_path)
        rewrite_xml(sub_path)

is_auto = False #change False True
if is_auto:
    for sub_name in sorted(os.listdir(worlds_path)):
        if sub_name.endswith('.xml'):
            if not sub_name[-9:] == 'label.xml':
                xml_name = os.path.splitext(sub_name)[0]
                p1=Process(target=run, args=('anne', xml_name)) #必须加,号
                p2=Process(target=run, args=('alice', xml_name+'_label'))
                p1.start()
                p2.start()
                time.sleep(5)
                aaa = get_pid("graspit_simulator")
                for pid in aaa:
                    os.kill(pid, signal.SIGKILL)
else:
    for sub_name in sorted(os.listdir(worlds_path)):
        if sub_name.endswith('.xml'):
            if not sub_name[-9:] == 'label.xml':
                xml_name = os.path.splitext(sub_name)[0]
                p1=Process(target=run, args=('anne', xml_name)) #必须加,号
                p1.start()
                os.system('/home/lm/graspit/build/graspit_simulator --world {}'.format(xml_name+'_label'))

# def run(name):
#     print('%s runing' %name)
#     os.environ['GRASPIT'] = '/home/lm/graspit'
#     worlds_path = '/home/lm/graspit/worlds'
#     for sub_name in sorted(os.listdir(worlds_path)):
#         if sub_name.endswith('.xml'):
#             os.system('/home/lm/graspit/build/graspit_simulator --world {}'.format(os.path.splitext(sub_name)[0]))
#     time.sleep(10)
#     print('%s running end' %name)
#
# p1=Process(target=run,args=('anne',)) #必须加,号
# p2=Process(target=run,args=('alice',))
# p1.start()
# p2.start()
# print('主线程')


# 直接调用的方式************************************************************
# def run(name):
#     print('%s runing' %name)
#     time.sleep(random.randrange(1,5))
#     print('%s running end' %name)

# p1=Process(target=run,args=('anne',)) #必须加,号
# p2=Process(target=run,args=('alice',))

#
# p1.start()
# p2.start()
# print('主线程')
# 直接调用的方式************************************************************

# 继承的方式调用************************************************************
# class Run(Process):
#     def __init__(self,name):
#         super().__init__()
#         self.name=name
#     def run(self):
#         print('%s runing' %self.name)
#         time.sleep(random.randrange(1,5))
#         print('%s runing end' %self.name)
#
# p1=Run('anne')
# p2=Run('alex')
# p3=Run('ab')
# p4=Run('hey')
# p1.start() #start会自动调用run
# p2.start()
# p3.start()
# p4.start()
# print('主线程')
# 继承的方式调用************************************************************