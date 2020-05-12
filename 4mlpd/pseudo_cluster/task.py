# -*- coding: utf-8 -*-

import datetime
import copy

def get_header_string():
      """
        Печать заголовка, с которого начинается 
        файл со статистикой.
      """
      s=""
      s+="\"id\"\t"
      s+="\"name\"\t"
      s+="\"time_submit\"\t"
      s+="\"time_start\"\t"
      s+="\"time_end\"\t"
      s+="\"user\"\t"
      s+="\"group\"\t"
      s+="\"time_limit\"\t"
      s+="\"required_cpus\"\t"
      s+="\"partition\"\t"
      s+="\"priority\"\t"
      s+="\"task_class\"\t"
      s+="\"state\"\t"
      s+="other\n"
      
      return s

#
# Состояния здесь упорядоченны именно так,
# как они упоядоченны в slurm.
#
# Если необходимо добавить новое состояние, 
# то необходимо сделать 2 альтернативы:
# 1. Дописать состояние здесь в конце.
#
# 2. Переписать parse_slurm_db.py, 
#    который пользуется именно этот 
#    порядок, так чтобы всё было 
#    порядконезависимо. Тоесть добавить 
#    туда dict с отображением номера на
#    строку состояния.
#
TASK_STATES=\
    [
        "pending",
        "running",
        "suspended",
        "completed",
        "canceled",
        "failed",
        "time_left",
        "node_fail"
    ]

class Task_record(object):
    """
    Класс для хранения атрибутов задач, полученных
    из файла со статистикой.
    """
    def __init__(self):
        self.job_id=None
        self.job_name=None
        self.time_submit=None
        self.time_start=None
        self.time_end=None
        self.user_name=None
        self.group_name=None
        self.time_limit=None
        self.required_cpus=None
        self.partition=None
        self.priority=None
        self.task_class=None
        self.task_state=None

        #
        # Cловарь с прочими атрибутами, ассоциированными с задачей на кластере.
        # ключ - имя атрибута, например имя поля в таблице [cluster_name]_job_table
        #
        self.other= {}

    def __str__(self):
        s="Class Task_record(object): "    
        s+="job_id='%s', "         % self.job_id
        s+="job_name='%s', "       % self.job_name
        s+="time_submit='%s', "    % self.time_submit.strftime("%Y-%m-%d %H:%M")
        s+="time_start='%s',  "    % self.time_start.strftime("%Y-%m-%d %H:%M")
        s+="time_end='%s', "       % self.time_end.strftime("%Y-%m-%d %H:%M")
        s+="user_name='%s', "      % self.user_name
        s+="group_name='%s', "     % self.group_name
        s+="time_limit=%d, "       % self.time_limit
        s+="required_cpus=%d, "    % self.required_cpus
        s+="partition='%s', "      % self.partition
        s+="priority=%d, "         % self.priority
        s+="task_class='%s', "     % self.task_class
        s+="task_state='%s', "     % self.task_state
        s+="other= "+str(self.other)
        return s

    def __copy__(self):
        r=Task_record()
        r.job_id=self.job_id
        r.job_name=self.job_name
        r.time_submit=self.time_submit
        r.time_start=self.time_start
        r.time_end=self.time_end
        r.user_name=self.user_name
        r.group_name=self.group_name
        r.time_limit=self.time_limit
        r.required_cpus=self.required_cpus
        r.partition=self.partition
        r.priority=self.priority
        r.task_class=self.task_class
        r.task_state=self.task_state
        r.other=copy.copy(self.other)
        return r

    
    def print_record_to_file(self,file_pointer):
        """
        Печать информации о задаче в файл
        """
        s=""    
        s+="\"%s\"\t"    % self.job_id
        s+="\"%s\"\t"    % self.job_name
        s+="\"%s\"\t"    % self.time_submit.strftime("%Y-%m-%d %H:%M")
        s+="\"%s\"\t"    % self.time_start.strftime("%Y-%m-%d %H:%M")
        s+="\"%s\"\t"    % self.time_end.strftime("%Y-%m-%d %H:%M")
        s+="\"%s\"\t"    % self.user_name
        s+="\"%s\"\t"    % self.group_name
        s+="%d\t"        % self.time_limit
        s+="%d\t"        % self.required_cpus
        s+="\"%s\"\t"    % self.partition
        s+="%d\t"        % self.priority
        s+="\"%s\"\t"    % self.task_class
        s+="\"%s\"\t"    % self.task_state

        subs=""
        for k in self.other.keys():
            subs+="%s='%s', " % (k,self.other[k])
        subs=subs.strip(' ,')

        s+="\"%s\"\n"    % subs


        return file_pointer.write(s)

    def read_record_from_file(self,file_pointer):
        """
          Читает запись информацию о задаче из файла.
             в случае успеха возвращает True,
             иначе False.
        """
        try:
            row=file_pointer.readline()
        except:
            return False
        
        tupl=row.split('\t')
        if len(tupl) < 13:
            return False
        
        self.job_id        = tupl[0].strip('"')
        self.job_name      = tupl[1].strip('"')
        self.time_submit   = datetime.datetime.strptime(tupl[2].strip('"'),"%Y-%m-%d %H:%M")
        self.time_start    = datetime.datetime.strptime(tupl[3].strip('"'),"%Y-%m-%d %H:%M")
        self.time_end      = datetime.datetime.strptime(tupl[4].strip('"'),"%Y-%m-%d %H:%M")
        #self.time_submit   = tupl[2].strip('"')
        #self.time_start    = tupl[3].strip('"')
        #self.time_end      = tupl[4].strip('"')
        self.user_name     = tupl[5].strip('"')
        self.group_name    = tupl[6].strip('"')
        self.time_limit    = int(tupl[7])
        self.required_cpus = int(tupl[8])
        self.partition     = tupl[9].strip('"')
        self.priority      = int (tupl[10])
        self.task_class    = tupl[11].strip('"')
        self.task_state    = tupl[12].strip('"')
        other_string       = tupl[13].strip('"\n\r')
        if other_string != "":
            #print "string='%s'" % other_string
            for item in other_string.split(','):
               pair=item.strip().split('=')
               self.other[pair[0]]=pair[1].strip("'\"")
            

        return True
