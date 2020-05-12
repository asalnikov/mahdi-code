# -*- coding: utf-8 -*-

import copy

import task

class Tasks_list(object):
    """
        Класс для хранения и вывода информации о 
        статистике прохождения задач через систему 
        ведения очередей кластера
    """
    def __init__(self):
         self.user_name_pattern="pseudo_cluster_user_"
         self.group_name_pattern="pseudo_cluster_group_"
         self.last_user_id=1
         self.last_group_id=1
         #
         #   Множество отображений: 
         #   пользователь на кластере --> порядковый номер, 
         #   как он встречается  в статистике.
         #
         self.users_map={}
         #
         #   Множество отображений: 
         #   группа на кластере --> порядковый номер,
         #   как он встречается  в статистике.
         #
         self.groups_map={}
         #
         # Включённость пользователя в группу
         # словарь множеств.
         #
         self.user_groups_relations={}
         self.real_user_fake_groups={}
         self.fake_user_real_groups={}
         #
         #  список задач.
         #
         self.tasks_list=[]

    def get_internal_user_id(self,user):
        """
          Возвращает уникальный идентификатор пользователя, 
          или если его ещё нет в словаре пользователей
          добавить его туда.
        """
        internal_id=None
        if user in self.users_map.keys():
            internal_id=self.users_map[user]
        else:
            internal_id=self.last_user_id
            self.last_user_id+=1
            self.users_map[user]=internal_id
        
        return internal_id

    def get_internal_group_id(self,group):
        """
          Возвращает уникальный идентификатор группы, 
          или если его ещё нет в словаре групп
          добавить его туда.
        """
        internal_id=None
        try:
            internal_id=self.groups_map[group]
        except KeyError:
            internal_id=self.last_group_id
            self.last_group_id+=1
            self.groups_map[group]=internal_id
        
        return internal_id

    def register_user_in_group(self, user_id, group_id, internal_user=True, internal_group=True):
        """
            Добавляет ассоциацию пользователь группа,
            если такой ассоциации до этого небыло
        """
        if internal_user and internal_group:
            groups_set=self.user_groups_relations.get(user_id)
            if groups_set == None:
                groups_set=set()
            if group_id not in groups_set:
                groups_set.add(group_id)
            #TODO
            #
            # Возможно в этом месте оно будет копировать
            # множества, вместо того, чтобы 
            # если множестсва одни и те же, оставить как есть. 
            #
            self.user_groups_relations[user_id]= groups_set

        if not internal_user and internal_group:
            groups_set=self.real_user_fake_groups.get(user_id)
            if groups_set == None:
                groups_set=set()
            if group_id not in groups_set:
                groups_set.add(group_id)
            self.real_user_fake_groups[user_id] = groups_set
            
        if internal_user and not internal_group:
            groups_set=self.fake_user_real_groups.get(user_id)
            if groups_set == None:
                groups_set=set()
            if group_id not in groups_set:
                groups_set.add(group_id)
            self.fake_user_real_groups[user_id] = groups_set

    def get_user_name_by_id(self,user_id):
        """
            получает имя пользователя в соответствии 
            с образцом по его идентификатору
        """
        return self.user_name_pattern+str(user_id)

    def get_group_name_by_id(self,group_id):
        """
            получает имя группы в соответствии с 
            образцом по её идентификатору
        """
        return self.group_name_pattern+str(group_id)


    def add_task_record(self,record):
        """
            Добавляет запись о задаче в список записей
        """
        #TODO Вставить сюда проверку соответствия типов
        #
        self.tasks_list.append(record)    

    def print_to_files(self, file_system_prefix):
        """
         Печатает всё в файловую систему
        """
        f=open(file_system_prefix+"statistics.csv","w")
        f.write(task.get_header_string())
        for tsk in self.tasks_list:
            tsk.print_record_to_file(f)
        f.close()
        
        f=open(file_system_prefix+"users_map","w")
        for k,v in self.users_map.items():
            f.write("%s:%s\n" % (self.get_user_name_by_id(v),k))
        f.close()

        f=open(file_system_prefix+"groups_map","w")
        for k,v in self.groups_map.items():
            f.write("%s:%s\n" % (self.get_group_name_by_id(v),k))
        f.close()

        f=open(file_system_prefix+"user_in_groups_map","w")
       
        internal_users_set=set()
        internal_users_set.update(self.user_groups_relations.keys())
        internal_users_set.update(self.fake_user_real_groups.keys())

        for user_id in internal_users_set:
            s=self.get_user_name_by_id(user_id)+":"
            groups=self.user_groups_relations.get(user_id)
            if groups:
                for group_id in groups:
                    s+=self.get_group_name_by_id(group_id)+","
            groups=self.fake_user_real_groups.get(user_id)
            if groups:
                for group in groups:
                    s+=group+","
            f.write("%s\n" % s.strip(" ,"))
        for user,group_id in self.real_user_fake_groups.items():
            s=user+":"
            for group_id in groups:
                s+=self.get_group_name_by_id(group_id)+","
            f.write("%s\n" % s.strip(" ,"))

        f.close()

    def read_statistics_from_file(self,file_system_prefix):
        f=open(file_system_prefix+"statistics.csv","r")
        f.readline()
        tsk=task.Task_record()
        while tsk.read_record_from_file(f):
            self.tasks_list.append(copy.copy(tsk))
            #print tsk
        f.close()
    
    def __getitem__(self, item):
        return self.tasks_list[item]

    def __len__(self):
        return len(self.tasks_list)
        

