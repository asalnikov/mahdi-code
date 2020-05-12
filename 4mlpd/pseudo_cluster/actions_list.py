# -*- coding: utf-8 -*-

import os
import pwd
import grp
import sys
import gettext

import extended_task

def prepare_child_to_run(extended_task_record, pipe, command_line):
    """
    Подготавливает сыновий процесс к запуску в нём 
    команды
    """
    #print "uid=%d, euid=%d, gid=%d, egid=%d, " % (
    #        os.getuid(),
    #        os.geteuid(),
    #        os.getgid(),
    #        os.getegid()
    #        )

    os.close(pipe[0])
    os.dup2(pipe[1],1)
    os.close(pipe[1])

    os.setpgrp()

    try:
        group_touple=grp.getgrnam(extended_task_record.group_name)
    except KeyError, e:
        print _("Group '%s' is not found in operating system")\
                            % extended_task_record.group_name
        sys.exit(2)
    gid=int(group_touple[2])    
    
    group_list=[]
    group_list.append(gid)
    os.setgroups(group_list)

   
    try:
        os.setgid(gid)
    except OSError, e:
        print _("Can't change gid from %d to %d:")\
                % (os.getgid(), gid)
        print e
        sys.exit(3)
 
    try:
        os.setegid(gid)
    except OSError, e:
        print _("Can't change effective gid from %d to %d:")\
                % (os.getegid(), gid)
        print e
 
        sys.exit(3)

    try:
        user_touple=pwd.getpwnam(extended_task_record.user_name)
    except KeyError, e:
        print _("User '%s' is not found in operating system")\
                % extended_task_record.user_name
        sys.exit(2)
    uid=int(user_touple[2])
  
        
    try:
       os.setuid(uid)
    except OSError, e:
        print _("Can't change uid from %d to %d:")\
                % (os.getuid(), uid)
        print e
        sys.exit(3)

    try:
        os.seteuid(uid)
    except OSError, e:
        print _("Can't change effective uid from %d to %d:")\
                % (os.geteuid(), uid)
        print e
        sys.exit(3)

 
    try:
        os.execvp(command_line[0],command_line)
    except OSError, e:
        print e
        sys.exit(4)

    return True

def print_output(file_pointer,extended_task_record,line):
    """
    Печать вывода от процесса ставящего/убирающего 
    задачу в очередь/из очереди
    """
    while line !="":
        print "\t\tTASK '%s'|'%s': %s"\
                        % ( 
                             extended_task_record.job_id,
                             extended_task_record.job_name, 
                             line
                          )
        line=file_pointer.readline()
    file_pointer.close()


class Scheduled_action(object):
    """
    Одиночное действие, которое нужно произвести с 
    системой ведения очередей.
    """
    def __init__(self,extended_task_record,action):
        self.extended_task_record = extended_task_record
        self.action               = action

    def __str__(self):
        s="class Scheduled_action(object):"
        s+="action='%s', " % self.action
        s+="extended_task_record='%s'" % str(self.extended_task_record)
        return s

    def submit_task(self,time_compression):
        """
        Ставит задачу в очередь c учётом коэффициента компрессии времени
        """
        time_limit=self.extended_task_record.time_limit
        if time_limit >0:
            time_limit=round(time_limit/float(time_compression))
            if time_limit == 0:
                time_limit=1
        
        duration=self.extended_task_record.time_end-self.extended_task_record.time_start
        
        s=self.extended_task_record.get_submit_string(time_limit,duration.total_seconds())
        #
        # Порождение процесса. 
        #
        pipe=os.pipe()
        pid=os.fork()
        if pid == 0:
            prepare_child_to_run(self.extended_task_record,pipe,s)
        #
        # father
        #
        os.close(pipe[1])
        f=os.fdopen(pipe[0],"r")
        line=f.readline()
        if not self.extended_task_record.parse_task_id(f,line):
            print_output(f,self.extended_task_record,line)
        else:
            f.close()

        pid,status = os.wait()
        if os.WIFEXITED(status) and (os.WEXITSTATUS(status) == 0):
            pass
        else:
            print _("-- Submitting for task ID '%s'|'%s' failed --")\
                    % ( 
                            self.extended_task_record.job_id,
                            self.extended_task_record.job_name
                      )
            print "-- %s --" % str(s)
            return False
        return True
       
    def cancel_task(self):
        """
         Принудительно завершает задачу
        """
        s=self.extended_task_record.get_cancel_string()
        pipe=os.pipe()
        pid=os.fork()
        if pid == 0:
            prepare_child_to_run(self.extended_task_record,pipe,s)
        #
        # father
        #
        os.close(pipe[1])
        f=os.fdopen(pipe[0],"r")
        print_output(f,self.extended_task_record,_("-- Cancel bellow --"))
        pid, status =os.wait()
        if os.WIFEXITED(status) and (os.WEXITSTATUS(status) == 0):
            pass
        else:
            print _("-- Canceling task '%s'|'%s' with actual ID '%s' failed --")\
                    % ( self.extended_task_record.job_id,
                        self.extended_task_record.job_name,
                        self.extended_task_record.actual_task_id
                      )
            print "-- %s --" % str(s)
            return False
        return True            

class Action_list(object):
    """
    Класс для хранения списка действий, которые нужно произвести 
    на один сеанс взаимодействия с системой ведения очередей.
    """
    def __init__(self):
        self.actions_list=list()

    def register_action(self,extended_task_record,action):
        """
         Регистрирует новое действие
        """ 
        action=Scheduled_action(extended_task_record,action)
        self.actions_list.append(action)

    def do_actions(self,time_compression):
        """
            Производит все действия, которые есть в списке
        """
        for action in self.actions_list:
            if   action.action == "submit":
                    action.submit_task(time_compression)
            elif action.action == "cancel":
                    action.cancel_task()
            #print action
        self.actions_list=list()

