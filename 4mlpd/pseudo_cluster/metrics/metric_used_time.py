# -*- coding: utf-8 -*-

import datetime

metric_short_description=\
        _("вычисляет степень разумности запрашиваемого времени исполнения.")

metric_description=\
_("""
Отношение времени выполнения задачи к запрошенному пользователем
времени (для успешно законченных задач).
требует параметров:
    count_mode - возможные значения: (user, day, total)
""")

class Metric_counter(object):
    """
        Класс задающий метрику
    """

    def __init__(self,tasks_list,parameters):
        """
        Собственно конструирование объекта
        """
        self.tasks_list=tasks_list
        self.parameters=dict()
        if parameters != "":
            print parameters
            for item in parameters.split(','):
                pair=item.strip().split('=')
                self.parameters[pair[0]]=pair[1].strip("'\"")

        if "count_mode" not in self.parameters.keys():
            self.parameters["count_mode"]="user"

    def __str__(self):
        s="package %s: Metric_counter: " % __name__
        s+="tasks_list=%s, " % str(self.tasks_list)
        s+="parameters=%s " % str(self.parameters)
        return s

    def get_metric_name(self):
        return __name__

    def calc_used_time(self, tasks):
        """
            Returns sum of task.running_time / task.time_limit for completed task in 'tasks',
            divided by number of completed tasks.
        """
        success = 0
        time_portion = 0
        aborted = 0

        for task in tasks:
            if task.task_state.strip().lower() != 'completed':
                aborted += 1
            else:
                time_portion += float((task.time_end - task.time_start).total_seconds())\
                        / (task.time_limit * 60)
                success += 1

        if success > 0:
            time_portion /= success

        return (time_portion, aborted)

    def count_values(self,compression):
        """
        Подсчитать и выдать число,
        словарь значений, и т.п.
        """
        mes=_("\n\n\trun metric %s:") % self.get_metric_name() 
        mes+=_("\tmetric parameters is: %s\n\n") % self.parameters
        print mes

        mode=self.parameters['count_mode']
        tmp_result=dict()

        if mode == "user":
            users = {t.user_name for t in self.tasks_list}
            for u in users:
                tmp_result[u] = self.calc_used_time(list(filter(lambda x: x.user_name == u, self.tasks_list)))

        
        if mode == "day":
            dates = {t.time_submit.date() for t in self.tasks_list}
            for d in dates:
                tmp_result[d] = self.calc_used_time(list(filter(lambda x: x.time_submit.date() == d, self.tasks_list)))

        if mode == "total":
            tmp_result["total"] = self.calc_used_time(self.tasks_list)

        result=tmp_result

        return result

    def get_header_string(self):
        """
        Выдаём строку заголовок для печати всего в 
        .csv файл
        """
        mode=self.parameters['count_mode']
        if mode == "user":
            return "\"%s\"\t\"%s\"\t\"%s\"" % (_("Users"), _("Used portion of time limit"), _("Number of aborted tasks"))
        if mode == "day":
            return "\"%s\"\t\"%s\"\t\"%s\"" % (_("Date (YYYY-MM-DD)"), _("Used portion of time limit"), _("Number of aborted tasks"))
        if mode == "total":
            return "\"%s\"\t\"%s\"\t\"%s\"" % (_("Totally"), _("Used portion of time limit"), _("Number of aborted tasks"))

        return None

    def get_draw_type(self):
        """
        Выдаёт:
            chart - если отображаемо как набор столбиков,
            plot - если кривая y=f(x)
        """
        mode=self.parameters['count_mode']
        if mode == "user":
            return "chart"
        if mode == "day":
            return "plot"

        return None



    def format_row(self,key,values_row):
        """
        Форматирует запись к виду пригодному для печати
        в .csv формате.
        """
        mode=self.parameters['count_mode']
        if mode == "user":
            return "\"%s\"\t%f\t%d" % (key, values_row[0], values_row[1])
        if mode == "day":
            return "\"%s\"\t%f\t%d" % (key.strftime("%Y-%m-%d"), values_row[0], values_row[1])
        if mode == "total":
            return "\"%s\"\t%f\t%d" % (key, values_row[0], values_row[1])

        return None
