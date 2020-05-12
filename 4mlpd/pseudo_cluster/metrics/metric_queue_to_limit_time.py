# -*- coding: utf-8 -*-

import datetime

metric_short_description=\
        _("вычисляет среднее отношение времени в очереди к запрошенному времени.")

metric_description=\
_("""
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


    def calc_queue_time_to_limit(self, tasks):
        """
            Returns sum of task.queue_time / task.time_limit for all tasks, divided by len(tasks)
        """
        result = 0.0

        for task in tasks:
            if task.time_limit == 0:
                continue

            queue_time = float((task.time_start - task.time_submit).total_seconds())
            time_limit = task.time_limit * 60 # in seconds for better precision

            result += queue_time / time_limit

        result /= len(tasks)

        return result

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
                tmp_result[u] = self.calc_queue_time_to_limit(list(filter(lambda x: x.user_name == u, self.tasks_list)))

        
        if mode == "day":
            dates = {t.time_submit.date() for t in self.tasks_list}
            for d in dates:
                tmp_result[d] = self.calc_queue_time_to_limit(list(filter(lambda x: x.time_submit.date() == d, self.tasks_list)))

        if mode == "total":
            tmp_result["total"] = self.calc_queue_time_to_limit(self.tasks_list)

        result=tmp_result

        return result

    def get_header_string(self):
        """
        Выдаём строку заголовок для печати всего в 
        .csv файл
        """
        mode=self.parameters['count_mode']
        if mode == "user":
            return "\"%s\"\t\"%s\"" % (_("Users"), _("Wait time / time limit"))
        if mode == "day":
            return "\"%s\"\t\"%s\"" % (_("Date (YYYY-MM-DD)"), _("Wait time / time limit"))
        if mode == "total":
            return "\"%s\"\t\"%s\"" % (_("Totally"), _("Wait time / time limit"))

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
            return "\"%s\"\t%f" % (key, values_row)
        if mode == "day":
            return "\"%s\"\t%f" % (key.strftime("%Y-%m-%d"), values_row)
        if mode == "total":
            return "\"%s\"\t%f" % (key, values_row)

        return None
