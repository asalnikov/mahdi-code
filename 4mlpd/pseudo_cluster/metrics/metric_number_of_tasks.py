# -*- coding: utf-8 -*-

import datetime

metric_short_description=\
        _("вычисляет количество задач.")

metric_description=\
_("""
Количество задач показывает, насколько много используется
кластер.
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


    def count_values(self, compression):
        mes=_("\n\n\trun metric %s:") % self.get_metric_name() 
        mes+=_("\tmetric parameters is: %s\n\n") % self.parameters
        print mes

        mode=self.parameters['count_mode']
        tmp_result=dict()

        if mode == "user":
            for task in self.tasks_list:
                tmp_result[task.user_name] = tmp_result.get(task.user_name, 0) + 1 
        
        if mode == "day":
            for task in self.tasks_list:
                date=task.time_submit.date()
                tmp_result[date] = tmp_result.get(date, 0) + 1

        if mode == "total":
            tmp_result["total"] = len(self.tasks_list)

        result = tmp_result
        return result

    def get_header_string(self):
        """
        Выдаём строку заголовок для печати всего в 
        .csv файл
        """
        mode=self.parameters['count_mode']
        if mode == "user":
            return "\"%s\"\t\"%s\"" % (_("Users"), _("Number of tasks"))
        if mode == "day":
            return "\"%s\"\t\"%s\"" % (_("Date (YYYY-MM-DD)"), _("Number of tasks"))
        if mode == "total":
            return "\"%s\"\t\"%s\"" % (_("Totally"), _("Number of tasks"))

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
            return "\"%s\"\t%d" % (key, values_row)
        if mode == "day":
            return "\"%s\"\t%d" % (key.strftime("%Y-%m-%d"), values_row)
        if mode == "total":
            return "\"%s\"\t%d" % (key, values_row)

        return None







