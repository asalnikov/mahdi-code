# -*- coding: utf-8 -*-

import datetime
import time

metric_short_description=\
        _("вычисляет среднюю заполненность вычислительной системы.")

metric_description=\
_("""
Средняя заполненность измеряется в процессорах. Она призвана показать, насколько много процессоров 
в среднем (на единицу времени) было использовано за весь период времени, когда запускались задачи 
из набора задач.
требует параметров:
    count_mode - возможные значения: (user, day, total)
    unit_time - единица времени
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

    def calc_average_congestion(self, tasks, unit_time, compr):
        """
            Returns average congestion.
        """
        events = []
        for task in tasks:
            events.append((time.mktime(task.time_start.timetuple()) * compr, task.required_cpus))
            events.append((time.mktime(task.time_end.timetuple()) * compr, -task.required_cpus))

        events = sorted(events)

        if len(events) == 0:
            return 0

        intervals, sum_area = 0, 0 
        prev_time = events[0][0]
        cur_procs = 0

        for event in events:
            t = (event[0] - prev_time) / unit_time
            intervals += t

            sum_area += t * cur_procs

            cur_procs += event[1]
            prev_time = int(event[0])
        
        return sum_area / intervals

    def count_values(self,compression):
        """
        Подсчитать и выдать число,
        словарь значений, и т.п.
        """
        mes=_("\n\n\trun metric %s:") % self.get_metric_name() 
        mes+=_("\tmetric parameters is: %s\n\n") % self.parameters
        print mes

        mode=self.parameters['count_mode']
        ut = self.parameters.get('unit_time', 1.0)
        tmp_result=dict()

        if mode == "user":
            users = {t.user_name for t in self.tasks_list}
            for u in users:
                tmp_result[u] = self.calc_average_congestion(list(filter(lambda x: x.user_name == u, self.tasks_list)), ut, compression)

        
        if mode == "day":
            dates = {t.time_submit.date() for t in self.tasks_list}
            for d in dates:
                tmp_result[d] = self.calc_average_congestion(list(filter(lambda x: x.time_submit.date() == d, self.tasks_list)), ut, compression)

        if mode == "total":
            tmp_result["total"] = self.calc_average_congestion(self.tasks_list, ut, compression)

        result=tmp_result

        return result

    def get_header_string(self):
        """
        Выдаём строку заголовок для печати всего в 
        .csv файл
        """
        mode=self.parameters['count_mode']
        if mode == "user":
            return "\"%s\"\t\"%s\"" % (_("Users"), _("Average congestion"))
        if mode == "day":
            return "\"%s\"\t\"%s\"" % (_("Date (YYYY-MM-DD)"), _("Average congestion"))
        if mode == "total":
            return "\"%s\"\t\"%s\"" % (_("Totally"), _("Average congestion"))

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
