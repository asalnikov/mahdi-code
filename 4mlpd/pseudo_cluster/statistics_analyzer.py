# -*- coding: utf-8 -*-

import imp
import sys
import gettext

import tasks_list
import metrics
import statistics_plotter


def metric_module_import(metric_name):
    """
    импортирует модуль, который 
    обеспечивает работу с метрикой.
    """
    try:
        #module=importlib("metrics.metric_"+metric_name)
        module=imp.load_source(
                    metric_name,
                    "%s/metric_%s.py" % (metrics.__path__[0], metric_name)
        )
    except IOError, e:
        print e
        sys.exit(3)
    
    return module


class Statistics_analyzer(object):
    """
    Класс для анализа статистики и её отображения
    в виде текстовых файлов.
     
    Поддерживает расчёт по некоторому набору метрик.
    """
    def __init__(self):
        """
        Конструктор
        """
        
        #
        #список задач
        #на котором будет 
        #вычисляться метрика
        #
        self.tasks_list = None
        
        #
        # Коэффициент сжатия времени
        #
        self.time_compression = 1
        
        #
        # Объект расчётчика метрики.
        # импортируется из 
        # соответствующего модуля.
        #
        self.metric_counter=None

        #
        # вычисленные значения
        #
        self.counted_values = None

    def get_metrics_list(self):
        """
        Получает список доступных в 
        текущий момент метрик
        """
        metrics_list=dict()
        for module_name in metrics.__all__:
           
            metric_name=module_name.partition('_')[2]
            module=metric_module_import(metric_name)
            metrics_list[metric_name]=module.metric_short_description

        return metrics_list


    def get_metric_description(self,metric_name):
        """
        Получает описание метрики по её имени
        """
        if "metric_%s" % metric_name not in  metrics.__all__:
            print _("Metric with name '%s' is not found") % metric_name
            sys.exit(3)

        module=metric_module_import(metric_name)
        return module.metric_description

    def register_metric_counter(self, metric_name, parameters):
        """
        Регистрирует вычислитель метрики, по имени и параметрам. 
        """
        module=metric_module_import(metric_name)
        self.metric_counter=module.Metric_counter(self.tasks_list,parameters)

    def count_metric(self):
        """
        Вычисляет значение метрики
        """
        self.counted_values=self.metric_counter.count_values(self.time_compression)

    def print_values(self,file_name):
        """
        Печатает вычисленные значения в файл
        """
        f=open(file_name,"w")
        s=self.metric_counter.get_header_string()
        f.write(s+'\n')
        for key in self.counted_values.keys():
            s=self.metric_counter.format_row(key,self.counted_values[key])
            f.write(s+'\n')
        f.close()

    def plot_values(self,metric_name,file_name):
        """
        Отрисовывает график, если это необходимо. 
        """

        plotter=statistics_plotter.Plotter(_("Print values by metric '%s'") % metric_name,
                self.counted_values,
                self.metric_counter.get_header_string().split('\t'),
                self.metric_counter.get_draw_type()
                )

        plotter.draw(file_name)

