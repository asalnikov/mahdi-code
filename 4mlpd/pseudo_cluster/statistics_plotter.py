# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

class Plotter(object):
    """
    Класс для отрисовки графиков
    """

    def __init__(self,title,data,labels,draw_type):
        self.title=title
        self.data=data
        self.labels=labels
        self.draw_type=draw_type

    def draw(self,file_name):
        """
        Отрисовывает в файл, или 
        в окно, если имя файла не задано.
        """
        plt.title(self.title)
        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])


        if self.draw_type == 'chart':
            fig,ax = plt.subplots()
            ax.set_xticklabels(self.data.keys())
            ax.bar(range(0,len(self.data)),self.data.values())

        if self.draw_type == 'plot':
            data_array=list()
            for k,v  in self.data.items():
                array_row=list()
                array_row.append(float(k))
                #XXX may be for future
                #for i in v:
                #    arrary_row.append(i)
                array_row.append(float(v))
                data_array.append(array_row)

            plt.plot(data_array)

        plt.show()

