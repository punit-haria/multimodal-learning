import numpy as np
from collections import defaultdict


class Series(Object):
    """
    Simple class to store single time series.
    """  
    def __init__(self, name):
        self.name = name
        self._steps = []
        self.series = {}

    def add(self, i, value):
        self._steps.append(i)
        self.series[i] = value        

    def get(self, i):
        return self.series[i]
    


class Mappings(Object):
    """
    Simple class to store sequence of structured mappings.
    """  
    def __init__(self, name):
        self.name = name
        self._steps = []
        self.initial = {}
        self.mapping = {}

    def add(self, i, initial, mapping):
        self._steps.append(i)
        self.initial[i] = value   
        self.mapping[i] = mapping     

    def get(self, i):
        return self.initial[i], self.mapping[i]



class Trial(object):
    """
    Class to store experimental results, including training and test performance,
    as well as image reconstructions and translations. 
    """
    def __init__(self, experiment_name, series_names, mapping_names):
        """
        experiment_name: name of experiment
        series_names: list of names of time series to track
        mapping_names: list of names of mapping sequences to track
        """
        self.name = experiment_name
        self.series_names = series_names
        self.mapping_names = mapping_names
        
        # manage time series results
        self.series = {}
        for s in self.series_names:
            self.series[s] = Series(s)

        # manage mapping sequences
        self.mappings = {}
        for m in self.mapping_names:
            self.mappings[m] = Mappings(m)
        
        
    def get_series(self, )


class Results(object):
    def __init__(self,):
        pass