import numpy as np
from collections import defaultdict
import pickle


class Series(object):
    """
    Simple class to store single time series.
    """  
    def __init__(self, name):
        self.name = name
        self.steps = []
        self.series = []
        self.sdict = {}

    def add(self, i, value):
        self.steps.append(i)
        self.series.append(value)
        self.sdict[i] = value

    def get(self, i=None):
        if i is None:
            return self.steps, self.series
        else:
            return self.sdict[i]



class Trial(object):
    """
    Class to store a single run's results, including training and test performance,
    as well as image reconstructions and translations. 
    """
    def __init__(self, experiment_name):
        """
        experiment_name: name of experiment
        """
        self.name = experiment_name
        self.series = {}
        

    def add_to_series(self, name, i, value):
        """
        Add to series (which may not yet exist). 

        name: series name
        i: time step
        value: value to add
        """    
        if name in self.series:
            self.series[name].add(i, value)
        else:
            s = Series(name)
            s.add(i, value)
            self.series[name] = s
    

    def get_series(self, name, i=None):
        """
        Get series by name. Returns entire series or return single point.
        """
        if i is None:
            return self.series[name].get()
        else:
            return self.series[name].get(i)



class Results(object):
    """
    Class to store experimental results across multiple runs.
    """
    def __init__(self, name):
        """
        name: experiment name
        """
        self.name = name
        self.runs = {}
        self.last = None  # tracks last added Trial


    def create_run(self, name):
        """
        name: name of the new run
        """
        t = Trial(name)
        self.runs[name] = t
        self.last = t


    def get(self, run_name):
        """
        Get experimental run by name.
        """
        return self.runs[run_name]
    
    
    def add(self, i, value, series_name, run_name=None):
        """
        Adds timestep and value to a given series in a given trial. If trial name is None, then 
        add to last added trial.

        i: time step
        value: value to add
        series_name: name of time series 
        run_name: name of experimental run (i.e. trial)
        """
        if run_name is None:
            if self.last is None:
                raise Exception("Result object is empty. No runs to add to. Create a run first.")
            else:
                self.last.add_to_series(series_name, i, value)
        else:
            self.runs[run_name].add_to_series(series_name, i, value)


    @staticmethod
    def save(result, file_path=None):
        """
        Static method to save Results.
        """
        if file_path is None:
            file_path = '../results/'+result.name+'.pickle'
        with open(file_path, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(result_name, file_path=None):
        """
        Static method to load Results.
        """
        if file_path is None:
            file_path = '../results/'
        with open(file_path+result_name, 'rb') as f:
            return pickle.load(f)
