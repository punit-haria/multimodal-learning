import tensorflow as tf
import pickle



def train(name, model, parameters, data, tracker):

    # load model
    print("Training Model: ", name, flush=True)
    model = model(arguments=parameters, name=name, tracker=tracker)

    # train model
    for i in range(parameters['train_steps'] + 1):

        x = data.sample(parameters['batch_size'], dtype='train')
        if type(x) in [list, tuple]:
            x = x[0]

        model.train(x)

        if i % parameters['test_steps'] == 0:
            print("At iteration ", i, flush=True)

            x = data.sample(parameters['test_sample_size'], dtype='test')
            if type(x) in [list, tuple]:
                x = x[0]

            model.test(x)

        if i % parameters['save_steps'] == 0:
            if i != 0:
                model.save_state(suffix=str(i))
                Results.save(tracker)

    # reset tensorflow session
    model.close()

    # save model performance results
    Results.save(tracker)



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
    def __init__(self, trial_name, model_name, parameters):
        """
        experiment_name: name of experiment
        """
        self.name = trial_name
        self.model_name = model_name
        self.parameters = parameters
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
        self.i = 0  # track maximum time step across all trials


    def create_run(self, run_name, model_name, parameters):
        """
        name: name of the new run
        """
        if not self.contains(run_name):
            self.runs[run_name] = Trial(run_name, model_name, parameters)


    def contains(self, run_name):
        """
        Check if key exists.
        """
        return run_name in self.runs


    def get(self, run_name):
        """
        Get experimental run by name.
        """
        return self.runs[run_name]


    def get_runs(self, ):
        """
        Return keys of runs dictionary.
        """
        return list(self.runs.keys())
    
    
    def add(self, i, value, series_name, run_name):
        """
        Adds timestep and value to a given series in a given trial. If trial name is None, then 
        add to last added trial.

        i: time step
        value: value to add
        series_name: name of time series 
        run_name: name of experimental run (i.e. trial)
        """
        self.runs[run_name].add_to_series(series_name, i, value)

        self.i = max(i, self.i)


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
        with open(file_path+result_name+'.pickle', 'rb') as f:
            return pickle.load(f)
