import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn import linear_model

from scipy.io import loadmat, matlab

def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)

class ApAvDataset:

    def __init__(self, path_to_data='/Users/julianstastny/Code/masters-thesis/data/AMED-DFG/behavior'):
        data = []
        for i, filename in enumerate(os.listdir(path_to_data)):
            if filename.startswith('behavior') and filename.endswith('.mat'):
                datapoint = load_mat(os.path.join(path_to_data, filename))
                important_info = datapoint['bsum']['raw']['apav']
                important_info.pop('basal', None)
                # if any([important_info[key]['datanum'] <= 1 for key in important_info.keys()]):
                #     continue
                contained_sessions = list(important_info.keys())
                datanums = {key: important_info[key]['datanum'] for key in important_info.keys()}
                for key in important_info.keys():
                    if important_info[key]['datanum'] <= 1:
                        contained_sessions.remove(key)
                        continue
                    for key_2 in list(important_info[key].keys()):
                        if not (key_2 in ['reward_amount', 'aversi_amount', 'appro1_avoid0', 'reaction_time', 'push1_pull0']):
                            important_info[key].pop(key_2, None)
                    as_pandas_df = pd.DataFrame.from_dict(important_info[key])
                    important_info[key] = as_pandas_df
                important_info['metadata'] = {
                    'monkey_id': datapoint['fsum']['monkey'], 
                    'datetime': datetime.strptime(datapoint['fsum']['session'][2:], '%y-%m-%d_%H-%M-%S'),
                    'contained_sessions': contained_sessions,
                    'datanums': datanums,
                    'matlab_filename': filename
                    }
                data += [important_info]
        
        self.raw_data = [datum for datum in sorted(data, key=lambda d: d['metadata']['datetime'])]
        self.full_sessions_data = [datum for datum in self.raw_data if len(datum['metadata']['contained_sessions'])==3]
    
    def get_data(self, monkey_id=None, full_sessions_only=False):
        data = self.full_sessions_data if full_sessions_only else self.raw_data
        if monkey_id is None:
            return data
        return [datum for datum in data if (datum['metadata']['monkey_id'] == monkey_id)]

    def get_psytrack_data(self, monkey_id, full_session_only=True, stimulus_multiplier=0.1):
        data = self.get_data(monkey_id=monkey_id, full_sessions_only=full_session_only)
        appro1_avoid0 = []
        daylengths = []
        true_daylengths = []
        stimulation = []
        reward_amount, aversi_amount = [], []
        for i, datum in enumerate(data):
            for key in ['stim0', 'stim1', 'resid']:
                if key in datum['metadata']['contained_sessions']:
                    appro1_avoid0.extend(datum[key]['appro1_avoid0'])

                    reward_amount.extend(datum[key]['reward_amount'])
                    aversi_amount.extend(datum[key]['aversi_amount'])

                    stim = [1.0] * datum['metadata']['datanums'][key] if key == 'stim1' else [0.0] * datum['metadata']['datanums'][key] 
                    stimulation.extend(stim)
                    daylengths.append(len(stim))

            true_daylengths.append(sum(datum['metadata']['datanums'].values()))


        psytrack_data = {
            'name': monkey_id,
            'y': np.array(appro1_avoid0),
            'inputs': {
                'reward_amount': np.array(reward_amount)[:, np.newaxis] * stimulus_multiplier,
                'aversi_amount': np.array(aversi_amount)[:, np.newaxis] * stimulus_multiplier,
                'stimulation': np.array(stimulation)[:, np.newaxis]
            },
            'dayLength': np.array(daylengths),
            'true_day_lengths': np.array(true_daylengths)
        }
        return psytrack_data
    
    
def significance_test(data):
    clf0 = linear_model.LogisticRegression()
    clf1 = linear_model.LogisticRegression()

    num_trials = 100

    for i in range(len(data)):
        X_stim0 = data[i]['stim0'][['reward_amount', 'aversi_amount']]
        y_stim0 = data[i]['stim0']['appro1_avoid0']

        X_stim1 = data[i]['stim1'][['reward_amount', 'aversi_amount']]
        y_stim1 = data[i]['stim1']['appro1_avoid0']

        clf0.fit(X_stim0, y_stim0)
        clf1.fit(X_stim1, y_stim1)
        ap_prob_stim0 = clf0.decision_function([[50, 50]])[0]
        ap_prob_stim1 = clf1.decision_function([[50, 50]])[0]
        data[i]['metadata']['stim_increases_avoidance'] = True if (ap_prob_stim0 > ap_prob_stim1) else False



        prob_1 = clf0.predict_proba(X_stim1)[:,1]

        clftest = linear_model.LogisticRegression()

        num_more_extreme_given_null = 0

        for j in range(num_trials):
            sample_y = (prob_1 > np.random.random(len(prob_1))).astype(int)
            clftest.fit(X_stim1, sample_y)

            ap_prob_stim1 = clf1.decision_function([[50, 50]])[0]
            ap_prob_stimtest = clftest.decision_function([[50, 50]])[0]

            if data[i]['metadata']['stim_increases_avoidance']:
                if (ap_prob_stim1 > ap_prob_stimtest):
                    num_more_extreme_given_null += 1
            else:
                if (ap_prob_stim1 < ap_prob_stimtest):
                    num_more_extreme_given_null += 1

        ratio = num_more_extreme_given_null/num_trials

        data[i]['metadata']['significant_diff'] = False if ratio > 0.05 else True
    return data