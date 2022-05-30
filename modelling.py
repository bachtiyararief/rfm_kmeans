import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

"""
Created on Thu May 26 09:41:45 2022

@author: Bachtiyar M. Arief
"""

class Modelling():

    def __init__(self, **parameter):
        self.data = parameter.get('data')
            
    def standarization(self, scalertype : str) -> np.ndarray:
        scalertype = re.sub('[^a-z]', '', scalertype.lower())
         
        if('minmax' in scalertype):
            scaler = preprocessing.MinMaxScaler()
        elif('maximumabsolute' in scalertype):
            scaler = preprocessing.MaxAbsScaler()
        elif('robust' in scalertype):
            scaler = preprocessing.RobustScaler()
        else:
            scaler = preprocessing.StandardScaler()
            
        datascaling = scaler.fit_transform(self.data)
        return datascaling
    
    def clustering(self, clustertype : str, **parameter) -> pd.DataFrame:
        
        clustertype = re.sub('[^a-z]', '', clustertype.lower())
        set_params  = parameter.get('set_params')
        isstandartization = set_params.get('standarization', True)
        
        if(isstandartization):
            scalertype = set_params.get('scalertype')
            datamodel  = self.standarization(scalertype = scalertype)
        else:
            datamodel = self.data 
            
        if('kmeans' in clustertype):
            #Define important parameter
            n_clusters = set_params.get('n_clusters', 6)
            iterations = set_params.get('iterations', 300)
            
            #Train model
            modelselected = KMeans(n_clusters = n_clusters, 
                                   init = 'k-means++', 
                                   max_iter = iterations,
                                   random_state = 42)
        
        result = modelselected.fit(datamodel)
        
        return result