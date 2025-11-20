""" Celal ALAGOZ - 6/17/2025
CFIRE: Cross-Domain Feature Integration for Robust Time Series Classification

Required packages:
    - aeon
    - PyWavelets
    - dtaidistance
    - tsfresh
    
"""
import numpy as np
from scipy.fft import rfft, dct
from scipy.signal import hilbert
from pywt import wavedec
from scipy.stats import kurtosis, skew, differential_entropy
from aeon.transformations.collection.feature_based import Catch22 as c22
from tsfresh.feature_extraction.feature_calculators import (c3,
                                                            cid_ce,
                                                            number_crossing_m,
                                                            mean_second_derivative_central,
                                                            time_reversal_asymmetry_statistic,
                                                            percentage_of_reoccurring_datapoints_to_all_datapoints)
from joblib import Parallel, delayed
import os

os.environ['PYTHONWARNINGS']='ignore'

def replace_nan(x,val_=-1):
    if np.any(np.isnan(x)):
        x[np.isnan(x)]=val_
    if np.any(np.isinf(x)):
        x[np.isinf(x)]=val_
    return x

def stack_features(features,features_new):
    if len(features)==0:
        features=features_new
    else:
        features=np.c_[features,features_new]
    return features

def safe_hist4(x_row):
    x_row = np.array(x_row, dtype=float)
    x_row = x_row[np.isfinite(x_row)]

    # Handle empty or all-NaN rows
    if len(x_row) == 0:
        return np.zeros(4)

    min_x, max_x = np.min(x_row), np.max(x_row)

    # Handle constant data — directly return one full bin
    if np.isclose(min_x, max_x):
        hist = np.zeros(4)
        hist[0] = 1.0  # put all mass in the first bin
        return hist

    # Add small margin to range to avoid precision collapse
    eps = (max_x - min_x) * 1e-9
    hist, _ = np.histogram(x_row, bins=4, range=(min_x - eps, max_x + eps))
    return hist / len(x_row)

class CFIRE():

    def __init__(self,
                 norms = True,
                 stats = True,
                 series= True, 
                 temp  = True,
                 replace_nans=True,
                 multiprocessing=True,
                 n_jobs=-1,
                 n_jobs_c22=1,
                 ):
        
        self.norms = norms
        self.stats = stats
        self.series= series
        self.temp  = temp
        self.replace_nans=replace_nans
        self.n_jobs=n_jobs
        self.n_jobs_c22=n_jobs_c22
        self.multiprocessing=multiprocessing
        
        self.representation_functions = \
        {
            'TIME': lambda x : x,
            'DT1' : lambda x : np.diff(x, n=1, axis=-1),
            'DT2' : lambda x : np.diff(x, n=2, axis=-1),
            'HLB' : lambda x : np.abs(hilbert(x, axis=-1)),
            'DWT' : lambda x : wavedec(x,'dmey',level=2)[0],
            'FFT' : lambda x : np.abs(rfft(x, axis=-1)),
            'DCT' : lambda x : dct(x,1), 
            'ACF' : lambda x : np.array([np.correlate(xi,xi,mode='same') for xi in x]),
        }
        
        if self.norms:
            self.norms_functions = \
            {
                'L1' : lambda x : np.linalg.norm(x, 1, axis=-1), # 1-norm
                # 'L2' : lambda x : np.linalg.norm(x, axis=-1), # 2-norm
                'Max': lambda x : np.max(x, axis=-1), # inf-norm
                'Min': lambda x : np.min(x, axis=-1), # -inf-norm
            }
        
        if self.stats:
            self.stats_functions = \
            {            
                'Med' : lambda x : np.median(x, axis=-1),
                'Std' : lambda x : np.std(x, axis=-1),
                'Kurt': lambda x : kurtosis(x, axis=-1),
                'Skew': lambda x : skew(x, axis=-1), 
                'DfEn': lambda x : differential_entropy(x, axis=-1), # DiffEntropy
                # 'HMod': lambda x : c22(features=[1],replace_nans=False).fit_transform(x), # HistogramMode10
                # 'P4B' : lambda x : [np.histogram(x_row, bins=4)[0]/len(x_row) for x_row in x], # Pdf4bin
                'P4B': lambda x: [safe_hist4(x_row) for x_row in x]
            }
        
        if self.series:
            self.series_functions = \
            {
                'MLD': lambda x : np.mean(np.ma.getdata(np.ma.masked_invalid(np.log(np.abs(np.diff(x,axis=-1))))), axis=-1), # MeanLogDiff                                
                'HCP' : lambda x : c22(features=[13],replace_nans=False).fit_transform(x), # MDhrvClassicPnn40
                'LMT' : lambda x : c22(features=[16],replace_nans=False).fit_transform(x), # FClocalSimpleMean1Tauresrat
                'FMA' : lambda x : c22(features=[6],replace_nans=False).fit_transform(x), # COfirstMinAc
                'MTQ' : lambda x : c22(features=[15],replace_nans=False).fit_transform(x), # SBmotifThreeQuantileHh
                'BML1': lambda x : c22(features=[14],replace_nans=False).fit_transform(x), # SBbinaryStatsMeanLongstretch1
                'BML0': lambda x : c22(features=[2],replace_nans=False).fit_transform(x), # SBbinaryStatsDiffLongstretch0
                'WRC' : lambda x : c22(features=[8],replace_nans=False).fit_transform(x), # SPsummariesWelchRectCentroid
                'MIS' : lambda x : c22(features=[12],replace_nans=False).fit_transform(x), # INautoMutualInfoStats40GaussianFmmi
                'PWT' : lambda x : c22(features=[21],replace_nans=False).fit_transform(x), # PDperiodicityWangTh0
                'DTD' : lambda x : c22(features=[17],replace_nans=False).fit_transform(x), # COembed2DistTauDExpfitMeandiff
                # 'feats_c22': lambda x : c22(features=[2,6,8,12,13,14,16,17,21],replace_nans=False,n_jobs=self.n_jobs_c22).fit_transform(x),
                
                'FLMx': lambda x : np.argmax(x, axis=-1) / x.shape[-1], # FirstLocMax
                # 'FLMn': lambda x : np.argmin(x, axis=-1) / x.shape[-1], # FirstlocMin
                'LLMx': lambda x : 1.0 - np.argmax(x[:,::-1], axis=-1) / x.shape[-1], # LastLocMax
                'LLMn': lambda x : 1.0 - np.argmin(x[:,::-1], axis=-1) / x.shape[-1], # LastLocMin
                'CAM' : lambda x: [len(x_row[x_row>x_row.mean()]/len(x_row)) for x_row in x], # CountAboveMean
                'MSC' : lambda x: [mean_second_derivative_central(x_row) for x_row in x], # MeanSecondDerivCentral
            }
        
        if self.temp:
            self.temp_functions = \
            {               
                'CID' : lambda x: [cid_ce(x_row,3) for x_row in x], # CidCe
                'C3'  : lambda x: [c3(x_row,1) for x_row in x], # C3
                'TRAS': lambda x: [time_reversal_asymmetry_statistic(x_row,3) for x_row in x], # TimeReversalAsymmetryStatistic
                'NCM' : lambda x: [number_crossing_m(x_row,np.mean(x_row)) for x_row in x], # NumCrossingMean
                'PRD' : lambda x: [percentage_of_reoccurring_datapoints_to_all_datapoints(x_row)  for x_row in x], # PercentReoccurrDpoints           
            }
        
        self.fitted = False

    def transform(self, X, rep_, fun_rep):
               
        features = []

        Z = fun_rep(X)
                
        if self.replace_nans:
            Z = replace_nan(Z)
        
        if self.norms:
            if rep_ not in ['FFT']:
                for norm_, fun_norm in self.norms_functions.items():
                    feats_norm = fun_norm(Z)
                    
                    features=stack_features(features,feats_norm)
                
        if self.stats:
            if rep_ not in ['FFT']:
                for stat_, fun_stat in self.stats_functions.items():
                    
                    feats_stat = fun_stat(Z)
                    
                    if self.replace_nans:
                        feats_stat = replace_nan(feats_stat)
                    
                    features=stack_features(features,feats_stat)
                
        if self.series:
            if rep_ not in ['DT2']:
                for extra_, fun_series in self.series_functions.items():
                    
                    feats_series = fun_series(Z)
                    
                    if self.replace_nans:
                        feats_series = replace_nan(feats_series)
                    
                    features=stack_features(features,feats_series) 
                    
        if self.temp:
            if rep_ in ['TIME','DT1','DT2']:
                for extra_, fun_temp in self.temp_functions.items():
                    
                    feats_temp = fun_temp(Z)
                    
                    if self.replace_nans:
                        feats_temp = replace_nan(feats_temp)
                    
                    features=stack_features(features,feats_temp)
        
        return features
    
    def fit_transform(self, X):
        if self.multiprocessing and len(self.representation_functions)>1:
            feats_list_ = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed(self.transform)(X,rep_,fun_rep) for rep_, fun_rep in self.representation_functions.items())
            
            feats_list1 = [np.array(x) for x in feats_list_ if len(x)>0]
            feats_list = [x.reshape(-1,1) if x.ndim==1 else x for x in feats_list1]
            
            features=np.hstack(feats_list)
        else:
            features = []
            for rep_, fun_rep in self.representation_functions.items():
                feats_=np.array(self.transform(X,rep_,fun_rep))
                features=stack_features(features,feats_)
                
        if features.ndim==1: features = np.expand_dims(features, axis=-1)
        self.fitted = True
        
        return features