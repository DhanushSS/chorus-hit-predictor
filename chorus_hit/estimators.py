"""Small, serializable estimator factory; all learned transforms stay in folds."""
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.svm import SVC


def seeded_mutual_information(X,y):
    return mutual_info_classif(X,y,random_state=42,n_jobs=1)


class StrictSelectKBest(SelectKBest):
    def fit(self,X,y=None):
        if self.k!='all' and self.k>X.shape[1]:
            raise ValueError(f'Requested k={self.k} exceeds {X.shape[1]} surviving features')
        return super().fit(X,y)


class FamilyColumns(TransformerMixin,BaseEstimator):
    def __init__(self,columns): self.columns=columns
    def fit(self,X,y=None):
        self.feature_names_in_=np.asarray(X.columns,object); self.n_features_in_=X.shape[1]
        if not set(self.columns).issubset(X.columns): raise ValueError('Missing family features')
        return self
    def transform(self,X): return X.loc[:,self.columns]


def make_estimator(spec,features,seed=42):
    family=spec['family']; params=spec.get('params',{}); representation=spec['representation']
    constructors={
        'dummy':lambda:DummyClassifier(strategy='prior'),
        'logistic':lambda:LogisticRegression(max_iter=3000,random_state=seed),
        'lda':lambda:LinearDiscriminantAnalysis(solver='lsqr'),
        'linear_svm':lambda:SVC(kernel='linear',random_state=seed),
        'rbf_svm':lambda:SVC(kernel='rbf',random_state=seed),
        'poly_svm':lambda:SVC(kernel='poly',degree=2,random_state=seed),
        'random_forest':lambda:RandomForestClassifier(n_estimators=250,random_state=seed,n_jobs=1),
        'extra_trees':lambda:ExtraTreesClassifier(n_estimators=250,random_state=seed,n_jobs=1),
        'gradient_boosting':lambda:GradientBoostingClassifier(n_estimators=100,random_state=seed),
        'hist_gradient_boosting':lambda:HistGradientBoostingClassifier(max_iter=100,early_stopping=False,random_state=seed),
        'knn':lambda:KNeighborsClassifier(n_jobs=1),
        'mlp':lambda:MLPClassifier(hidden_layer_sizes=(64,32),max_iter=800,early_stopping=False,random_state=seed,tol=1e-4)}
    if family not in constructors: raise ValueError(f'Unknown candidate family: {family}')
    model=constructors[family]().set_params(**params)
    if getattr(model,'early_stopping',False): raise ValueError('Internal random early stopping violates grouped protocol')
    steps=[]
    subset=representation.get('columns','all')
    if subset not in {'all','core_statistics','std_only'}: raise ValueError('Unknown feature subset')
    if subset in {'core_statistics','std_only'}:
        statistics={'std'} if subset=='std_only' else {'mean','median','std'}
        columns=[c for c in features if c.rsplit('_',2)[-2] in statistics]
        if not columns: raise ValueError('Statistic subsets require named audio features')
        steps.append(('columns',FamilyColumns(columns)))
    if representation['kind']=='mfcc_spectral':
        prefixes=('mfcc_','rms_','spectral_','zero_crossing_rate_')
        columns=[c for c in features if c.startswith(prefixes)]
        if not columns: raise ValueError('Family ablation requires named audio features')
        steps.append(('family',FamilyColumns(columns)))
    scaling=representation.get('scaling','standard')
    if scaling not in {'standard','quantile'}: raise ValueError('Unknown feature scaling')
    quantiles=representation.get('n_quantiles',50)
    if type(quantiles) is not int or quantiles<2: raise ValueError('n_quantiles must be an integer >=2')
    scaler=StandardScaler() if scaling=='standard' else QuantileTransformer(
        n_quantiles=quantiles,output_distribution='normal',subsample=None,random_state=seed)
    steps.extend([('imputer',SimpleImputer(strategy='median')),('variance',VarianceThreshold()),('scale',scaler)])
    kind=representation['kind']
    if kind=='pca': steps.append(('reduce',PCA(n_components=representation['value'],svd_solver='full')))
    elif kind in {'anova','mi'}:
        steps.append(('reduce',StrictSelectKBest(score_func=f_classif if kind=='anova' else seeded_mutual_information,k=representation['value'])))
    elif kind not in {'none','mfcc_spectral'}: raise ValueError('Unknown representation')
    return Pipeline([*steps,('model',model)])
