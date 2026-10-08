"""Bounded estimators with all learned transforms confined to fit calls."""
from contextlib import contextmanager
import signal
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, average_precision_score)
from chorus_hit.config import FEATURE_COLUMNS
from .schema import require


def estimator(c, seed):
    representation = c['representation']
    cols = [x for x in FEATURE_COLUMNS if '_std_' in x] if representation == 'std74' else FEATURE_COLUMNS
    require(representation in {'all518', 'std74', 'pca95'}, 'Unknown representation')
    family = c['family']; weight = c.get('class_weight')
    if family == 'dummy':
        model = DummyClassifier(strategy='prior')
    elif family == 'logistic':
        model = LogisticRegression(C=c['C'], class_weight=weight, max_iter=2000, solver='lbfgs', random_state=seed)
    elif family == 'linear_svm':
        model = LinearSVC(C=c['C'], class_weight=weight, dual='auto', max_iter=5000, random_state=seed)
    elif family == 'rbf_svm':
        model = SVC(C=c['C'], gamma='scale', class_weight=weight, cache_size=256, max_iter=10000)
    elif family in {'random_forest', 'extra_trees'}:
        cls = RandomForestClassifier if family == 'random_forest' else ExtraTreesClassifier
        model = cls(n_estimators=200, min_samples_leaf=c['min_samples_leaf'], max_features='sqrt', class_weight=weight, n_jobs=1, random_state=seed)
    elif family == 'hist_boosting':
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1, early_stopping=False, random_state=seed)
    elif family == 'mlp':
        # Row-random early stopping would break the artist validation boundary.
        model = MLPClassifier(hidden_layer_sizes=(32,), alpha=.1, max_iter=200, early_stopping=False, random_state=seed)
    else:
        raise ValueError('Unknown model family')
    steps = [('columns', ColumnTransformer([('audio', 'passthrough', cols)], remainder='drop')),
             ('variance', VarianceThreshold()), ('scale', StandardScaler())]
    if representation == 'pca95':
        steps.append(('pca', PCA(n_components=.95, svd_solver='full')))
    return Pipeline(steps + [('model', model)])


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError('Predeclared per-fit time limit exceeded')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def score_values(model, X):
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X)), {'kind': 'uncalibrated_decision_margin', 'threshold': 0.0}
    return np.asarray(model.predict_proba(X))[:, 1], {'kind': 'uncalibrated_class_score', 'threshold': .5}


def metrics(y, predicted, scores=None):
    require(set(y) == {0, 1}, 'Metrics require two classes')
    out = {'balanced_accuracy': balanced_accuracy_score(y, predicted), 'accuracy': accuracy_score(y, predicted),
           'macro_f1': f1_score(y, predicted, average='macro', zero_division=0),
           'precision': precision_score(y, predicted, zero_division=0), 'recall': recall_score(y, predicted, zero_division=0),
           'negative_recall': recall_score(y, predicted, pos_label=0, zero_division=0),
           'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
           'class_counts': {str(k): int(sum(np.asarray(y) == k)) for k in [0, 1]},
           'predicted_counts': {str(k): int(sum(np.asarray(predicted) == k)) for k in [0, 1]}}
    if scores is not None:
        out.update(roc_auc=roc_auc_score(y, scores), average_precision=average_precision_score(y, scores))
    return out


def cluster_interval(y, predicted, groups, seed, draws=1000):
    y, predicted, groups = np.asarray(y), np.asarray(predicted), np.asarray(groups)
    units = np.unique(groups)
    if len(units) < 5 or any(len(set(groups[y == k])) < 5 for k in [0, 1]):
        return {'balanced_accuracy': None, 'reason': 'Fewer than five independent groups per class'}
    rng = np.random.default_rng(seed); values = []
    for _ in range(draws):
        idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(units, len(units), replace=True)])
        if set(y[idx]) == {0, 1}:
            values.append(balanced_accuracy_score(y[idx], predicted[idx]))
    return {'balanced_accuracy': np.quantile(values, [.025, .975]).tolist() if values else None,
            'valid_resamples': len(values), 'method': 'connected-group percentile bootstrap', 'groups': len(units)}
