"""Fold-local HSP-S preprocessing; no legacy chorus feature contract is reused."""
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from chorus_hit.hit_nonhit.modeling import estimator as family_estimator
from chorus_hit.hit_nonhit.schema import require


def estimator(candidate, columns, seed):
    view = candidate['representation']
    require(view in {'all_scalar', 'summary_scalar', 'pca95'}, 'Unknown HSP-S representation')
    require(all(n.startswith(('lowlevel.', 'rhythm.', 'tonal.')) for n in columns), 'Non-audio predictors forbidden')
    chosen = [n for n in columns if n.endswith(('.mean', '.var')) or n.count('.') == 1] if view == 'summary_scalar' else columns
    require(bool(chosen), 'Empty audio feature view')
    model = family_estimator({**candidate, 'representation': 'all518'}, seed).named_steps['model']
    steps = [('columns', ColumnTransformer([('audio', 'passthrough', chosen)], remainder='drop')),
             ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
             ('variance', VarianceThreshold()), ('scale', StandardScaler())]
    if view == 'pca95': steps.append(('pca', PCA(n_components=.95, svd_solver='full')))
    return Pipeline(steps + [('model', model)])


def metrics(y, predicted, scores=None):
    from chorus_hit.hit_nonhit.modeling import metrics as shared_metrics
    return {**shared_metrics(y, predicted, scores), 'f1': f1_score(y, predicted, zero_division=0),
            'per_class': classification_report(y, predicted, labels=[0, 1], output_dict=True, zero_division=0)}
