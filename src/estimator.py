import dill
import numpy as np
from .feature import composition_to_feature

with open('../data/estimator.pkl','rb') as f:
    target_pca, kde_temperature, kde_time = dill.load(f)

def estimate_temperature(target_composition, temperature_range=[500,2500], n_points=401):
    xs = np.linspace(*temperature_range, n_points)
    feat = np.hstack([
        composition_to_feature(target_composition, feature_type='comp'),
#        composition_to_feature(target_composition, feature_type='elemnet'),
        composition_to_feature(target_composition, feature_type='cgcnn')
    ])
    pca_feat = target_pca.transform(feat).reshape(-1)
    data = np.vstack([np.hstack([[x], pca_feat]) for x in xs]).T
    ys = kde_temperature(data)
    return xs, ys

def estimate_time(target_composition, time_range=[500,2500], n_points=401):
    xs = np.linspace(*time_range, n_points)
    feat = np.hstack([
        composition_to_feature(target_composition, feature_type='comp'),
#        composition_to_feature(target_composition, feature_type='elemnet'),
        composition_to_feature(target_composition, feature_type='cgcnn')
    ])
    pca_feat = target_pca.transform(feat).reshape(-1)
    data = np.vstack([np.hstack([[x], pca_feat]) for x in xs]).T
    ys = kde_time(data)
    return xs, ys
