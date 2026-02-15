"""
EEG Statistical Feature Extraction
------------------------------------
Extracts statistical features from preprocessed EEG signals.
These features are used as input to classification models.

Authors: Prakriti Sharma, Aaditya Upadhyay
Manipal University Jaipur, 2026
"""

import numpy as np
from scipy.stats import kurtosis, skew


def extract_channel_features(signal):
    """
    Extract statistical features from a single preprocessed EEG channel.
    
    Parameters
    ----------
    signal : np.ndarray
        Preprocessed EEG signal (1D array).
    
    Returns
    -------
    dict
        Dictionary of feature name -> value.
    """
    features = {}
    
    # Central tendency and dispersion
    features['mean'] = np.mean(signal)
    features['variance'] = np.var(signal)
    features['std'] = np.std(signal)
    
    # Signal energy and amplitude
    features['energy'] = np.sum(signal ** 2)
    features['rms'] = np.sqrt(np.mean(signal ** 2))
    features['peak_to_peak'] = np.max(signal) - np.min(signal)
    features['mad'] = np.mean(np.abs(signal - np.mean(signal)))
    
    # Distribution shape
    features['kurtosis'] = kurtosis(signal, fisher=True)
    features['skewness'] = skew(signal)
    
    # Information-theoretic
    features['entropy'] = _shannon_entropy(signal)
    
    return features


def _shannon_entropy(signal, n_bins=50):
    """
    Compute Shannon entropy of signal amplitude distribution.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    n_bins : int
        Number of histogram bins for probability estimation.
    
    Returns
    -------
    float
        Shannon entropy value.
    """
    hist, _ = np.histogram(signal, bins=n_bins, density=True)
    hist = hist[hist > 0]  # remove zero bins
    bin_width = (signal.max() - signal.min()) / n_bins
    probs = hist * bin_width
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def extract_trial_features(trial_data, channel_names=None):
    """
    Extract features from all channels in a trial.
    
    Parameters
    ----------
    trial_data : np.ndarray
        Shape (n_channels, n_samples).
    channel_names : list of str, optional
        Channel names (e.g., ['AF3', 'F7', ...]). If None, uses
        default Emotiv EPOC+ channel names.
    
    Returns
    -------
    dict
        Flattened feature dictionary with keys like 'AF3_mean', 'AF3_variance', etc.
    """
    if channel_names is None:
        channel_names = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
        ]
    
    n_channels = trial_data.shape[0]
    all_features = {}
    
    for ch_idx in range(n_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'CH{ch_idx}'
        ch_features = extract_channel_features(trial_data[ch_idx])
        
        for feat_name, feat_val in ch_features.items():
            all_features[f'{ch_name}_{feat_name}'] = feat_val
    
    return all_features


def build_feature_matrix(all_trials, labels, channel_names=None):
    """
    Build feature matrix from a list of EEG trials.
    
    Parameters
    ----------
    all_trials : list of np.ndarray
        Each element has shape (n_channels, n_samples).
    labels : list or np.ndarray
        Class labels for each trial.
    channel_names : list of str, optional
        EEG channel names.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_trials, n_features).
    y : np.ndarray
        Label array, shape (n_trials,).
    feature_names : list of str
        Names of each feature column.
    """
    feature_dicts = []
    for trial in all_trials:
        fd = extract_trial_features(trial, channel_names)
        feature_dicts.append(fd)
    
    feature_names = sorted(feature_dicts[0].keys())
    
    X = np.array([
        [fd[fn] for fn in feature_names]
        for fd in feature_dicts
    ])
    y = np.array(labels)
    
    return X, y, feature_names
