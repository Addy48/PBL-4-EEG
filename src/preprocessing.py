"""
EEG Signal Preprocessing Utilities
-----------------------------------
Preprocessing pipeline for raw EEG signals from SAM-40 dataset.
Includes baseline drift removal and wavelet-based denoising.

Authors: Prakriti Sharma, Aaditya Upadhyay
Manipal University Jaipur, 2026
"""

import numpy as np
from scipy.signal import savgol_filter
import pywt


def remove_baseline_drift(signal, window_length=51, polyorder=3):
    """
    Remove baseline drift using Savitzky-Golay filter.
    
    Parameters
    ----------
    signal : np.ndarray
        Raw EEG signal (1D array, single channel).
    window_length : int
        Length of the filter window (must be odd).
    polyorder : int
        Order of the polynomial used to fit samples.
    
    Returns
    -------
    np.ndarray
        Signal with baseline drift removed.
    """
    baseline = savgol_filter(signal, window_length, polyorder)
    return signal - baseline


def wavelet_denoise(signal, wavelet='db2', level=4, threshold_mode='soft'):
    """
    Denoise EEG signal using Daubechies-2 wavelet thresholding.
    
    Applies universal threshold (VisuShrink) to detail coefficients
    while preserving the approximation coefficients.
    
    Parameters
    ----------
    signal : np.ndarray
        Input EEG signal (1D).
    wavelet : str
        Wavelet family to use. Default 'db2' (Daubechies-2).
    level : int
        Decomposition level.
    threshold_mode : str
        'soft' or 'hard' thresholding.
    
    Returns
    -------
    np.ndarray
        Denoised signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Universal threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Threshold detail coefficients only (keep approximation intact)
    denoised_coeffs = [coeffs[0]]  # approximation coefficients
    for detail in coeffs[1:]:
        denoised_coeffs.append(
            pywt.threshold(detail, threshold, mode=threshold_mode)
        )
    
    return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]


def preprocess_channel(raw_signal, fs=128):
    """
    Full preprocessing pipeline for a single EEG channel.
    
    Steps:
        1. Remove baseline drift (Savitzky-Golay)
        2. Wavelet denoising (Daubechies-2)
    
    Parameters
    ----------
    raw_signal : np.ndarray
        Raw EEG signal from one channel.
    fs : int
        Sampling frequency in Hz (default: 128 for Emotiv EPOC+).
    
    Returns
    -------
    np.ndarray
        Preprocessed signal.
    """
    # Step 1: Baseline drift removal
    signal = remove_baseline_drift(raw_signal)
    
    # Step 2: Wavelet denoising
    signal = wavelet_denoise(signal)
    
    return signal


def preprocess_trial(trial_data, fs=128):
    """
    Preprocess all channels in an EEG trial.
    
    Parameters
    ----------
    trial_data : np.ndarray
        Shape (n_channels, n_samples). Raw EEG data for one trial.
    fs : int
        Sampling frequency.
    
    Returns
    -------
    np.ndarray
        Preprocessed trial data, same shape as input.
    """
    n_channels = trial_data.shape[0]
    preprocessed = np.zeros_like(trial_data)
    
    for ch in range(n_channels):
        preprocessed[ch] = preprocess_channel(trial_data[ch], fs=fs)
    
    return preprocessed
