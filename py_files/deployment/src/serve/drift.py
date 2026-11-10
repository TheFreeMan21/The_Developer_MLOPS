import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from scipy.stats import ks_2samp
from src.config_deploy import drift_ks_pvalue_threshold

def load_inference_logs(log_path):
    if not Path(log_path).exists():
        return []
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
            return logs if isinstance(logs, list) else []
    except Exception:
        return []

def filter_logs_by_time(logs, days):
    if not logs:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for log in logs:
        try:
            timestamp_str = log.get('timestamp', '')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff:
                    filtered.append(log)
        except (ValueError, TypeError):
            filtered.append(log)
    return filtered

def load_baseline_stats(drift_baseline_stats_path):
    if not Path(drift_baseline_stats_path).exists():
        return None
    try:
        with open(drift_baseline_stats_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def check_confidence_drift(current_confidences, baseline_stats):
    if not baseline_stats or 'confidence_statistics' not in baseline_stats:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline = baseline_stats['confidence_statistics']
    current_array = np.array(current_confidences)

    if len(current_array) < 2:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline_samples = np.random.normal(baseline['mean'], baseline['std'], size=len(current_array))
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current_array)

    mean_diff = abs(float(np.mean(current_array)) - baseline['mean'])
    drift_detected = bool((ks_pvalue < drift_ks_pvalue_threshold) or (mean_diff > 0.1))

    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'mean_diff': mean_diff,
        'drift_detected': drift_detected,
        'threshold_ks': drift_ks_pvalue_threshold
    }

def check_class_distribution_drift(predictions, baseline_stats):
    if not baseline_stats or 'class_distribution' not in baseline_stats:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline = baseline_stats['class_distribution']
    current_total = len(predictions)
    
    if current_total == 0:
        return {'ks_statistic': 0.0, 'ks_pvalue': 1.0, 'drift_detected': False, 'threshold_ks': drift_ks_pvalue_threshold}

    baseline_dist = np.array([baseline['class_0_count'], baseline['class_1_count']]) / baseline['total']
    baseline_samples = np.random.choice([0, 1], size=baseline['total'], p=baseline_dist)
    
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)
    return {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'drift_detected': bool(ks_pvalue < drift_ks_pvalue_threshold),
        'threshold_ks': drift_ks_pvalue_threshold,
    }

def detect_drift(logs, days, baseline_stats):
    filtered_logs = filter_logs_by_time(logs, days=days)
    if not filtered_logs:
        return {'overall_drift_detected': False, 'num_samples': 0, 'error': 'No logs found'}

    confidences = [float(l['confidence']) for l in filtered_logs if l.get('confidence') is not None]
    predictions = [int(l['prediction']) for l in filtered_logs if l.get('prediction') is not None]

    confidence_results = check_confidence_drift(confidences, baseline_stats or {})
    class_dist_results = check_class_distribution_drift(predictions, baseline_stats or {})

    return {
        'overall_drift_detected': confidence_results.get('drift_detected', False) or class_dist_results.get('drift_detected', False),
        'num_samples': len(filtered_logs),
        'confidence_drift': confidence_results,
        'class_distribution_drift': class_dist_results
    }