import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tabulate import tabulate

def load_audio(file_path):
    """Load an audio file and return its waveform and sample rate"""
    try:
        y, sr = librosa.load(file_path, sr=None)  # Preserve original sample rate
        return y, sr
    except Exception as e:
        raise Exception(f"Error loading audio file {file_path}: {str(e)}")

def align_audio(y1, y2):
    """Align two audio tracks by truncating the longer one"""
    min_len = min(len(y1), len(y2))
    return y1[:min_len], y2[:min_len]

def compute_waveform_differences(y1, y2, sr, interval=0.2):
    """Compute waveform differences over time using specified interval in seconds"""
    if len(y1) != len(y2):
        raise ValueError("Audio lengths must match")
    
    # Calculate samples per interval
    samples_per_interval = int(sr * interval)
    
    # Calculate number of intervals
    num_intervals = len(y1) // samples_per_interval
    
    timestamps = []
    diff_values = []
    rms_values1 = []
    rms_values2 = []
    peak_diffs = []  # Add peak difference tracking
    
    for i in range(num_intervals):
        start_sample = i * samples_per_interval
        end_sample = start_sample + samples_per_interval
        
        # Get audio segments for this interval
        segment1 = y1[start_sample:end_sample]
        segment2 = y2[start_sample:end_sample]
        
        # Calculate RMS for each segment
        rms1 = float(np.sqrt(np.mean(segment1**2)))
        rms2 = float(np.sqrt(np.mean(segment2**2)))
        
        # Calculate difference metrics
        avg_diff = float(np.mean(np.abs(segment1 - segment2)))
        peak_diff = float(np.max(np.abs(segment1 - segment2)))  # Add peak difference
        
        timestamps.append(i * interval)
        diff_values.append(avg_diff)
        rms_values1.append(rms1)
        rms_values2.append(rms2)
        peak_diffs.append(peak_diff)
    
    return timestamps, diff_values, rms_values1, rms_values2, peak_diffs

def detect_change_timestamps(diff_values, timestamps, rms1, rms2, peak_diffs, threshold=0.01):
    """Detect major change points in the track"""
    change_points = []
    for i in range(len(diff_values)):
        if diff_values[i] > threshold or peak_diffs[i] > threshold * 2:  # Consider both average and peak differences
            change_points.append({
                'timestamp': float(timestamps[i]),
                'difference_magnitude': float(diff_values[i]),
                'peak_difference': float(peak_diffs[i]),
                'rms_file1': float(rms1[i]),
                'rms_file2': float(rms2[i])
            })
    return change_points

def compute_audio_features(y, sr):
    """Compute various audio features"""
    features = {}
    
    # Basic statistics
    features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
    features['peak_amplitude'] = float(np.max(np.abs(y)))
    features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # Spectral features
    spec = np.abs(librosa.stft(y))
    features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(S=spec, sr=sr)))
    features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(S=spec, sr=sr)))
    features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(S=spec, sr=sr)))
    
    # Mel-frequency features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_energy'] = float(np.mean(mel_spec))
    
    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)
    
    # Loudness features
    features['perceived_loudness'] = float(librosa.amplitude_to_db(features['rms_energy'], ref=1.0))
    
    return features

def compare_wav_files(file1_path, file2_path, threshold=0.01):
    """Compare two WAV files and display analysis"""
    # Load audio files
    print(f"Loading audio files...")
    y1, sr1 = load_audio(file1_path)
    y2, sr2 = load_audio(file2_path)
    
    if sr1 != sr2:
        print(f"Warning: Sample rates differ ({sr1} vs {sr2}). Resampling second file...")
        y2, sr2 = librosa.load(file2_path, sr=sr1)
    
    # Align audio
    print("Aligning audio files...")
    y1_aligned, y2_aligned = align_audio(y1, y2)
    
    # Compute differences
    print("Computing differences...")
    timestamps, diff_values, rms1, rms2, peak_diffs = compute_waveform_differences(y1_aligned, y2_aligned, sr1, interval=0.2)
    
    # Detect changes
    changes = detect_change_timestamps(diff_values, timestamps, rms1, rms2, peak_diffs, threshold)
    
    # Compute audio features for both files
    print("Analyzing audio features...")
    features1 = compute_audio_features(y1_aligned, sr1)
    features2 = compute_audio_features(y2_aligned, sr2)
    
    # Print file information
    print("\nAudio File Comparison Report")
    print("=========================")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}\n")
    
    # Print file information table
    file_info = [
        ['Property', 'File 1', 'File 2'],
        ['Sample Rate', int(sr1), int(sr2)],
        ['Duration (s)', f"{len(y1)/float(sr1):.2f}", f"{len(y2)/float(sr2):.2f}"],
        ['Number of Samples', len(y1), len(y2)]
    ]
    print("File Information:")
    print(tabulate(file_info, headers='firstrow', tablefmt='grid'))
    print()
    
    # Print timestamp differences
    print("Timestamp Analysis (0.2-second intervals):")
    if changes:
        changes_table = [['Time (s)', 'RMS File 1', 'RMS File 2', 'Avg Diff', 'Peak Diff', 'Change Type']]
        for change in changes:
            # Determine the type of change based on RMS comparison
            rms_diff = change['rms_file1'] - change['rms_file2']
            if abs(rms_diff) < 0.001:
                change_type = "Content Different"
            elif rms_diff > 0:
                change_type = "Louder in File 1"
            else:
                change_type = "Louder in File 2"
                
            # Add row to table
            changes_table.append([
                f"{change['timestamp']:.2f}-{change['timestamp']+0.2:.2f}",
                f"{change['rms_file1']:.4f}",
                f"{change['rms_file2']:.4f}",
                f"{change['difference_magnitude']:.4f}",
                f"{change['peak_difference']:.4f}",
                change_type
            ])
        print(tabulate(changes_table, headers='firstrow', tablefmt='grid'))
    else:
        print("No significant differences detected above the threshold.")
    print()
    
    # Print audio features comparison
    print("Overall Audio Features Comparison:")
    features_table = [['Feature', 'File 1', 'File 2', 'Absolute Diff', 'Diff (%)']]
    for feature in features1.keys():
        val1 = features1[feature]
        val2 = features2[feature]
        diff = abs(val1 - val2)
        diff_percent = (diff / abs(val1)) * 100 if val1 != 0 else float('inf')
        features_table.append([
            feature,
            f"{val1:.4f}",
            f"{val2:.4f}",
            f"{diff:.4f}",
            f"{diff_percent:.2f}"
        ])
    print(tabulate(features_table, headers='firstrow', tablefmt='grid'))
    
    # Create visualization plots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Waveform Comparison
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(len(y1_aligned)) / sr1, y1_aligned, label='File 1', alpha=0.7)
    plt.plot(np.arange(len(y2_aligned)) / sr1, y2_aligned, label='File 2', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Waveform Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Waveform Difference
    plt.subplot(4, 1, 2)
    diff_signal = y1_aligned - y2_aligned
    plt.plot(np.arange(len(diff_signal)) / sr1, diff_signal, label='Difference', color='red', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude Difference')
    plt.title('Direct Waveform Difference')
    plt.grid(True)
    
    # Plot 3: Average and Peak Differences over time
    plt.subplot(4, 1, 3)
    plt.plot(timestamps, diff_values, label="Average Difference", color="blue", alpha=0.7)
    plt.plot(timestamps, peak_diffs, label="Peak Difference", color="red", alpha=0.7)
    plt.axhline(y=threshold, color="black", linestyle="--", label="Threshold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Difference Magnitude")
    plt.title("Difference Analysis Over Time")
    plt.legend()
    plt.grid(True)
    
    # Plot 4: RMS Energy Comparison
    plt.subplot(4, 1, 4)
    plt.plot(timestamps, rms1, label="File 1 RMS", color="blue", alpha=0.7)
    plt.plot(timestamps, rms2, label="File 2 RMS", color="green", alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS Energy")
    plt.title("RMS Energy Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return changes

if __name__ == "__main__":
    # Example usage
    file1_path = "Sample_1EP.wav"
    file2_path = "Sample_2EP.wav"
    try:
        differences = compare_wav_files(file1_path, file2_path, threshold=0.01)
        print("\nAnalysis completed successfully.")
    except Exception as e:
        print(f"Error comparing WAV files: {str(e)}")
