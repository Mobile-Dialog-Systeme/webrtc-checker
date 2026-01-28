import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
import pandas as pd


# -----------------------------------------------------
# Folders
# -----------------------------------------------------

input_folder = r"path"
output_folder = r"path"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------------------------------
# Allowed file extensions
# -----------------------------------------------------
valid_ext = (".mp3", ".ogg", ".wav")

# Summary results
summary_results = []

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr


def check_no_signal(y, sr):
    events = []

    max_abs = np.max(np.abs(y))
    mean_amp = np.mean(np.abs(y))

    if max_abs == 0:
        events.append({"name": "No Signal", "times": []})
        return True, events  # allows skipping further analysis but still records in table

    if mean_amp < 0.01:
        events.append({"name": "Very quiet", "times": []})

    return False, events
# Otsu threshold

def otsu_threshold(vols):
    vols = np.clip(vols, 0, 1)
    hist, bins = np.histogram(vols, bins=256, range=(0, 1))
    total = vols.size
    sum_total = np.sum(bins[:-1] * hist)

    sumB = 0
    wB = 0
    max_var = 0
    threshold = 0

    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += bins[i] * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = bins[i]

    return threshold

# -----------------------------------------------------
# Main analyses
# -----------------------------------------------------
def analyse_reverberation(y, sr, audio_file):

    frame_length = int(0.1 * sr)
    hop_length = int(0.05 * sr)

    min_duration_sec = 1.5
    min_frames = int(min_duration_sec * sr / hop_length)

    min_oscillations = 4
    band_drop_ratio = 0.7   # 30% less Engergy in in certain frequency

    # -------------------------------------------------
    # 1. Intensity
    # -------------------------------------------------
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    if np.max(rms) > 0:
        rms = rms / np.max(rms)

    # -------------------------------------------------
    # 2. Intensity-Oszillation
    # -------------------------------------------------
    drms = np.diff(rms)
    sign_changes = np.diff(np.sign(drms)) != 0

    # diration changes
    osc_count = np.zeros(len(rms))
    window = int(0.8 * sr / hop_length)

    for i in range(window, len(rms)):
        osc_count[i] = np.sum(sign_changes[i - window:i])

    # -------------------------------------------------
    # 3. Spectral energie
    # -------------------------------------------------
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    band_low = (freqs >= 800) & (freqs <= 2200)
    band_full = (freqs >= 220) & (freqs <= 5000)

    energy_low = np.sum(S[band_low, :], axis=0)
    energy_full = np.sum(S[band_full, :], axis=0) + 1e-12
    band_ratio = energy_low / energy_full

    # -------------------------------------------------
    # 4. Wave cinditions
    # -------------------------------------------------
    wave_condition = (
        (osc_count >= min_oscillations) |
        (band_ratio < band_drop_ratio)
    )

    # -------------------------------------------------
    # 5. durable waves
    # -------------------------------------------------
    wave_starts = []
    active = False
    start_idx = 0

    for i, val in enumerate(wave_condition):
        if val and not active:
            start_idx = i
            active = True

        if active and not val:
            if i - start_idx >= min_frames:
                wave_starts.append(start_idx)
            active = False

    wave_detected=False
    if len(wave_starts) > 0:
        wave_detected=True

    if wave_detected:
        return [{"name": "reverberation/rough Sound", "times": []}]
    return []



def analyse_volume(y, sr, audio_file):
    # Frame parameters
    frame_length = int(0.1 * sr)
    hop_length = int(0.05 * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_smooth = scipy.signal.medfilt(rms, kernel_size=7)

    noise_sec = 0.2
    noise_frames = int(noise_sec * sr / hop_length)

    noise_th = max(2 * np.max(rms_smooth[:noise_frames]), 0.05 * np.max(rms_smooth))

    rms_no_noise = rms_smooth.copy()
    rms_no_noise[rms_no_noise < noise_th] = 0

    active = rms_no_noise[rms_no_noise > 0]
    mean_amp = np.mean(active)
    max_allowed = min(0.9 * np.max(active), 0.9)
    active_usage = active[active < max_allowed]

    th_low = otsu_threshold(active_usage)

    # Transition detection
    margin_sec = 0.3
    margin_frames = int(margin_sec / (hop_length / sr))
    long_margin_sec = 1
    long_frames = int(long_margin_sec / (hop_length / sr))

    is_high = False
    low_idx = []

    for idx in range(margin_frames, len(rms_smooth) - margin_frames):
        vol = rms_smooth[idx]

        # Noise check in margin
        local = rms_smooth[idx - margin_frames: idx + margin_frames + 1]
        if np.any(local < noise_th):
            is_high = vol > th_low
            continue

        # Long-term difference
        lf_start = max(0, idx - long_frames)
        lf_end = min(len(rms_smooth), idx + long_frames)
        before = rms_smooth[lf_start:idx]
        after = rms_smooth[idx:lf_end]
        before = before[before >= noise_th]
        after = after[after >= noise_th]

        if len(before) == 0 or len(after) == 0:
            is_high = vol > th_low
            continue

        diff = abs(np.mean(after) - np.mean(before))
        if diff < 0.5 * mean_amp:
            is_high = vol > th_low
            continue

        # Two-threshold logic
        new_is_high = vol > th_low
        if (not new_is_high) and is_high:
            low_idx.append(idx)
            is_high = new_is_high
        else:
            is_high = new_is_high

    low_times = librosa.frames_to_time(low_idx, sr=sr, hop_length=hop_length)

    # Check faster drops
    fast_low_idx = []
    drms = np.diff(rms_smooth)
    if len(drms) > 0:
        d_min = np.min(drms)
        if d_min < 0:
            th_drop = 0.1 * d_min
            for i, d in enumerate(drms):
                if d <= th_drop:
                    fast_low_idx.append(i + 1)

    return [{"name": "Volume down", "times": low_times}]

# -----------------------------------------------------
# clicking detection
# -----------------------------------------------------
def analyse_clicking1(y, sr, audio_file):
    frame_length = int(0.005 * sr)
    hop_length = int(0.003 * sr)
    max_amp = np.max(np.abs(y))
    low_thresh = 0.01 * max_amp
    very_low_thresh = min(0.01 * max_amp, 0.0003)
    high_thresh = 0.05 * max_amp
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)
    indices = []
    for i in range(3, len(max_abs_per_frame) - 3):
        current = max_abs_per_frame[i]
        if current <= low_thresh:
            prev = max_abs_per_frame[i-3:i]
            next_ = max_abs_per_frame[i+1:i+4]
            if np.max(prev) >= high_thresh and np.max(next_) >= high_thresh:
                indices.append(i)
    #for bigger pauses
    for i in range(14, len(max_abs_per_frame) - 22):
        current = max_abs_per_frame[i:i+7]  # slice statt 2D-Index
        if all(current <= very_low_thresh):
            prev = max_abs_per_frame[i-14:i]
            next_ = max_abs_per_frame[i+8:i+22]
            if np.max(prev) >= high_thresh and np.max(next_) >= high_thresh:
                indices.append(i)
    times_selected = librosa.frames_to_time(indices, sr=sr, hop_length=hop_length)
    # Group if <1ms apart
    if len(times_selected) > 1:
        grouped = []
        buffer = [times_selected[0]]
        for t in times_selected[1:]:
            if (t - buffer[-1]) <= 0.01:
                buffer.append(t)
            else:
                grouped.append(np.mean(buffer))
                buffer = [t]
        grouped.append(np.mean(buffer))
        times_selected = np.array(grouped)
    return [{"name": "Clicking1", "times": times_selected}]


def analyse_clicking2(y, sr, audio_file):
    cutoff = 10000
    nyquist = sr / 2
    cutoff = min(cutoff, nyquist*0.99)
    b, a = scipy.signal.butter(1, cutoff/nyquist , 'high')
    y_hp = scipy.signal.filtfilt(b, a, y)
    diff = np.abs(np.diff(y_hp))
    noise = 0.15 * max(np.abs(y))
    knacks2 = []
    block_size = 500
    window_size = 1000
    thresholds = np.zeros(len(diff))
    little_thresholds = np.zeros(len(diff))
    for start in range(0, len(diff), block_size):
        end = min(len(diff), start + block_size)
        w_start = max(0, start - window_size//2)
        w_end = min(len(diff), end + window_size//2)
        local_median = np.median(diff[w_start:w_end])
        local_max = np.max(diff[w_start:w_end])
        thresholds[start:end] = 50 * local_median
        little_thresholds[start:end] = 0.25 * local_max
        context_sec = 0.01  # checking noise theshold 10ms befor and after
        context_frames = int(context_sec * sr)
    for i in range(50, len(diff)-50):
        if diff[i] > thresholds[i] and diff[i] > little_thresholds[i]:
            before = y[max(0, i-context_frames):i]
            after  = y[i:i+context_frames]
            if np.mean(np.abs(before)) > noise/2 and np.mean(np.abs(after)) > noise/2:
                if diff[i-1] < thresholds[i-1] and diff[i+1] < thresholds[i+1]:
                    if np.sign(y_hp[i-1]) != np.sign(y_hp[i]) or np.sign(y_hp[i]) != np.sign(y_hp[i+1]):
                        if np.max(np.abs(y[i-50:i+51])) > noise:
                            knacks2.append(i)
    knacks2 = np.array(knacks2)
    times_selected = np.round(knacks2 / sr, 4)
    # Group if <1ms apart
    if len(times_selected) > 1:
        grouped = []
        buffer = [times_selected[0]]
        for t in times_selected[1:]:
            if (t - buffer[-1]) <= 0.01:
                buffer.append(t)
            else:
                grouped.append(np.mean(buffer))
                buffer = [t]
        grouped.append(np.mean(buffer))
        times_selected = np.array(grouped)
    return [{"name": "Clicking2", "times": times_selected}]

# -----------------------------------------------------
# Overload detection
# -----------------------------------------------------
def analyse_clipping(y, sr, audio_file):
    frame_length = int(0.05 * sr)
    hop_length = int(0.05 * sr)
    max_amp = np.max(np.abs(y))
    duration=0.5
    duration_frames=int(duration * sr / hop_length)
    if max_amp >= 1.0:
        high_thresh = 1.0
        low_thresh = 0.98
    else:
        high_thresh = 0.95
        low_thresh = 0.93
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    max_abs_per_frame = np.max(np.abs(frames), axis=0)
    ueber = []
    ueber_end = []
    high = False
    for i in range(0, len(max_abs_per_frame)-duration_frames):
        if high_thresh==1.0:
            if any(max_abs_per_frame[i:i+duration_frames] >= high_thresh) and all(max_abs_per_frame[i:i+duration_frames] >= low_thresh) and high is False:
                ueber.append(i)
                high = True
            if max_abs_per_frame[i] < low_thresh and high is True:
                high = False
                ueber_end.append(i)
        else: 
            if any(max_abs_per_frame[i:i+duration_frames] >= high_thresh) and all(max_abs_per_frame[i:i+duration_frames] >= low_thresh) and high is False:
                ueber.append(i)
                high = True
            if max_abs_per_frame[i] < low_thresh and high is True:
                high = False
                ueber_end.append(i)
    ueber_times = librosa.frames_to_time(ueber, sr=sr, hop_length=hop_length)
    ueber_end_times = librosa.frames_to_time(ueber_end, sr=sr, hop_length=hop_length)
    return [{"name": "Clipping Start", "times": ueber_times},
            {"name": "Clipping End", "times": ueber_end_times}]

# -----------------------------------------------------
# Noise detection
# -----------------------------------------------------
def analyse_noise(y, sr, audio_file):
    events = []
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    low = np.percentile(rms, 10)
    high = np.percentile(rms, 90)
    if low <= 1e-10:
        return []
    dynamic_range_db = 20 * np.log10(high / low)
    if dynamic_range_db < 20:
        events.append({"name": "Noise", "times": []})
    return events

# -----------------------------------------------------
# Overview plot
# -----------------------------------------------------
def plot_overview(y, sr, events, audio_file):
    times = np.arange(len(y)) / sr
    plt.figure(figsize=(14, 5))
    plt.plot(times, y, color="lightgray", label="Waveform")
    color_map = {
        "Volume down": "green",
        "Clicking1": "orange",
        "Clicking2": "yellow",
        "Clipping Start": "purple",
        "Clipping End": "blue"}
    plotted_labels = set()
    warning_names = ["No Signal", "Very quiet", "Noise", "reverberation/rough Sound"]
    warnings = []
    for ev in events:
        name = ev["name"]
        color = color_map.get(name, "black")
        if name in warning_names and name not in warnings:
            warnings.append(name)
        y_pos = 0.95
        for w in warnings:
            plt.text(0.01, y_pos, w, transform=plt.gca().transAxes, fontsize=12, fontweight="bold", color="red")
            y_pos -= 0.06
        for t in ev["times"]:
            if name not in plotted_labels:
                plt.axvline(t, color=color, linestyle="--", alpha=0.8, label=name)
                plotted_labels.add(name)
            else:
                plt.axvline(t, color=color, linestyle="--", alpha=0.8)
            plt.text(t, 0.98, f"{t:.2f}s", rotation=90, verticalalignment="top",
                     horizontalalignment="right", transform=plt.gca().get_xaxis_transform(),
                     fontsize=8, color=color, alpha=0.8)
    plt.title(audio_file)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(output_folder, f"{audio_file}_overview.png")
    plt.savefig(out)
    plt.close()
    print("➡ Overview plot saved:", out)

# -----------------------------------------------------
# Summary table
# -----------------------------------------------------
def make_summary_table(audio_file, events):
    row = {
        "File": audio_file,
        "No Signal": "",
        "Very quiet": "",
        "Noise": "",
        "Clipping": "",
        "Clicking1": "",
        "Clicking2": "",
        "Volume": "",
        "reverberation/rough Sound": ""
    }
    clipping_start = []
    clicking1 = []
    clicking2 = []
    louder = []
    quieter = []
    for ev in events:
        name = ev["name"]
        times = ev.get("times", [])
        if name == "No Signal":
            row["No Signal"] = "No Signal"
        if name == "Very quiet":
            row["Very quiet"] = "Very quiet"
        if name == "Noise":
            row["Noise"] = "Noise detected"
        if name == "reverberation/rough Sound":
            row["reverberation/rough Sound"] = "reverberation/rough Sound detected"
        if name == "Clicking1" and len(times) > 0:
            clicking1.extend(times)
        if name == "Clicking2" and len(times) > 0:
            clicking2.extend(times)
        if name == "Clipping Start" and len(times) > 0:
            clipping_start.extend(times)
        if name == "Volume up" and len(times) > 0:
            louder.extend(times)
        if name == "Volume down" and len(times) > 0:
            quieter.extend(times)
    if clicking1:
        row["Clicking1"] = ", ".join(f"{t:.2f}" for t in clicking1)
    if clicking2:
        row["Clicking2"] = ", ".join(f"{t:.2f}" for t in clicking2)
    if clipping_start:
        row["Clipping"] = "Start times:\n" + ", ".join(f"{t:.2f}" for t in clipping_start)
    if louder or quieter:
        txt = []
        if louder:
            txt.append("Louder:\n" + ", ".join(f"{t:.2f}" for t in louder))
        if quieter:
            txt.append("Quieter:\n" + ", ".join(f"{t:.2f}" for t in quieter))
        row["Volume"] = "\n".join(txt)
    return row

# -----------------------------------------------------
# Run all analyses
# -----------------------------------------------------
def run_all_analyses(filepath):
    audio_file = os.path.basename(filepath)
    print("\nProcessing:", audio_file)
    y, sr = load_audio(filepath)
    skip, events = check_no_signal(y, sr)
    all_events = []
    all_events += events
    if skip:
        plot_overview(y, sr, all_events, audio_file)
        summary_results.append(make_summary_table(audio_file, all_events))
        return
    all_events += analyse_volume(y, sr, audio_file)
    all_events += analyse_clicking1(y, sr, audio_file)
    all_events += analyse_clicking2(y,sr, audio_file)
    all_events += analyse_clipping(y, sr, audio_file)
    all_events += analyse_noise(y, sr, audio_file)
    all_events += analyse_reverberation(y, sr, audio_file)

    plot_overview(y, sr, all_events, audio_file)
    summary_results.append(make_summary_table(audio_file, all_events))



# -----------------------------------------------------
# Find audio files in input folder
# -----------------------------------------------------
#solution 1 audio_files in on folder:
#audio_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)])
#if not audio_files:
#    print("No audio files found in the input folder!")
#else:
#    print("Found audio files:", audio_files)
#for filename in audio_files:
#    try:
#        run_all_analyses(os.path.join(input_folder, filename))
#    except Exception as e:
#        print("error in:", filename)
#        print(e)
#print("\nFertig!")

#or audiofiles in subfolders in a folder:
audio_files_2 = []
for subdir, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(valid_ext):
            audio_files_2.append(os.path.join(subdir, file))

if not audio_files_2:
    print("No audio files found in the input folder!")
else:
    print("Found audio files:", audio_files_2)

for filename in audio_files_2:
    try:
        run_all_analyses(os.path.join(input_folder, filename))
    except Exception as e:
        print("error in:", filename)
        print(e)


if summary_results:
    df = pd.DataFrame(summary_results)

    def color_cells(val):
        if val is None or val == "":
            return "background-color: #c6efce"  # green
        return "background-color: #ffc7ce"      # red

    styled = df.style.applymap(
        color_cells,
        subset=df.columns[1:]
    )

    out_xlsx = os.path.join(output_folder, "summary.xlsx")
    styled.to_excel(out_xlsx, engine="openpyxl", index=False)

    print("➡Overview:", out_xlsx)


print("\nFinished!")
