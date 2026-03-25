import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
root = "/Users/sahas_biddappa/Desktop/freq_resp/horizontal"   # change this
sensor_key = "d7_84"                                          # change this
channel_col = "Z"
fs = 100.0
V_INPUT = 5.0                                                 # shaker velocity input (mm/s)
device_label = "horizontal"                                   # change this
trim_cycles = 2                                               # trim this many cycles from start and end
fft_plot_max_hz = 2.0                                         # shown in spectrum plot
dip_check_min_hz = 0.6
dip_check_max_hz = 1.0

# =========================
# HELPERS
# =========================
def safe_name(x):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(x))

def extract_nominal_freq(folder_name):
    s = folder_name.lower().replace("hz", "")
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    if not nums:
        return None
    return float(nums[-1])

def get_fft_response(x, fs, f0):
    """
    Extract amplitude and phase from FFT at the nearest bin to f0.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 16:
        return np.nan, np.nan, None, None, None

    win = np.hanning(n)
    xw = x * win
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1/fs)

    k = np.argmin(np.abs(f - f0))
    complex_val = (2.0 / np.sum(win)) * X[k]

    amp = np.abs(complex_val)
    phase_deg = np.degrees(np.angle(complex_val))

    return amp, phase_deg, f, X, k

def estimate_dominant_frequency(x, fs, fmin=0.03, fmax=2.0):
    """
    Estimate dominant frequency from FFT peak in a limited band.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 16:
        return np.nan

    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(n, d=1/fs)

    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return np.nan

    mag = np.abs(X[mask])
    f_sel = f[mask]
    return f_sel[np.argmax(mag)]

def sine_fit_amplitude(x, fs, f0):
    """
    Fit x(t) = a*sin(2*pi*f0*t) + b*cos(2*pi*f0*t)
    Return amplitude and phase.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 16 or not np.isfinite(f0):
        return np.nan, np.nan

    t = np.arange(n) / fs
    A = np.column_stack([
        np.sin(2 * np.pi * f0 * t),
        np.cos(2 * np.pi * f0 * t)
    ])

    coeff, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    a, b = coeff

    amp = np.sqrt(a**2 + b**2)
    phase_deg = np.degrees(np.arctan2(b, a))

    return amp, phase_deg

# =========================
# MAIN
# =========================
master_rows = []

folders = [d for d in os.listdir(root) if "hz" in d.lower()]
folders = sorted(
    folders,
    key=lambda x: extract_nominal_freq(x) if extract_nominal_freq(x) is not None else 1e9
)

for folder in folders:
    nominal_freq = extract_nominal_freq(folder)
    if nominal_freq is None:
        print(f"Skipping {folder}: could not parse frequency")
        continue

    folder_path = os.path.join(root, folder)
    if not os.path.isdir(folder_path):
        continue

    matches = [
        f for f in os.listdir(folder_path)
        if sensor_key in f
        and f.lower().endswith(".csv")
        and "_fft_stats" not in f.lower()
        and "_stats" not in f.lower()
        and "_master_summary" not in f.lower()
    ]

    if not matches:
        print(f"Skipping {folder}: no matching CSV for {sensor_key}")
        continue

    csv_file = matches[0]
    csv_path = os.path.join(folder_path, csv_file)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Skipping {folder}: read error -> {e}")
        continue

    if channel_col not in df.columns:
        print(f"Skipping {folder}: column {channel_col} not found")
        continue

    sig = pd.to_numeric(df[channel_col], errors="coerce").dropna().values
    if len(sig) < 100:
        print(f"Skipping {folder}: too few samples")
        continue

    # -------------------------
    # Cycle-based trimming
    # -------------------------
    trim_samples = int(trim_cycles * fs / nominal_freq)

    if len(sig) > 2 * trim_samples + 100:
        sig_trim = sig[trim_samples:-trim_samples]
    else:
        sig_trim = sig.copy()

    duration_sec = len(sig_trim) / fs
    time = np.arange(len(sig_trim)) / fs
    mean_val = float(np.mean(sig_trim))
    std_val = float(np.std(sig_trim, ddof=0))
    rms_val = float(np.sqrt(np.mean((sig_trim - np.mean(sig_trim)) ** 2)))
    peak_val = float(np.max(sig_trim))
    valley_val = float(np.min(sig_trim))
    ptp_val = float(peak_val - valley_val)

    # Time-domain amplitude and sensitivity
    td_amp = ptp_val / 2.0
    td_sens = td_amp / V_INPUT

    # Dominant frequency estimate
    measured_freq = estimate_dominant_frequency(
        sig_trim,
        fs,
        fmin=0.03,
        fmax=max(2.0, nominal_freq * 2)
    )

    # FFT at nominal frequency
    fft_amp_nom, fft_phase_nom, f_axis, X_fft, k_nom = get_fft_response(sig_trim, fs, nominal_freq)

    # FFT at measured dominant frequency
    if np.isfinite(measured_freq):
        fft_amp_meas, fft_phase_meas, _, _, k_meas = get_fft_response(sig_trim, fs, measured_freq)
    else:
        fft_amp_meas, fft_phase_meas, k_meas = np.nan, np.nan, None

    sens_nom = fft_amp_nom / V_INPUT if np.isfinite(fft_amp_nom) else np.nan
    sens_meas = fft_amp_meas / V_INPUT if np.isfinite(fft_amp_meas) else np.nan

    # Sine-fit at measured dominant frequency
    if np.isfinite(measured_freq):
        sine_amp_meas, sine_phase_meas = sine_fit_amplitude(sig_trim, fs, measured_freq)
    else:
        sine_amp_meas, sine_phase_meas = np.nan, np.nan

    sine_sens_meas = sine_amp_meas / V_INPUT if np.isfinite(sine_amp_meas) else np.nan

    # Frequency error
    freq_error_hz = measured_freq - nominal_freq if np.isfinite(measured_freq) else np.nan
    freq_error_pct = (100.0 * freq_error_hz / nominal_freq) if np.isfinite(freq_error_hz) and nominal_freq != 0 else np.nan

    # -------------------------
    # Save per-frequency stats CSV
    # -------------------------
    fft_stats_df = pd.DataFrame([{
        "Folder": folder,
        "CSV_File": csv_file,
        "Sensor_Key": sensor_key,
        "Channel": channel_col,
        "Nominal_Frequency_Hz": nominal_freq,
        "Measured_Frequency_Hz": measured_freq,
        "Freq_Error_Hz": freq_error_hz,
        "Freq_Error_pct": freq_error_pct,
        "Trim_Cycles": trim_cycles,
        "Trim_Samples_each_side": trim_samples,
        "Duration_s": duration_sec,
        "Samples": len(sig_trim),
        "Mean": mean_val,
        "Std": std_val,
        "RMS_centered": rms_val,
        "Peak": peak_val,
        "Valley": valley_val,
        "Peak_to_Peak": ptp_val,
        "TimeDomain_Amplitude": td_amp,
        "TimeDomain_Sensitivity": td_sens,
        "FFT_Amplitude_at_Nominal": fft_amp_nom,
        "FFT_Phase_deg_at_Nominal": fft_phase_nom,
        "FFT_Sensitivity_at_Nominal": sens_nom,
        "FFT_Amplitude_at_Measured": fft_amp_meas,
        "FFT_Phase_deg_at_Measured": fft_phase_meas,
        "FFT_Sensitivity_at_Measured": sens_meas,
        "SineFit_Amplitude_at_Measured": sine_amp_meas,
        "SineFit_Phase_deg_at_Measured": sine_phase_meas,
        "SineFit_Sensitivity_at_Measured": sine_sens_meas
    }])

    stats_csv_path = os.path.join(folder_path, f"{safe_name(csv_file[:-4])}_fft_stats.csv")
    fft_stats_df.to_csv(stats_csv_path, index=False)

    # -------------------------
    # Save per-frequency TXT report
    # -------------------------
    txt_path = os.path.join(folder_path, f"{safe_name(csv_file[:-4])}_fft_report.txt")
    with open(txt_path, "w") as f:
        f.write("FFT-Based Frequency Response Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"File: {csv_file}\n")
        f.write(f"Device/Test Label: {device_label}\n")
        f.write(f"Sensor Key: {sensor_key}\n")
        f.write(f"Folder: {folder}\n")
        f.write(f"Nominal Frequency: {nominal_freq:.6f} Hz\n")
        f.write(f"Measured Dominant Frequency: {measured_freq:.6f} Hz\n" if np.isfinite(measured_freq) else "Measured Dominant Frequency: nan\n")
        f.write(f"Frequency Error: {freq_error_hz:.6f} Hz ({freq_error_pct:.3f} %)\n" if np.isfinite(freq_error_hz) else "Frequency Error: nan\n")
        f.write(f"Sampling Rate: {fs:.1f} Hz\n")
        f.write(f"Duration: {duration_sec:.3f} s\n")
        f.write(f"Samples: {len(sig_trim)}\n")
        f.write("\n")
        f.write("PREPROCESSING:\n")
        f.write("  DC offset removed: Yes\n")
        f.write(f"  Trimmed cycles at start/end: {trim_cycles}\n")
        f.write(f"  Trimmed samples at start/end: {trim_samples}\n")
        f.write("  Window used for FFT: Hann\n")
        f.write("\n")
        f.write("TIME-DOMAIN SUMMARY:\n")
        f.write(f"  Mean: {mean_val:,.3f}\n")
        f.write(f"  Std: {std_val:,.3f}\n")
        f.write(f"  RMS (centered): {rms_val:,.3f}\n")
        f.write(f"  Peak: {peak_val:,.3f}\n")
        f.write(f"  Valley: {valley_val:,.3f}\n")
        f.write(f"  Peak-to-Peak: {ptp_val:,.3f}\n")
        f.write(f"  Time-domain Amplitude (P-P/2): {td_amp:,.3f}\n")
        f.write(f"  Time-domain Sensitivity: {td_sens:,.3f}\n")
        f.write("\n")
        f.write("FFT RESPONSE AT NOMINAL FREQUENCY:\n")
        f.write(f"  FFT Amplitude: {fft_amp_nom:,.3f}\n")
        f.write(f"  FFT Phase: {fft_phase_nom:,.3f} deg\n")
        f.write(f"  FFT Sensitivity: {sens_nom:,.3f}\n")
        f.write("\n")
        f.write("FFT RESPONSE AT MEASURED DOMINANT FREQUENCY:\n")
        f.write(f"  FFT Amplitude: {fft_amp_meas:,.3f}\n" if np.isfinite(fft_amp_meas) else "  FFT Amplitude: nan\n")
        f.write(f"  FFT Phase: {fft_phase_meas:,.3f} deg\n" if np.isfinite(fft_phase_meas) else "  FFT Phase: nan\n")
        f.write(f"  FFT Sensitivity: {sens_meas:,.3f}\n" if np.isfinite(sens_meas) else "  FFT Sensitivity: nan\n")
        f.write("\n")
        f.write("SINE-FIT RESPONSE AT MEASURED DOMINANT FREQUENCY:\n")
        f.write(f"  Sine-Fit Amplitude: {sine_amp_meas:,.3f}\n" if np.isfinite(sine_amp_meas) else "  Sine-Fit Amplitude: nan\n")
        f.write(f"  Sine-Fit Phase: {sine_phase_meas:,.3f} deg\n" if np.isfinite(sine_phase_meas) else "  Sine-Fit Phase: nan\n")
        f.write(f"  Sine-Fit Sensitivity: {sine_sens_meas:,.3f}\n" if np.isfinite(sine_sens_meas) else "  Sine-Fit Sensitivity: nan\n")
        f.write("\n")
        f.write("METHOD SUMMARY:\n")
        f.write("  1. DC offset removed.\n")
        f.write("  2. Start/end transients removed using cycle-based trimming.\n")
        f.write("  3. FFT used to estimate dominant frequency.\n")
        f.write("  4. Amplitude compared using:\n")
        f.write("     - Time-domain peak-to-peak / 2\n")
        f.write("     - FFT at nominal frequency\n")
        f.write("     - FFT at measured dominant frequency\n")
        f.write("     - Sine-fit at measured dominant frequency\n")

    # -------------------------
    # Save per-frequency PNG report
    # -------------------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[2.0, 1.0])

    ax_top = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_fft = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"FFT-Based Response Report ({nominal_freq:.2f} Hz Test) - {duration_sec:.0f}s | File: {csv_file}",
        fontsize=15,
        y=0.98
    )

    # Full waveform
    ax_top.plot(time, sig_trim, color="blue", linewidth=0.8)
    ax_top.axhline(mean_val, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.0f}")
    ax_top.set_title("Raw Waveform")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Amplitude")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="lower right")

    # Zoom waveform
    zoom_sec = min(30, duration_sec)
    n_zoom = int(zoom_sec * fs)
    ax_zoom.plot(time[:n_zoom], sig_trim[:n_zoom], color="blue", linewidth=1.0)
    ax_zoom.set_title(f"Zoom: First ~{zoom_sec:.0f} seconds")
    ax_zoom.set_xlabel("Time (s)")
    ax_zoom.set_ylabel("Amplitude")
    ax_zoom.grid(True, alpha=0.25)

    # FFT spectrum
    if f_axis is not None and X_fft is not None:
        mag_fft = np.abs(X_fft)
        mask = f_axis <= fft_plot_max_hz
        ax_fft.plot(f_axis[mask], mag_fft[mask], color="green", linewidth=1.5)

        ax_fft.axvline(nominal_freq, color="black", linestyle="--", alpha=0.7, label=f"Nominal: {nominal_freq:.3f} Hz")
        if np.isfinite(measured_freq):
            ax_fft.axvline(measured_freq, color="red", linestyle=":", alpha=0.7, label=f"Measured: {measured_freq:.3f} Hz")

    ax_fft.set_title(f"Frequency Spectrum (0-{fft_plot_max_hz:.1f} Hz)")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.grid(True, alpha=0.25)
    ax_fft.legend()

    info_text = (
        f"Nominal f = {nominal_freq:.3f} Hz\n"
        f"Measured f = {measured_freq:.3f} Hz\n"
        f"Freq error = {freq_error_hz:.4f} Hz\n"
        f"TD Sens = {td_sens:,.1f}\n"
        f"FFT Sens @ nominal = {sens_nom:,.1f}\n"
        f"FFT Sens @ measured = {sens_meas:,.1f}\n"
        f"Sine-fit Sens = {sine_sens_meas:,.1f}"
    )
    ax_zoom.text(
        1.02, 0.98, info_text,
        transform=ax_zoom.transAxes,
        fontsize=10.0,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(folder_path, f"{safe_name(csv_file[:-4])}_fft_chart.png")
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Add to master summary
    # -------------------------
    master_rows.append({
        "Folder": folder,
        "CSV_File": csv_file,
        "Sensor_Key": sensor_key,
        "Channel": channel_col,
        "Nominal_Frequency_Hz": nominal_freq,
        "Measured_Frequency_Hz": measured_freq,
        "Freq_Error_Hz": freq_error_hz,
        "Freq_Error_pct": freq_error_pct,
        "Trim_Cycles": trim_cycles,
        "Trim_Samples_each_side": trim_samples,
        "Duration_s": duration_sec,
        "Samples": len(sig_trim),
        "Mean": mean_val,
        "Std": std_val,
        "RMS_centered": rms_val,
        "Peak": peak_val,
        "Valley": valley_val,
        "Peak_to_Peak": ptp_val,
        "TimeDomain_Amplitude": td_amp,
        "TimeDomain_Sensitivity": td_sens,
        "FFT_Amplitude_at_Nominal": fft_amp_nom,
        "FFT_Phase_deg_at_Nominal": fft_phase_nom,
        "FFT_Sensitivity_at_Nominal": sens_nom,
        "FFT_Amplitude_at_Measured": fft_amp_meas,
        "FFT_Phase_deg_at_Measured": fft_phase_meas,
        "FFT_Sensitivity_at_Measured": sens_meas,
        "SineFit_Amplitude_at_Measured": sine_amp_meas,
        "SineFit_Phase_deg_at_Measured": sine_phase_meas,
        "SineFit_Sensitivity_at_Measured": sine_sens_meas,
        "FFT_Stats_CSV": os.path.basename(stats_csv_path),
        "FFT_Report_TXT": os.path.basename(txt_path),
        "FFT_Report_PNG": os.path.basename(png_path),
    })

    print(f"Done: {folder}")

# =========================
# SAVE MASTER SUMMARY
# =========================
if master_rows:
    master_df = pd.DataFrame(master_rows).sort_values("Nominal_Frequency_Hz")
    master_csv_path = os.path.join(root, f"{device_label}_{sensor_key}_fft_master_summary.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"\nMaster summary saved to: {master_csv_path}")

    # =========================
    # DIP-REGION CHECK TABLE
    # =========================
    dip_region = master_df[
        (master_df["Nominal_Frequency_Hz"] >= dip_check_min_hz) &
        (master_df["Nominal_Frequency_Hz"] <= dip_check_max_hz)
    ].copy()

    if not dip_region.empty:
        print(f"\n=== Dip-region check ({dip_check_min_hz:.2f} to {dip_check_max_hz:.2f} Hz) ===")
        cols = [
            "Nominal_Frequency_Hz",
            "Measured_Frequency_Hz",
            "Freq_Error_Hz",
            "Freq_Error_pct",
            "TimeDomain_Sensitivity",
            "FFT_Sensitivity_at_Nominal",
            "FFT_Sensitivity_at_Measured",
            "SineFit_Sensitivity_at_Measured"
        ]
        print(dip_region[cols].to_string(index=False))

    # =========================
    # FINAL RESPONSE PLOT
    # no smoothing used here
    # =========================
    plot_df = master_df.dropna(subset=[
        "Nominal_Frequency_Hz",
        "TimeDomain_Sensitivity",
        "FFT_Sensitivity_at_Nominal",
        "FFT_Sensitivity_at_Measured",
        "SineFit_Sensitivity_at_Measured",
        "FFT_Phase_deg_at_Measured"
    ]).copy()

    if len(plot_df) >= 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

        freqs = plot_df["Nominal_Frequency_Hz"].values
        sens_td = plot_df["TimeDomain_Sensitivity"].values
        sens_nom = plot_df["FFT_Sensitivity_at_Nominal"].values
        sens_meas = plot_df["FFT_Sensitivity_at_Measured"].values
        sens_sine = plot_df["SineFit_Sensitivity_at_Measured"].values
        phase = plot_df["FFT_Phase_deg_at_Measured"].values

        # unwrap phase
        phase_unwrapped = np.degrees(np.unwrap(np.radians(phase)))

        # Raw comparison plots only
        ax1.loglog(freqs, sens_td, 'd-.', linewidth=1.8, markersize=5, label='Time-domain P-P/2')
        ax1.loglog(freqs, sens_nom, 's--', linewidth=1.8, markersize=5, label='FFT @ nominal bin')
        ax1.loglog(freqs, sens_meas, 'o-', linewidth=2.0, markersize=6, label='FFT @ measured bin')
        ax1.loglog(freqs, sens_sine, '^-', linewidth=1.8, markersize=5, label='Sine fit @ measured f')

        ax2.semilogx(freqs, phase_unwrapped, 'o-', linewidth=2.0, markersize=6, label='Phase @ measured bin')

        ax1.set_title(f"Frequency Response Validation: {device_label} ({sensor_key})", fontsize=14)
        ax1.set_ylabel("Sensitivity", fontweight="bold")
        ax1.grid(True, which="both", alpha=0.2)
        ax1.legend()

        ax2.set_ylabel("Phase (degrees)", fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)", fontweight="bold")
        ax2.grid(True, which="both", alpha=0.2)
        ax2.legend()

        plt.tight_layout()
        final_plot_path = os.path.join(root, f"{device_label}_{sensor_key}_fft_frequency_response_validation.png")
        plt.savefig(final_plot_path, dpi=180, bbox_inches="tight")
        plt.show()

        print(f"Final validation plot saved to: {final_plot_path}")
    else:
        print("Not enough valid frequency points to create final response plot.")
else:
    print("No valid data processed.")