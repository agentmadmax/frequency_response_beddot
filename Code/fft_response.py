import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# =========================================================
# CONFIGURATION
# =========================================================

# Project root = parent of Code folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(BASE_DIR, "Data")
RESULTS_ROOT = os.path.join(BASE_DIR, "Results")

CHANNEL_COL = "Z"
FS = 100.0
V_INPUT = 5.0
TRIM_SAMPLES = 1000
FFT_PLOT_MAX_HZ = 2.0

# Your 4 sensors
DIRECTIONS = {
    "horizontal": ["d7_84", "dd_94"],
    "vertical": ["cb_60", "db_a8"],
}

# Ideal 2nd-order high-pass reference parameters
FC_THEORY = 1.0
Q_THEORY = 0.707

# =========================================================
# HELPERS
# =========================================================

def safe_name(x):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))

def extract_nominal_freq(folder_name):
    """
    Extract the LAST number from folder name.
    Example:
      '1.55 0.1hz' -> 0.1
      '2.01 0.2hz' -> 0.2
    """
    s = folder_name.lower().replace("hz", "")
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if not nums:
        return None
    return float(nums[-1])

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clean_dir(path):
    """
    Delete a folder fully and recreate it.
    This ensures new results overwrite old results.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def get_fft_response_nominal_bin(x, fs, f0):
    """
    Nominal-bin FFT response:
    extract amplitude and phase at the FFT bin nearest to nominal excitation frequency.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)

    if n < 16:
        return np.nan, np.nan, None, None, None

    win = np.hanning(n)
    xw = x * win
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1 / fs)

    k = np.argmin(np.abs(f - f0))
    complex_val = (2.0 / np.sum(win)) * X[k]

    amp = np.abs(complex_val)
    phase_deg = np.degrees(np.angle(complex_val))
    return amp, phase_deg, f, X, k

def theoretical_hp2_response(freqs_hz, fc_hz, q, scale=1.0):
    """
    Ideal 2nd-order high-pass response:
    H(jw) = (jw/w0)^2 / [1 - (w/w0)^2 + j*(w/w0)/Q]
    Returns magnitude and wrapped phase in degrees.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    w = 2 * np.pi * freqs_hz
    w0 = 2 * np.pi * fc_hz
    s = 1j * w

    H = (s / w0)**2 / (1 - (w / w0)**2 + 1j * (w / w0) / q)

    mag = np.abs(H) * scale
    phase_deg = np.degrees(np.angle(H))
    return mag, phase_deg

def wrap_phase_deg(ph):
    return (ph + 180) % 360 - 180

def fit_theory_magnitude_scale(freqs_exp, sens_exp, fc_hz, q, fit_min_hz=2.0):
    """
    Auto-scale theoretical magnitude to experimental sensitivity.
    Prefer fitting in higher-frequency region if possible.
    """
    theory_mag, _ = theoretical_hp2_response(freqs_exp, fc_hz, q, scale=1.0)

    mask = (
        np.isfinite(freqs_exp) &
        np.isfinite(sens_exp) &
        np.isfinite(theory_mag) &
        (freqs_exp >= fit_min_hz) &
        (sens_exp > 0) &
        (theory_mag > 0)
    )

    if np.sum(mask) < 2:
        mask = (
            np.isfinite(freqs_exp) &
            np.isfinite(sens_exp) &
            np.isfinite(theory_mag) &
            (sens_exp > 0) &
            (theory_mag > 0)
        )

    if np.sum(mask) < 2:
        return 1.0

    log_ratio = np.log10(sens_exp[mask]) - np.log10(theory_mag[mask])
    return 10 ** np.median(log_ratio)

def fit_theory_phase_alignment(freqs_exp, phase_exp_deg, fc_hz, q):
    """
    Align theory phase to experimental phase using:
    - optional sign flip
    - constant phase offset
    Returns best_sign, best_offset_deg
    """
    _, theory_phase = theoretical_hp2_response(freqs_exp, fc_hz, q, scale=1.0)

    candidates = []
    for sign in [1, -1]:
        ph_trial = sign * theory_phase
        offset = np.median(wrap_phase_deg(phase_exp_deg - ph_trial))
        ph_aligned = wrap_phase_deg(ph_trial + offset)
        err = np.median(np.abs(wrap_phase_deg(phase_exp_deg - ph_aligned)))
        candidates.append((err, sign, offset))

    candidates.sort(key=lambda x: x[0])
    _, best_sign, best_offset = candidates[0]
    return best_sign, best_offset

# =========================================================
# CORE PROCESSING
# =========================================================

def process_single_sensor(direction, sensor_key):
    """
    Process one sensor across all frequency folders.
    Saves:
    - per-frequency CSV/TXT/PNG in Results/<direction>/<folder>/
    - master summary in Results/<direction>/
    - final response plot in Results/<direction>/
    """
    direction_data_root = os.path.join(DATA_ROOT, direction)
    direction_results_root = os.path.join(RESULTS_ROOT, direction)

    ensure_dir(direction_results_root)

    if not os.path.isdir(direction_data_root):
        print(f"[WARN] Missing data folder: {direction_data_root}")
        return None

    folders = [d for d in os.listdir(direction_data_root) if "hz" in d.lower()]
    folders = sorted(
        folders,
        key=lambda x: extract_nominal_freq(x) if extract_nominal_freq(x) is not None else 1e9
    )

    master_rows = []

    for folder in folders:
        nominal_freq = extract_nominal_freq(folder)
        if nominal_freq is None:
            print(f"[SKIP] {direction}/{folder}: could not parse frequency")
            continue

        folder_path = os.path.join(direction_data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        result_freq_dir = os.path.join(direction_results_root, folder)
        clean_dir(result_freq_dir)  # overwrite old results for this frequency folder

        matches = [
            f for f in os.listdir(folder_path)
            if sensor_key in f
            and f.lower().endswith(".csv")
            and "_fft_stats" not in f.lower()
            and "_stats" not in f.lower()
            and "_master_summary" not in f.lower()
        ]

        if not matches:
            print(f"[SKIP] {direction}/{folder}: no matching CSV for sensor {sensor_key}")
            continue

        csv_file = matches[0]
        csv_path = os.path.join(folder_path, csv_file)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] {direction}/{folder}: read error -> {e}")
            continue

        if CHANNEL_COL not in df.columns:
            print(f"[SKIP] {direction}/{folder}: column {CHANNEL_COL} not found")
            continue

        sig = pd.to_numeric(df[CHANNEL_COL], errors="coerce").dropna().values
        if len(sig) < 100:
            print(f"[SKIP] {direction}/{folder}: too few samples")
            continue

        if len(sig) > 2 * TRIM_SAMPLES + 100:
            sig_trim = sig[TRIM_SAMPLES:-TRIM_SAMPLES]
        else:
            sig_trim = sig.copy()

        duration_sec = len(sig_trim) / FS
        time = np.arange(len(sig_trim)) / FS
        mean_val = float(np.mean(sig_trim))
        std_val = float(np.std(sig_trim, ddof=0))
        rms_val = float(np.sqrt(np.mean((sig_trim - np.mean(sig_trim)) ** 2)))
        peak_val = float(np.max(sig_trim))
        valley_val = float(np.min(sig_trim))
        ptp_val = float(peak_val - valley_val)

        # NOMINAL BIN METHOD
        fft_amp_nom, fft_phase_nom, f_axis, X_fft, k_nom = get_fft_response_nominal_bin(
            sig_trim, FS, nominal_freq
        )
        sens_nom = fft_amp_nom / V_INPUT if np.isfinite(fft_amp_nom) else np.nan

        # -------------------------
        # Save per-frequency stats CSV
        # -------------------------
        stats_df = pd.DataFrame([{
            "Direction": direction,
            "Folder": folder,
            "CSV_File": csv_file,
            "Sensor_Key": sensor_key,
            "Channel": CHANNEL_COL,
            "Nominal_Frequency_Hz": nominal_freq,
            "Duration_s": duration_sec,
            "Samples": len(sig_trim),
            "Mean": mean_val,
            "Std": std_val,
            "RMS_centered": rms_val,
            "Peak": peak_val,
            "Valley": valley_val,
            "Peak_to_Peak": ptp_val,
            "FFT_Amplitude_at_Nominal": fft_amp_nom,
            "FFT_Phase_deg_at_Nominal": fft_phase_nom,
            "FFT_Sensitivity_at_Nominal": sens_nom
        }])

        stats_csv_path = os.path.join(
            result_freq_dir,
            f"{safe_name(sensor_key)}_{safe_name(csv_file[:-4])}_fft_stats.csv"
        )
        stats_df.to_csv(stats_csv_path, index=False)

        # -------------------------
        # Save per-frequency TXT report
        # -------------------------
        txt_path = os.path.join(
            result_freq_dir,
            f"{safe_name(sensor_key)}_{safe_name(csv_file[:-4])}_fft_report.txt"
        )
        with open(txt_path, "w") as f:
            f.write("FFT-Based Frequency Response Report (Nominal Bin Method)\n")
            f.write("=" * 68 + "\n")
            f.write(f"Direction: {direction}\n")
            f.write(f"File: {csv_file}\n")
            f.write(f"Sensor Key: {sensor_key}\n")
            f.write(f"Folder: {folder}\n")
            f.write(f"Nominal Frequency: {nominal_freq:.6f} Hz\n")
            f.write(f"Sampling Rate: {FS:.1f} Hz\n")
            f.write(f"Duration: {duration_sec:.3f} s\n")
            f.write(f"Samples: {len(sig_trim)}\n\n")

            f.write("PREPROCESSING:\n")
            f.write("  DC offset removed: Yes\n")
            f.write(f"  Trimmed samples at start/end: {TRIM_SAMPLES}\n")
            f.write("  Window used: Hann\n\n")

            f.write("TIME-DOMAIN SUMMARY:\n")
            f.write(f"  Mean: {mean_val:,.3f}\n")
            f.write(f"  Std: {std_val:,.3f}\n")
            f.write(f"  RMS (centered): {rms_val:,.3f}\n")
            f.write(f"  Peak: {peak_val:,.3f}\n")
            f.write(f"  Valley: {valley_val:,.3f}\n")
            f.write(f"  Peak-to-Peak: {ptp_val:,.3f}\n\n")

            f.write("FFT RESPONSE AT NOMINAL FREQUENCY:\n")
            f.write(f"  FFT Amplitude: {fft_amp_nom:,.3f}\n")
            f.write(f"  FFT Phase: {fft_phase_nom:,.3f} deg\n")
            f.write(f"  Sensitivity: {sens_nom:,.3f}\n\n")

            f.write("METHOD SUMMARY:\n")
            f.write("  The signal was DC-corrected, trimmed to remove transients, windowed using a Hann window,\n")
            f.write("  and transformed using FFT. The complex FFT coefficient nearest to the nominal excitation\n")
            f.write("  frequency was extracted, and its magnitude and angle were used to compute amplitude and phase.\n")

        # -------------------------
        # Save per-frequency PNG report
        # -------------------------
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[2.0, 1.0])

        ax_top = fig.add_subplot(gs[0, :])
        ax_zoom = fig.add_subplot(gs[1, 0])
        ax_fft = fig.add_subplot(gs[1, 1])

        fig.suptitle(
            f"Nominal-Bin FFT Report | {direction} | {sensor_key} | {nominal_freq:.2f} Hz",
            fontsize=15,
            y=0.98
        )

        ax_top.plot(time, sig_trim, linewidth=0.8)
        ax_top.axhline(mean_val, linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.0f}")
        ax_top.set_title("Raw Waveform")
        ax_top.set_xlabel("Time (s)")
        ax_top.set_ylabel("Amplitude")
        ax_top.grid(True, alpha=0.25)
        ax_top.legend(loc="lower right")

        zoom_sec = min(30, duration_sec)
        n_zoom = int(zoom_sec * FS)
        ax_zoom.plot(time[:n_zoom], sig_trim[:n_zoom], linewidth=1.0)
        ax_zoom.set_title(f"Zoom: First ~{zoom_sec:.0f} seconds")
        ax_zoom.set_xlabel("Time (s)")
        ax_zoom.set_ylabel("Amplitude")
        ax_zoom.grid(True, alpha=0.25)

        if f_axis is not None and X_fft is not None:
            mag_fft = np.abs(X_fft)
            mask = f_axis <= FFT_PLOT_MAX_HZ
            ax_fft.plot(f_axis[mask], mag_fft[mask], linewidth=1.5)
            ax_fft.axvline(
                nominal_freq,
                linestyle="--",
                alpha=0.7,
                label=f"Nominal: {nominal_freq:.3f} Hz"
            )

        ax_fft.set_title(f"Frequency Spectrum (0-{FFT_PLOT_MAX_HZ:.1f} Hz)")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude")
        ax_fft.grid(True, alpha=0.25)
        ax_fft.legend()

        info_text = (
            f"Nominal f = {nominal_freq:.3f} Hz\n"
            f"FFT Amp @ nominal = {fft_amp_nom:,.1f}\n"
            f"Phase @ nominal = {fft_phase_nom:.2f} deg\n"
            f"Sensitivity = {sens_nom:,.1f}"
        )
        ax_zoom.text(
            1.02, 0.98, info_text,
            transform=ax_zoom.transAxes,
            fontsize=10.5,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        png_path = os.path.join(
            result_freq_dir,
            f"{safe_name(sensor_key)}_{safe_name(csv_file[:-4])}_fft_chart.png"
        )
        plt.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        master_rows.append({
            "Direction": direction,
            "Folder": folder,
            "CSV_File": csv_file,
            "Sensor_Key": sensor_key,
            "Channel": CHANNEL_COL,
            "Nominal_Frequency_Hz": nominal_freq,
            "Duration_s": duration_sec,
            "Samples": len(sig_trim),
            "Mean": mean_val,
            "Std": std_val,
            "RMS_centered": rms_val,
            "Peak": peak_val,
            "Valley": valley_val,
            "Peak_to_Peak": ptp_val,
            "FFT_Amplitude_at_Nominal": fft_amp_nom,
            "FFT_Phase_deg_at_Nominal": fft_phase_nom,
            "FFT_Sensitivity_at_Nominal": sens_nom,
            "FFT_Stats_CSV": os.path.basename(stats_csv_path),
            "FFT_Report_TXT": os.path.basename(txt_path),
            "FFT_Report_PNG": os.path.basename(png_path),
        })

        print(f"[DONE] {direction}/{folder} | sensor {sensor_key}")

    if not master_rows:
        print(f"[WARN] No valid data processed for {direction} | sensor {sensor_key}")
        return None

    master_df = pd.DataFrame(master_rows).sort_values("Nominal_Frequency_Hz")
    master_csv_path = os.path.join(
        direction_results_root,
        f"{direction}_{sensor_key}_fft_master_summary.csv"
    )
    master_df.to_csv(master_csv_path, index=False)
    print(f"[SAVED] Master summary -> {master_csv_path}")

    # -------------------------
    # Final response plot
    # -------------------------
    plot_df = master_df.dropna(
        subset=["Nominal_Frequency_Hz", "FFT_Sensitivity_at_Nominal", "FFT_Phase_deg_at_Nominal"]
    ).copy()

    if len(plot_df) >= 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

        freqs = plot_df["Nominal_Frequency_Hz"].values
        sens = plot_df["FFT_Sensitivity_at_Nominal"].values
        phase = plot_df["FFT_Phase_deg_at_Nominal"].values
        phase_wrapped = wrap_phase_deg(phase)

        f_smooth = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 400)

        # Experimental magnitude
        ax1.loglog(freqs, sens, 'o', alpha=0.7, label=f"{direction} ({sensor_key}) nominal-bin")
        if len(freqs) >= 3:
            try:
                sens_interp = PchipInterpolator(freqs, sens)
                ax1.plot(f_smooth, sens_interp(f_smooth), linewidth=2.0)
            except Exception:
                ax1.plot(freqs, sens, linewidth=1.8)
        else:
            ax1.plot(freqs, sens, linewidth=1.8)

        # Experimental phase
        ax2.semilogx(freqs, phase_wrapped, 'o', alpha=0.7, label=f"{direction} ({sensor_key}) nominal-bin")
        if len(freqs) >= 3:
            try:
                phase_interp = PchipInterpolator(freqs, phase_wrapped)
                phase_smooth = wrap_phase_deg(phase_interp(f_smooth))
                ax2.plot(f_smooth, phase_smooth, linewidth=2.0)
            except Exception:
                ax2.plot(freqs, phase_wrapped, linewidth=1.8)
        else:
            ax2.plot(freqs, phase_wrapped, linewidth=1.8)

        # Aligned theory overlay
        best_scale = fit_theory_magnitude_scale(freqs, sens, FC_THEORY, Q_THEORY, fit_min_hz=2.0)
        best_sign, best_offset = fit_theory_phase_alignment(freqs, phase_wrapped, FC_THEORY, Q_THEORY)

        theory_mag, theory_phase = theoretical_hp2_response(
            f_smooth, FC_THEORY, Q_THEORY, scale=best_scale
        )
        theory_phase_aligned = wrap_phase_deg(best_sign * theory_phase + best_offset)

        ax1.loglog(
            f_smooth,
            theory_mag,
            '--',
            linewidth=2.2,
            label=f"Aligned ideal HP2 ref (fc={FC_THEORY:.2f}Hz, Q={Q_THEORY:.3f})"
        )

        ax2.semilogx(
            f_smooth,
            theory_phase_aligned,
            '--',
            linewidth=2.2,
            label="Aligned ideal HP2 ref"
        )

        ax1.set_title(f"Frequency Response | {direction} | {sensor_key}", fontsize=14)
        ax1.set_ylabel("Sensitivity", fontweight="bold")
        ax1.grid(True, which="both", alpha=0.2)
        ax1.legend()

        ax2.set_ylabel("Phase (degrees)", fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)", fontweight="bold")
        ax2.grid(True, which="both", alpha=0.2)
        ax2.legend()

        plt.tight_layout()
        final_plot_path = os.path.join(
            direction_results_root,
            f"{direction}_{sensor_key}_fft_frequency_response.png"
        )
        plt.savefig(final_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] Final response plot -> {final_plot_path}")

    return master_df

# =========================================================
# COMBINED PLOTS
# =========================================================

def make_combined_plots(all_master_dfs):
    """
    Creates combined plots for all sensors.
    Old combined results are deleted and recreated.
    """
    combined_root = os.path.join(RESULTS_ROOT, "combined")
    clean_dir(combined_root)

    if not all_master_dfs:
        print("[WARN] No master data available for combined plots.")
        return

    fig1, ax1 = plt.subplots(figsize=(11, 7))
    fig2, ax2 = plt.subplots(figsize=(11, 7))

    global_fmin = None
    global_fmax = None
    global_sens_freqs = []
    global_sens_vals = []
    global_phase_freqs = []
    global_phase_vals = []

    for label, master_df in all_master_dfs.items():
        plot_df = master_df.dropna(
            subset=["Nominal_Frequency_Hz", "FFT_Sensitivity_at_Nominal", "FFT_Phase_deg_at_Nominal"]
        ).copy()

        if len(plot_df) < 2:
            continue

        freqs = plot_df["Nominal_Frequency_Hz"].values
        sens = plot_df["FFT_Sensitivity_at_Nominal"].values
        phase = wrap_phase_deg(plot_df["FFT_Phase_deg_at_Nominal"].values)

        ax1.loglog(freqs, sens, 'o-', linewidth=1.8, markersize=5, label=label)
        ax2.semilogx(freqs, phase, 'o-', linewidth=1.8, markersize=5, label=label)

        global_sens_freqs.extend(freqs.tolist())
        global_sens_vals.extend(sens.tolist())
        global_phase_freqs.extend(freqs.tolist())
        global_phase_vals.extend(phase.tolist())

        fmin = np.min(freqs)
        fmax = np.max(freqs)
        global_fmin = fmin if global_fmin is None else min(global_fmin, fmin)
        global_fmax = fmax if global_fmax is None else max(global_fmax, fmax)

    if global_fmin is not None and global_fmax is not None:
        f_smooth = np.logspace(np.log10(global_fmin), np.log10(global_fmax), 400)

        global_sens_freqs = np.asarray(global_sens_freqs, dtype=float)
        global_sens_vals = np.asarray(global_sens_vals, dtype=float)
        global_phase_freqs = np.asarray(global_phase_freqs, dtype=float)
        global_phase_vals = np.asarray(global_phase_vals, dtype=float)

        best_scale = fit_theory_magnitude_scale(global_sens_freqs, global_sens_vals, FC_THEORY, Q_THEORY, fit_min_hz=2.0)
        best_sign, best_offset = fit_theory_phase_alignment(global_phase_freqs, global_phase_vals, FC_THEORY, Q_THEORY)

        theory_mag, theory_phase = theoretical_hp2_response(
            f_smooth, FC_THEORY, Q_THEORY, scale=best_scale
        )
        theory_phase_aligned = wrap_phase_deg(best_sign * theory_phase + best_offset)

        ax1.loglog(
            f_smooth,
            theory_mag,
            '--',
            linewidth=2.3,
            label=f"Aligned ideal HP2 ref (fc={FC_THEORY:.2f}Hz, Q={Q_THEORY:.3f})"
        )
        ax2.semilogx(
            f_smooth,
            theory_phase_aligned,
            '--',
            linewidth=2.3,
            label="Aligned ideal HP2 ref"
        )

    ax1.set_title("Combined Sensitivity Response (Nominal Bin Method)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Sensitivity")
    ax1.grid(True, which="both", alpha=0.2)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(combined_root, "combined_sensitivity_response.png"), dpi=180, bbox_inches="tight")
    plt.close(fig1)

    ax2.set_title("Combined Phase Response (Nominal Bin Method)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(combined_root, "combined_phase_response.png"), dpi=180, bbox_inches="tight")
    plt.close(fig2)

    print(f"[SAVED] Combined plots -> {combined_root}")

# =========================================================
# MAIN
# =========================================================

def main():
    # Full overwrite of old results
    clean_dir(RESULTS_ROOT)

    all_master_dfs = {}

    for direction, sensor_keys in DIRECTIONS.items():
        for sensor_key in sensor_keys:
            master_df = process_single_sensor(direction, sensor_key)
            if master_df is not None and not master_df.empty:
                all_master_dfs[f"{direction}_{sensor_key}"] = master_df

    make_combined_plots(all_master_dfs)
    print("[DONE] Processing complete.")

if __name__ == "__main__":
    main()