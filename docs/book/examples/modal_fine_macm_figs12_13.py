"""Reproduce Figs. 12 and 13 of Matthews et al. (2026): MACM for good vs. poor
laboratory input locations on the Case 1 two-beam system.

Usage: python modal_fine_macm_figs12_13.py [output_dir]

Setup matches Zwink's MATLAB source (run_model_case_1/2.m, make_plots.m):
200 field / 200 laboratory modes, Zwink's force-DOF locations, DUT response
DOFs at nodes 14/26/39/53, field excitation at fixture node 138, a Hann-windowed
sine burst at field mode 4, a 0.1% plotting floor, 'jet' colormap, and the
paper's plot windows (Fig. 12: 25x20 modes; Fig. 13: 35x200 modes).
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa
import sdynpy as sdpy
from sdynpy.demo.two_beam_modal_fine import create_case_1_models, in_plane_shapes, sine_burst_force
from sdynpy.modal.sdynpy_macm import (
    modal_transformation_matrix,
    modal_amplitude_contribution_map,
    modal_response_from_force,
    effective_mode_count,
)

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT_DIR, exist_ok=True)

models = create_case_1_models()

NUM_MODES = 830
field_shapes = in_plane_shapes(models['field'], NUM_MODES)
lab_shapes = in_plane_shapes(models['lab'], NUM_MODES)
field_shapes.damping = 0.02
lab_shapes.damping = 0.02

KEEP_MODES = 200
field_shapes = field_shapes[:KEEP_MODES]
lab_shapes = lab_shapes[:KEEP_MODES]

response_dofs = sdpy.coordinate_array(node=[1000 + n for n in (14, 26, 39, 53)], direction=3)  # sys1_idofs=[27,51,77,105] -> DUT nodes 14,26,39,53
field_excitation_dof = sdpy.coordinate_array(node=[2000 + 138], direction=3)  # sys1_jdofs=[457] -> fixture node 138
poor_force_dofs = sdpy.coordinate_array(node=[2000 + n for n in (1, 3, 5, 7)], direction=3)
good_force_dofs = sdpy.coordinate_array(node=[2000 + n for n in (26, 50, 114, 138)], direction=3)

field_mode_index = 3
field_mode_freq = field_shapes.frequency[field_mode_index]
print(f'field mode {field_mode_index + 1} frequency: {field_mode_freq:.3f} Hz (paper: 30.1 Hz)')

all_frequencies, force_spectrum_full = sine_burst_force(field_mode_freq)

# --- Match the amplitude convention of Zwink's MATLAB FFT (fft_bz.m) --------
# This only rescales the colorbar; it does not change the shape of the MACM
# or effective_mode_count, because the MACM is linear in the force spectrum.
#
# sine_burst_force returns an *unnormalized* one-sided FFT (scipy rfft, the
# same convention as sdynpy's TimeHistoryArray.fft), so a sine of amplitude A
# shows up as a peak of roughly A*N/2 for N time samples. fft_bz.m instead
# returns a *single-sided amplitude spectrum*, i.e. a sine of amplitude A
# shows up as roughly A. It gets there in three steps:
#   1. Zero-pad the signal to nfft = 2N before the FFT.
#   2. Divide by nfft/2, which equals N (the signal length) because nfft = 2N.
#      Zero-padding does not change the FFT magnitude at a given frequency
#      (it appends zeros, not energy), so this is the same 1/N one would use
#      without padding.
#   3. Keep only the nonnegative frequencies and double every bin except DC
#      and Nyquist, since the dropped negative-frequency half carried the
#      other half of the amplitude of each real sinusoid.
# Net effect: 2/N on every bin except DC and Nyquist, which get 1/N. That is
# what is applied below. Checked against the paper: this makes the colorbar
# maxima ~199 (Fig. 12) and ~4508 (Fig. 13) versus the paper's ~180-200 and
# 4500.
#
# What is NOT replicated: the zero-padding itself. It makes fft_bz.m's
# frequency grid twice as fine (nfft = 2N gives 0.1 Hz spacing here, vs. 0.2 Hz
# for our unpadded 5 s record). Padding only interpolates the same spectrum
# at more points, so it does not change the scaling above, but it can land
# closer to the true spectral peak: for this Hann-windowed 30.1 Hz burst the
# largest sampled value is ~3.7% higher with padding than without (measured),
# so expect our colorbar maxima to read ~4% below Zwink's. Zwink's plotting
# code also subsamples to 500 frequency lines (plotmat_3d.m), so the two
# frequency grids are not meant to coincide.
n_samples = 2 * (len(all_frequencies) - 1)
force_spectrum_full = force_spectrum_full * (2.0 / n_samples)
force_spectrum_full[0] *= 0.5   # DC bin: not doubled (1/N)
force_spectrum_full[-1] *= 0.5  # Nyquist bin: not doubled (1/N)

band_max = 3 * field_mode_freq
mask = all_frequencies <= band_max
frequencies = all_frequencies[mask]
force_spectrum = force_spectrum_full[mask][np.newaxis, :]
freq_index = np.argmin(np.abs(frequencies - field_mode_freq))
print(f'frequency band: [0, {band_max:.1f}] Hz, {len(frequencies)} points')

field_modal_response = modal_response_from_force(
    field_shapes, field_excitation_dof, force_spectrum, frequencies)

macm = {}
for label, force_dofs in [("good", good_force_dofs), ("poor", poor_force_dofs)]:
    T = modal_transformation_matrix(field_shapes, lab_shapes, response_dofs, force_dofs, frequencies)
    macm[label] = modal_amplitude_contribution_map(T, field_modal_response)
    count = effective_mode_count(macm[label][:, field_mode_index, freq_index], threshold=0.95)
    print(label, "effective mode count:", count, "(out of", lab_shapes.shape[0], "lab modes)")

HIDE_BELOW = 0.001  # Brandon's plotmat_3d.m convention

# Marker size: area proportional to |MACM| (same law as plotmat_3d.m, which
# scales marker area by 20000/max). MATLAB's '.' marker is not drawn at the
# nominal area, so the absolute scale was calibrated against the paper's Fig. 12
# instead: across 28 isolated dots (excluding the red-boxed field mode 4), the
# measured dot diameter vs. our |MACM| at the same grid point has correlation
# 0.997 and log-log slope 0.56 (0.5 = area proportional to magnitude), and the
# paper's largest dot works out to ~5.1% of the axes width. A max marker area
# of ~335 pt^2 gives that on this 7.2 in wide figure. This is a visual match to
# a low-resolution PDF image, good to perhaps +/-10-15% in diameter.
#
# KNOWN, UNEXPLAINED: this calibration is from Fig. 12 only. In Fig. 13 the
# paper's dots through the middle of the stripes (lab modes ~20-100, where our
# |MACM| is only ~0.1-3% of the window max) are drawn much wider than this
# size law would give from our magnitudes (a rough stripe-width check gave
# ratios of ~2x to >10x, contaminated by neighbouring stripes touching). That
# means either the paper's magnitudes there are larger than ours, or its
# marker size does not follow area ~ |MACM| at small values. Nothing in
# plotmat_3d.m/make_plots.m explains it, so no correction is applied; Fig. 13
# will therefore look sparser than the paper's in those rows.
MARKER_AREA_MAX = 335

windows = {
    12: dict(label="good", n_field=25, n_lab=20),
    13: dict(label="poor", n_field=35, n_lab=200),
}

LABEL_FS = 15
TICK_FS = 13
TITLE_FS = 15

for fig_num, spec in windows.items():
    label = spec["label"]
    n_field, n_lab = spec["n_field"], spec["n_lab"]
    lab_window = slice(0, min(n_lab, lab_shapes.shape[0]))
    field_window = slice(0, min(n_field, field_shapes.shape[0]))

    m = macm[label][lab_window, field_window, :]
    n_lab_actual, n_field_actual, n_freq = m.shape
    lab_idx, field_idx, freq_idx = np.meshgrid(
        np.arange(lab_window.start, lab_window.start + n_lab_actual) + 1,
        np.arange(field_window.start, field_window.start + n_field_actual) + 1,
        np.arange(n_freq), indexing='ij')
    mags = np.abs(m)
    max_mag = mags.max()

    lab_r, field_r, freq_r, mag_r = lab_idx.ravel(), field_idx.ravel(), freq_idx.ravel(), mags.ravel()
    keep = mag_r >= HIDE_BELOW * max_mag
    lab_r, field_r, freq_r, mag_r = lab_r[keep], field_r[keep], freq_r[keep], mag_r[keep]
    sizes = MARKER_AREA_MAX * mag_r / max_mag

    # Standalone top-view (b) figure -- this is what goes into the comparison.
    fig_b, ax2 = plt.subplots(figsize=(7.2, 6.4))
    sc2 = ax2.scatter(field_r, lab_r, s=sizes, c=mag_r, cmap='jet')
    ax2.set_xlabel('Field Environment System Mode', fontsize=LABEL_FS)
    ax2.set_ylabel('Laboratory Test System Mode', fontsize=LABEL_FS)
    ax2.set_xlim(0.5, n_field_actual + 0.5)
    ax2.set_ylim(0.5, n_lab_actual + 0.5)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(axis='both', labelsize=TICK_FS)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.set_title(f"Fig. {fig_num} reproduction -- {label} laboratory input locations (top view)",
                  fontsize=TITLE_FS, pad=18)
    cb = fig_b.colorbar(sc2, ax=ax2, label='|MACM|')
    cb.ax.tick_params(labelsize=TICK_FS - 1)
    cb.set_label('|MACM|', fontsize=LABEL_FS - 1)
    cb.ax.yaxis.get_offset_text().set_fontsize(TICK_FS - 3)
    cb.ax.yaxis.get_offset_text().set_position((1.0, 1.06))
    fig_b.tight_layout()
    out_b = os.path.join(OUT_DIR, f"fig{fig_num}_{label}_topview.png")
    fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print("saved standalone top view ->", out_b)

    # Full isometric+top composite (kept for completeness / the "(a)" view).
    fig = plt.figure(figsize=(12, 5.5))
    fig.suptitle(f"My reproduction of Fig. {fig_num} -- MACM, {label} laboratory input locations "
                 f"(field mode {field_mode_index + 1}, {field_mode_freq:.1f} Hz)")

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.view_init(elev=18, azim=-115)
    ax1.scatter(field_r, lab_r, frequencies[freq_r], s=sizes, c=mag_r, cmap='jet')
    ax1.set_xlabel('Field Environment System Mode')
    ax1.set_ylabel('Laboratory Test System Mode')
    ax1.set_zlabel('Frequency (Hz)')
    ax1.set_xlim(0.5, n_field_actual + 0.5)
    ax1.set_ylim(n_lab_actual + 0.5, 0.5)
    ax1.set_zlim(0, 100)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title('(a) isometric view')

    ax2b = fig.add_subplot(1, 2, 2)
    sc2b = ax2b.scatter(field_r, lab_r, s=sizes, c=mag_r, cmap='jet')
    ax2b.set_xlabel('Field Environment System Mode')
    ax2b.set_ylabel('Laboratory Test System Mode')
    ax2b.set_xlim(0.5, n_field_actual + 0.5)
    ax2b.set_ylim(0.5, n_lab_actual + 0.5)
    ax2b.invert_yaxis()
    ax2b.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2b.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2b.grid(True, linestyle='--', alpha=0.5)
    ax2b.set_axisbelow(True)
    ax2b.set_title('(b) top view')
    fig.colorbar(sc2b, ax=ax2b, label='|MACM|')

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"fig{fig_num}_{label}_isometric_top.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved composite ->", out)
