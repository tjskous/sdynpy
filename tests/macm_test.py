"""
Unit tests for the Modal FINE / Modal Amplitude Contribution Map (MACM)
math, from:

    Matthews, J., Zwink, B., Avitabile, P., "Modal Based Fixture
    Neutralization (FINE) Method - Field to Laboratory Considerations,"
    Experimental Techniques (2026).

These tests use a small, fully synthetic 3-mode "field" system and 4-mode
"lab" system (no FEM/beam models involved) so the underlying tensor math
(Eqs. 7, 10', 11', 12) can be checked independently of anything beam- or
FEM-related.

Note on the paper's Eqs. 8/10/11: as printed, the synthesized-FRF term
inside these equations uses the response-DOF mode shape matrix `U_i` in
every slot. That is inconsistent with Eq. 5 (which it is derived from) and
Eq. 7 (which supplies its own `U_j`), and with the paper's own description
of `T` as depending on "laboratory forcing locations" -- forcing locations
only enter through `U_j`. This was independently confirmed against the
paper's page images. `sdynpy_macm` therefore implements the corrected
general form (using the force-DOF mode shapes `U_j` where the paper's
apparent typo has `U_i`), which reduces exactly to the printed equations
when the force DOFs and response DOFs coincide -- see the `test_*_collocated`
case below.
"""

import numpy as np
import pytest

import sdynpy as sdpy
from sdynpy.modal.sdynpy_macm import (
    modal_transformation_matrix,
    modal_amplitude_contribution_map,
    compute_modal_fine,
    effective_mode_count,
    plot_macm,
    modal_response_from_force,
)
from sdynpy.demo.two_beam_modal_fine import (
    create_case_1_models,
    create_case_3_models,
    in_plane_shapes,
    sine_burst_force,
)
from sdynpy.signal_processing.sdynpy_correlation import frac as _frac

# Field system (DL1): 3 modes
FIELD_FREQ = np.array([12.0, 47.0, 91.0])
FIELD_DAMPING = np.array([0.02, 0.015, 0.030])
FIELD_MODAL_MASS = np.array([1.0, 1.0, 1.0])
# Field mode shapes at the 4 control ("i") DOFs, shape (n_i, n_field_modes)
U_I_DL1 = np.array([
    [1.00, 1.00, 1.00],
    [0.62, -0.41, 1.35],
    [-0.35, 0.88, -0.27],
    [0.11, -1.20, -0.75],
])

# Lab system (DL2): 4 modes
LAB_FREQ = np.array([15.0, 40.0, 78.0, 120.0])
LAB_DAMPING = np.array([0.025, 0.010, 0.020, 0.035])
LAB_MODAL_MASS = np.array([1.0, 1.0, 1.0, 1.0])
# Lab mode shapes at the SAME 4 control ("i") DOFs, shape (n_i, n_lab_modes)
U_I_DL2 = np.array([
    [1.00, 1.00, 1.00, 1.00],
    [0.55, -0.70, 1.05, -0.33],
    [-0.40, 0.95, -0.15, 0.82],
    [0.20, -1.05, -0.60, 1.41],
])
# Lab mode shapes at 3 laboratory force ("j") DOFs -- physically distinct
# nodes from the "i" DOFs, shape (n_j, n_lab_modes)
U_J_DL2 = np.array([
    [0.90, -0.30, 0.75, 0.20],
    [-0.45, 1.20, 0.10, -0.95],
    [0.65, 0.55, -1.10, 0.40],
])

FREQUENCIES = np.array([5.0, 13.0, 30.0, 44.0, 60.0, 85.0, 110.0])

RESPONSE_DOFS = sdpy.coordinate_array(node=[1, 2, 3, 4], direction=1)
FORCE_DOFS = sdpy.coordinate_array(node=[101, 102, 103], direction=1)


def _make_field_shapes(response_dofs=RESPONSE_DOFS):
    return sdpy.shape_array(response_dofs, U_I_DL1.T, FIELD_FREQ, FIELD_DAMPING,
                            FIELD_MODAL_MASS)


def _make_lab_shapes(response_dofs=RESPONSE_DOFS, force_dofs=FORCE_DOFS,
                     u_i=U_I_DL2, u_j=U_J_DL2):
    coordinate = np.concatenate([response_dofs.flatten(), force_dofs.flatten()])
    shape_matrix = np.concatenate([u_i, u_j], axis=0).T  # (n_modes, n_i+n_j)
    return sdpy.shape_array(coordinate, shape_matrix, LAB_FREQ, LAB_DAMPING,
                            LAB_MODAL_MASS)


def _modal_frf_diag(omega, wn_hz, zeta, modal_mass):
    """Independent reference implementation of Eq. 9 (receptance)."""
    wn = 2 * np.pi * np.asarray(wn_hz)
    omega = np.atleast_1d(omega)[:, None]
    wn = wn[None, :]
    zeta = np.asarray(zeta)[None, :]
    m = np.asarray(modal_mass)[None, :]
    return 1.0 / (m * (wn**2 - omega**2 + 2j * zeta * wn * omega))


def _reference_modal_fine(u_i_dl1, u_i_dl2, u_j_dl2, hbar_dl2, p_dl1):
    """Independent (non-implementation) reference for Eqs. 10'/11'/12,
    used as the ground truth that `sdynpy_macm` is checked against.
    """
    n_f = hbar_dl2.shape[0]
    n_m2, n_m1 = u_i_dl2.shape[1], u_i_dl1.shape[1]
    n_j = u_j_dl2.shape[0]
    transformation = np.zeros((n_m2, n_m1, n_f), dtype=complex)
    forces = np.zeros((n_j, n_f), dtype=complex)
    for w in range(n_f):
        hb = np.diag(hbar_dl2[w])
        a = u_i_dl2 @ hb @ u_j_dl2.T  # Eq. 7: H_ij^(DL2)
        a_pinv = np.linalg.pinv(a)
        transformation[:, :, w] = hb @ u_j_dl2.T @ a_pinv @ u_i_dl1
        forces[:, w] = a_pinv @ (u_i_dl1 @ p_dl1[:, w])
    macm = transformation * p_dl1[None, :, :]
    return transformation, macm, forces


@pytest.fixture
def field_modal_response():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((3, FREQUENCIES.size))
            + 1j * rng.standard_normal((3, FREQUENCIES.size)))


def test_transformation_matrix_matches_reference(field_modal_response):
    field_shapes = _make_field_shapes()
    lab_shapes = _make_lab_shapes()

    transformation = modal_transformation_matrix(
        field_shapes, lab_shapes, RESPONSE_DOFS, FORCE_DOFS, FREQUENCIES)

    omega = 2 * np.pi * FREQUENCIES
    hbar_dl2 = _modal_frf_diag(omega, LAB_FREQ, LAB_DAMPING, LAB_MODAL_MASS)
    reference_transformation, _, _ = _reference_modal_fine(
        U_I_DL1, U_I_DL2, U_J_DL2, hbar_dl2, field_modal_response)

    assert transformation.shape == (4, 3, FREQUENCIES.size)
    np.testing.assert_allclose(transformation, reference_transformation, rtol=1e-10)


def test_macm_sums_to_lab_modal_response(field_modal_response):
    field_shapes = _make_field_shapes()
    lab_shapes = _make_lab_shapes()

    transformation = modal_transformation_matrix(
        field_shapes, lab_shapes, RESPONSE_DOFS, FORCE_DOFS, FREQUENCIES)
    macm = modal_amplitude_contribution_map(transformation, field_modal_response)

    lab_modal_response = macm.sum(axis=1)

    omega = 2 * np.pi * FREQUENCIES
    hbar_dl2 = _modal_frf_diag(omega, LAB_FREQ, LAB_DAMPING, LAB_MODAL_MASS)
    _, reference_macm, reference_forces = _reference_modal_fine(
        U_I_DL1, U_I_DL2, U_J_DL2, hbar_dl2, field_modal_response)

    np.testing.assert_allclose(macm, reference_macm, rtol=1e-10)
    np.testing.assert_allclose(lab_modal_response, reference_macm.sum(axis=1), rtol=1e-10)

    # And the forces implied by this modal response must match Eq. 3 applied
    # directly to the synthesized H_ij -- the whole point of the modal route
    # is that it is an exact decomposition of Eq. 3, never an approximation.
    field_shape_matrix = field_shapes[RESPONSE_DOFS]
    x_i_dl1 = field_shape_matrix.T @ field_modal_response
    h_ij = np.einsum('ik,wk,jk->wij', U_I_DL2, hbar_dl2, U_J_DL2)
    direct_forces = np.stack([
        np.linalg.pinv(h_ij[w]) @ x_i_dl1[:, w] for w in range(FREQUENCIES.size)
    ], axis=-1)
    np.testing.assert_allclose(reference_forces, direct_forces, rtol=1e-10)


def test_printed_paper_formula_would_be_wrong_when_j_ne_i():
    """Guards against silently regressing to the paper's literal (typo'd)
    Eq. 11, which uses U_i in the force-side slots instead of U_j. That
    formula only agrees with the correct one when the force DOFs and
    response DOFs coincide (see test_collocated_matches_printed_formula).
    """
    field_shapes = _make_field_shapes()
    lab_shapes = _make_lab_shapes()

    transformation = modal_transformation_matrix(
        field_shapes, lab_shapes, RESPONSE_DOFS, FORCE_DOFS, FREQUENCIES)

    omega = 2 * np.pi * FREQUENCIES
    hbar_dl2 = _modal_frf_diag(omega, LAB_FREQ, LAB_DAMPING, LAB_MODAL_MASS)
    printed_transformation = np.zeros_like(transformation)
    for w in range(FREQUENCIES.size):
        hb = np.diag(hbar_dl2[w])
        a_ii = U_I_DL2 @ hb @ U_I_DL2.T  # paper's literal H_ii, not H_ij
        printed_transformation[:, :, w] = hb @ U_I_DL2.T @ np.linalg.pinv(a_ii) @ U_I_DL1

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(transformation, printed_transformation, rtol=1e-6)


def test_collocated_matches_printed_formula():
    """When the force DOFs equal the response DOFs (j == i), the corrected
    formula collapses exactly onto the paper's literally-printed Eq. 11.
    """
    field_shapes = _make_field_shapes()
    lab_shapes = _make_lab_shapes(force_dofs=RESPONSE_DOFS, u_j=U_I_DL2)

    transformation = modal_transformation_matrix(
        field_shapes, lab_shapes, RESPONSE_DOFS, RESPONSE_DOFS, FREQUENCIES)

    omega = 2 * np.pi * FREQUENCIES
    hbar_dl2 = _modal_frf_diag(omega, LAB_FREQ, LAB_DAMPING, LAB_MODAL_MASS)
    printed_transformation = np.zeros_like(transformation)
    for w in range(FREQUENCIES.size):
        hb = np.diag(hbar_dl2[w])
        a_ii = U_I_DL2 @ hb @ U_I_DL2.T
        printed_transformation[:, :, w] = hb @ U_I_DL2.T @ np.linalg.pinv(a_ii) @ U_I_DL1

    np.testing.assert_allclose(transformation, printed_transformation, rtol=1e-10)


def test_modal_response_from_force_matches_direct_frf():
    """The modal-domain forward calculation (force -> modal coordinates via
    modal expansion) should exactly match applying the physical FRF directly
    to the same force spectrum -- a cross-check against already-trusted
    `ShapeArray.compute_frf` machinery, with no pseudo-inverse involved.
    """
    lab_shapes = _make_lab_shapes()
    rng = np.random.default_rng(1)
    force = (rng.standard_normal((len(FORCE_DOFS), FREQUENCIES.size))
             + 1j * rng.standard_normal((len(FORCE_DOFS), FREQUENCIES.size)))

    modal_response = modal_response_from_force(lab_shapes, FORCE_DOFS, force, FREQUENCIES)

    u_i = lab_shapes[RESPONSE_DOFS].T  # (n_i, n_modes)
    response_from_modal = u_i @ modal_response

    frf = lab_shapes.compute_frf(FREQUENCIES, responses=RESPONSE_DOFS,
                                 references=FORCE_DOFS).ordinate  # (n_i, n_j, n_freq)
    response_from_frf = np.einsum('ijw,jw->iw', frf, force)

    np.testing.assert_allclose(response_from_modal, response_from_frf, rtol=1e-10)


def test_sine_burst_force_spreads_energy_near_target_frequency():
    """A windowed sine burst (not an idealized continuous tone) should have
    spectral content near, but not exactly localized to, the target
    frequency -- this is what puts real MACM contribution into field modes
    *near* an excited mode, matching the paper's own figures, rather than an
    idealized single-mode assumption that would be zero everywhere else.
    """
    frequencies, spectrum = sine_burst_force(30.10, duration=5.0, sample_rate=10000.0)
    magnitude = np.abs(spectrum)
    peak_index = np.argmax(magnitude)
    assert abs(frequencies[peak_index] - 30.10) < 0.5

    neighbor_index = np.argmin(np.abs(frequencies - (frequencies[peak_index] + 1.0)))
    assert 0 < magnitude[neighbor_index] < magnitude[peak_index]


# ---------------------------------------------------------------------------
# Integration tests against the paper's beam FEM test cases
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def case_1_modal_model():
    """Case 1 field/lab modal models (Appendix A.6/A.7), with a small amount
    of uniform modal damping assigned (the paper's Case 1 model is
    undamped for the natural-frequency comparison in Task 1/2, but a
    non-zero damping value is needed here to evaluate FRFs near resonance).
    """
    models = create_case_1_models()
    field_shapes = in_plane_shapes(models['field'], 300)
    lab_shapes = in_plane_shapes(models['lab'], 300)
    field_shapes.damping = 0.01
    lab_shapes.damping = 0.01
    return field_shapes, lab_shapes


def test_case_1_good_and_poor_force_locations(case_1_modal_model):
    """Reproduces the qualitative finding of the paper's Case 1.1 vs 1.2:
    spreading the laboratory forces out (good) vs. clustering them close
    together (poor) both achieve a near-perfect response match, but the
    poor placement needs dramatically larger forces and a MACM spread
    across far more laboratory modes to do it.
    """
    field_shapes, lab_shapes = case_1_modal_model

    response_dofs = sdpy.coordinate_array(node=[1000 + n for n in (10, 50, 90)],
                                          direction=3)
    field_excitation_dof = sdpy.coordinate_array(node=[2000 + 136], direction=3)
    good_force_dofs = sdpy.coordinate_array(node=[2000 + n for n in (6, 70, 134)],
                                            direction=3)
    poor_force_dofs = sdpy.coordinate_array(node=[2000 + n for n in (69, 70, 71)],
                                            direction=3)

    # Field mode 4 (the paper's Case 1.1/1.2 sine excitation frequency),
    # matching Table 1's 30.1 Hz field mode 4 (our model now gives ~30.10 Hz,
    # within 0.5% after the connection-stiffness/spring-location fixes).
    field_mode_index = 3
    field_mode_freq = field_shapes.frequency[field_mode_index]
    frequencies = np.linspace(field_mode_freq - 5, field_mode_freq + 5, 11)
    frequencies = frequencies[frequencies > 0.1]

    target_response = field_shapes.compute_frf(
        frequencies, responses=response_dofs,
        references=field_excitation_dof).ordinate[:, 0, :]

    results = {}
    for label, force_dofs in [('good', good_force_dofs), ('poor', poor_force_dofs)]:
        result = compute_modal_fine(field_shapes, lab_shapes, response_dofs,
                                    force_dofs, target_response, frequencies)
        lab_frf = lab_shapes.compute_frf(frequencies, responses=response_dofs,
                                         references=force_dofs).ordinate
        reconstructed = np.einsum('ijw,jw->iw', lab_frf, result.lab_forces)
        results[label] = dict(result=result, reconstructed=reconstructed,
                              max_force=np.max(np.abs(result.lab_forces)))

    # Both placements must reproduce the target response almost exactly --
    # the paper's point is that Eq. 3 is exact regardless of DOF placement
    # quality when the force/response DOF counts are matched (both are 3x3
    # here).
    for label in ('good', 'poor'):
        frac_values = _frac(results[label]['reconstructed'], target_response)
        assert np.all(np.abs(frac_values) > 0.999), (
            f'{label} placement did not reproduce the target response '
            f'(FRAC={frac_values})')

    # The poor placement's forces must be dramatically larger -- this is the
    # practical, physical symptom of poor DOF placement the paper reports.
    assert results['poor']['max_force'] > 100 * results['good']['max_force']

    # And the MACM must show *why*: far more laboratory modes are needed to
    # reconstruct the driven field mode's contribution for the poor
    # placement than for the good one.
    freq_index = np.argmin(np.abs(frequencies - field_mode_freq))
    good_slice = results['good']['result'].macm[:, field_mode_index, freq_index]
    poor_slice = results['poor']['result'].macm[:, field_mode_index, freq_index]
    good_count = effective_mode_count(good_slice, threshold=0.95)
    poor_count = effective_mode_count(poor_slice, threshold=0.95)
    assert poor_count > 2 * good_count, (
        f'expected the poor placement to need far more effective lab modes '
        f'than the good placement (good={good_count}, poor={poor_count})')


@pytest.fixture(scope='module')
def case_3_modal_model():
    """Case 3 field/lab modal models (Appendix A.1-A.5)."""
    models = create_case_3_models()
    field_shapes = in_plane_shapes(models['field'], 250)
    lab_shapes = in_plane_shapes(models['lab'], 250)
    field_shapes.damping = 0.01
    lab_shapes.damping = 0.01
    return field_shapes, lab_shapes


def test_case_3_broadband_reconstruction(case_3_modal_model):
    """Reproduces the paper's Case 3 finding: even with a 10x stiffness
    mismatch between the field and lab fixtures, a broadband target
    response can be reconstructed almost exactly (paper reports
    TRAC=FRAC=1.0) using two laboratory forces.
    """
    field_shapes, lab_shapes = case_3_modal_model

    response_dofs = sdpy.coordinate_array(node=[1000 + n for n in (4, 8, 12)],
                                          direction=3)
    field_excitation_dof = sdpy.coordinate_array(node=[2000 + 3], direction=3)
    force_dofs = sdpy.coordinate_array(node=[2000 + n for n in (5, 24)], direction=3)

    frequencies = np.linspace(10, 300, 30)

    target_response = field_shapes.compute_frf(
        frequencies, responses=response_dofs,
        references=field_excitation_dof).ordinate[:, 0, :]

    result = compute_modal_fine(field_shapes, lab_shapes, response_dofs,
                                force_dofs, target_response, frequencies)
    lab_frf = lab_shapes.compute_frf(frequencies, responses=response_dofs,
                                     references=force_dofs).ordinate
    reconstructed = np.einsum('ijw,jw->iw', lab_frf, result.lab_forces)

    frac_values = _frac(reconstructed, target_response)
    assert np.all(np.abs(frac_values) > 0.999), (
        f'lab reconstruction did not match the field target (FRAC={frac_values})')


# ---------------------------------------------------------------------------
# Small edge cases / utility coverage
# ---------------------------------------------------------------------------

def test_modal_transformation_matrix_accepts_explicit_rcond():
    """`rcond` should thread through to the per-frequency pseudo-inverse
    rather than only being exercised via the default (`rcond=None`) path.
    """
    field_shapes = _make_field_shapes()
    lab_shapes = _make_lab_shapes()

    transformation = modal_transformation_matrix(
        field_shapes, lab_shapes, RESPONSE_DOFS, FORCE_DOFS, FREQUENCIES,
        rcond=1e-10)

    assert transformation.shape == (4, 3, FREQUENCIES.size)
    assert np.all(np.isfinite(transformation))


def test_complex_mode_shapes_not_supported():
    """Complex (non-proportionally-damped) mode shapes are out of scope for
    this formulation and must fail clearly rather than silently produce
    wrong numbers.
    """
    field_shapes = _make_field_shapes()
    complex_lab_shapes = sdpy.shape_array(
        np.concatenate([RESPONSE_DOFS.flatten(), FORCE_DOFS.flatten()]),
        np.concatenate([U_I_DL2, U_J_DL2], axis=0).T.astype(complex),
        LAB_FREQ, LAB_DAMPING, LAB_MODAL_MASS)

    with pytest.raises(NotImplementedError):
        modal_transformation_matrix(field_shapes, complex_lab_shapes,
                                    RESPONSE_DOFS, FORCE_DOFS, FREQUENCIES)


def test_effective_mode_count_zero_contribution():
    """A slice with no contribution at all (e.g. an unexcited field mode)
    should report zero effective modes rather than raising or dividing by
    zero.
    """
    assert effective_mode_count(np.zeros(10)) == 0


def test_effective_mode_count_single_dominant_mode():
    contributions = np.array([1.0, 0.001, 0.001, 0.001])
    assert effective_mode_count(contributions, threshold=0.95) == 1


def test_plot_macm_summed_over_frequency():
    rng = np.random.default_rng(0)
    macm = rng.standard_normal((6, 4, 3)) + 1j * rng.standard_normal((6, 4, 3))
    field_freqs = np.array([10.0, 20.0, 30.0, 40.0])
    lab_freqs = np.array([12.0, 22.0, 32.0, 42.0, 52.0, 62.0])

    ax = plot_macm(macm, field_freqs, lab_freqs)
    assert ax.get_xlabel() == 'Field Environment System Mode'
    assert ax.get_ylabel() == 'Laboratory Test System Mode'


def test_plot_macm_lab_axis_matches_paper_direction():
    """The paper's own Figs. 12/13 plot the laboratory mode axis increasing
    *downward* (mode 1 at the top), not matplotlib's default upward -- the
    whole point of this function is to reproduce those figures.
    """
    rng = np.random.default_rng(0)
    macm = rng.standard_normal((6, 4, 3)) + 1j * rng.standard_normal((6, 4, 3))
    field_freqs = np.array([10.0, 20.0, 30.0, 40.0])
    lab_freqs = np.array([12.0, 22.0, 32.0, 42.0, 52.0, 62.0])

    ax = plot_macm(macm, field_freqs, lab_freqs)
    assert ax.yaxis_inverted()


def test_plot_macm_single_frequency_line():
    rng = np.random.default_rng(0)
    macm = rng.standard_normal((6, 4, 3)) + 1j * rng.standard_normal((6, 4, 3))
    field_freqs = np.array([10.0, 20.0, 30.0, 40.0])
    lab_freqs = np.array([12.0, 22.0, 32.0, 42.0, 52.0, 62.0])

    ax = plot_macm(macm, field_freqs, lab_freqs, frequency_index=1)
    assert ax.get_xlabel() == 'Field Environment System Mode'


def test_plot_macm_handles_all_zero_macm():
    """The marker-size normalization must not divide by zero when the MACM
    (or the selected frequency slice) is identically zero.
    """
    macm = np.zeros((6, 4, 3), dtype=complex)
    field_freqs = np.array([10.0, 20.0, 30.0, 40.0])
    lab_freqs = np.array([12.0, 22.0, 32.0, 42.0, 52.0, 62.0])

    ax = plot_macm(macm, field_freqs, lab_freqs)
    assert ax is not None


def test_plot_macm_hide_below_declutters_small_points():
    """`hide_below` should drop points under the given fraction of the
    plotted slice's max magnitude, matching Zwink's own MATLAB plotting
    code (`plotmat_3d.m`), which discards points below 0.1% of the global
    maximum before rendering. Default (`hide_below=None`) must still plot
    every point.
    """
    field_freqs = np.array([10.0, 20.0])
    lab_freqs = np.array([12.0, 22.0])
    # (lab, field, freq) = (2, 2, 1); magnitudes 100, 1, 0.05, 0 -- only the
    # first is >= 2% of the max (100), the rest should be dropped.
    macm = np.array([[[100.0], [1.0]], [[0.05], [0.0]]], dtype=complex)

    ax_full = plot_macm(macm, field_freqs, lab_freqs)
    assert ax_full.collections[0].get_offsets().shape[0] == 4

    ax_declutter = plot_macm(macm, field_freqs, lab_freqs, hide_below=0.02)
    assert ax_declutter.collections[0].get_offsets().shape[0] == 1


def test_plot_macm_hide_below_with_all_zero_macm():
    """`hide_below` combined with an all-zero MACM should drop every point
    rather than raising or dividing by zero.
    """
    macm = np.zeros((6, 4, 3), dtype=complex)
    field_freqs = np.array([10.0, 20.0, 30.0, 40.0])
    lab_freqs = np.array([12.0, 22.0, 32.0, 42.0, 52.0, 62.0])

    ax = plot_macm(macm, field_freqs, lab_freqs, hide_below=0.001)
    assert ax.collections[0].get_offsets().shape[0] == 0
