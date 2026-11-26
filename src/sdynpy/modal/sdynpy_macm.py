# -*- coding: utf-8 -*-
"""
Modal Based Fixture Neutralization (Modal FINE) and the Modal Amplitude
Contribution Map (MACM), from:

    Matthews, J., Zwink, B., Avitabile, P., "Modal Based Fixture
    Neutralization (FINE) Method - Field to Laboratory Considerations,"
    Experimental Techniques (2026).

    Zwink, B., "Dynamic Response Matching From Field to Laboratory
    Replication Methodology," PhD dissertation, University of Massachusetts
    Lowell (2020) -- the origin of the MACM technique and of the two-beam
    test cases in `sdynpy.demo.two_beam_modal_fine`; several model
    parameters used there (connection stiffnesses, spring locations, bolt
    hardware mass) are only documented in the dissertation's MATLAB
    appendix, not in the journal paper's own tables.

Modal FINE computes a laboratory ("DL2") excitation force that causes a
Device Under Test (DUT) response to match a field ("DL1") response, using
only mode shapes/frequencies/damping of the two assemblies rather than
measured connection-DOF FRFs directly (contrast with the related Impedance
FINE technique). The MACM decomposes that calculation mode-by-mode to show
which laboratory modes are needed to reconstruct each field mode's
contribution to the target response -- a diagnostic for judging whether a
chosen set of laboratory force locations is well- or poorly-conditioned
*before* a test is run.

Note on the name "MACM"
------------------------
Zwink's dissertation consistently calls this the Modal Amplitude
Contribution *Map*. The Matthews et al. (2026) paper introduces it once as
the Modal Amplitude Contribution *Matrix*, then uses "Map" for the rest of
the document (including a section heading), matching the dissertation.
This module follows the dissertation's original, consistently-used name;
the abbreviation "MACM" is unaffected either way.

Note on the paper's Eqs. 8, 10, and 11
---------------------------------------
As printed, the synthesized-FRF term inside these equations uses the
response-DOF mode shape matrix ``U_i`` in every slot. That is inconsistent
with Eq. 5 (which Eq. 8 is explicitly derived from, and which uses
``H_ij``, not ``H_ii``) and with Eq. 7 (which supplies its own force-DOF
mode shape matrix ``U_j``), and with the paper's own description of the
transformation matrix as depending on "laboratory forcing locations" --
forcing locations only enter through ``U_j``. This was confirmed against
the paper's page images independently twice. The functions in this module
implement the corrected general form (substituting ``U_j`` where the
paper's printed equations have ``U_i`` in the force-side slots), which is
provably an *exact* restatement of Eq. 3 for any DOF selection or rank, and
which reduces exactly to the paper's printed equations when the force DOFs
and response DOFs coincide.

This correction is independently confirmed against Zwink's dissertation
(cited above), which derives the same relationship under the name ICE
(Intrinsic Connection Excitation), Section 2.5.5.6. The dissertation's
Eqs. 2.89, 2.91, and 2.92 -- corresponding to the paper's Eqs. 8, 10, and 11
respectively -- consistently use ``U_j`` in the force-side slots throughout,
matching the correction implemented here rather than the paper's printed
all-``U_i`` form. The typo therefore appears to have been introduced when
the paper condensed the dissertation's derivation into journal form, not an
error in the original derivation.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _pinv(matrix, rcond):
    if rcond is None:
        return np.linalg.pinv(matrix)
    return np.linalg.pinv(matrix, rcond=rcond)


def _modal_frf_diagonal(shapes, frequencies, displacement_derivative=2):
    """Per-mode scalar modal FRFs (Eq. 9), i.e. the diagonal of `[Hbar]`.

    Matches the same modal-FRF convention used by `ShapeArray.compute_frf`
    (real, proportionally damped modes only) so that forces/transformations
    computed here are consistent with FRFs synthesized elsewhere in SDynPy
    at the same `displacement_derivative`.
    """
    if shapes.is_complex():
        raise NotImplementedError(
            'Complex mode shapes are not supported by sdynpy_macm; the '
            'Modal FINE / MACM formulation in Matthews et al. (2026) is '
            'derived for real, proportionally damped modes.')
    flat_shapes = shapes.flatten()
    damping_ratios = flat_shapes.damping
    angular_natural_frequencies = flat_shapes.frequency * 2 * np.pi
    angular_frequencies = np.asarray(frequencies, dtype=float) * 2 * np.pi
    modal_mass = flat_shapes.modal_mass
    denominator = 1 / (modal_mass * (
        -angular_frequencies[:, np.newaxis]**2
        + angular_natural_frequencies**2
        + 2j * damping_ratios * angular_frequencies[:, np.newaxis] * angular_natural_frequencies))
    if displacement_derivative > 0:
        denominator = denominator * (1j * angular_frequencies[:, np.newaxis])**displacement_derivative
    return denominator  # (n_freq, n_lab_modes)


def _lab_synthesis(lab_shapes, response_dofs, force_dofs, frequencies, rcond,
                   displacement_derivative):
    """Shared setup for `modal_transformation_matrix` and `compute_modal_fine`:
    the lab mode shapes at the response/force DOFs, the modal FRF diagonal,
    and the per-frequency pseudo-inverse of the synthesized H_ij^(DL2)
    (Eq. 7), computed once so it is never redone (and can never disagree
    between the transformation matrix and the force calculation).
    """
    u_i_lab = lab_shapes[response_dofs].T  # (n_i, n_lab_modes)
    u_j_lab = lab_shapes[force_dofs].T     # (n_j, n_lab_modes)
    hbar_lab = _modal_frf_diagonal(lab_shapes, frequencies, displacement_derivative)
    n_freq = hbar_lab.shape[0]
    n_i, n_j = u_i_lab.shape[0], u_j_lab.shape[0]
    synthesized_frf_pinv = np.zeros((n_freq, n_j, n_i), dtype=complex)
    for i in range(n_freq):
        hbar_diag = np.diag(hbar_lab[i])
        synthesized_frf = u_i_lab @ hbar_diag @ u_j_lab.T  # Eq. 7: H_ij^(DL2)
        synthesized_frf_pinv[i] = _pinv(synthesized_frf, rcond)
    return u_i_lab, u_j_lab, hbar_lab, synthesized_frf_pinv


def modal_transformation_matrix(field_shapes, lab_shapes, response_dofs, force_dofs,
                                frequencies, rcond=None, displacement_derivative=2):
    """Computes the modal transformation matrix `[T^(DL1->DL2)]` (Eq. 11,
    corrected -- see module docstring) relating field modal responses to
    laboratory modal responses for a given choice of DUT response DOFs and
    laboratory force DOFs.

    Parameters
    ----------
    field_shapes : ShapeArray
        Mode shapes/frequencies/damping of the field ("DL1") assembly,
        evaluated at least at `response_dofs`.
    lab_shapes : ShapeArray
        Mode shapes/frequencies/damping of the laboratory ("DL2") assembly,
        evaluated at both `response_dofs` and `force_dofs`.
    response_dofs : CoordinateArray
        The DUT response DOFs ("i") common to both assemblies.
    force_dofs : CoordinateArray
        The candidate laboratory force DOFs ("j") on the DL2 assembly.
    frequencies : ndarray
        Frequency lines (Hz) at which to evaluate the transformation.
    rcond : float, optional
        Condition number threshold passed to `numpy.linalg.pinv` when
        inverting the synthesized FRF at each frequency line.
    displacement_derivative : int, optional
        0 for displacement, 1 for velocity, 2 (default) for acceleration --
        must match the convention used for the field response supplied to
        `compute_modal_fine`/`project_modal_response`.

    Returns
    -------
    transformation : ndarray
        Complex array of shape (n_lab_modes, n_field_modes, n_freq).
    """
    u_i_field = field_shapes[response_dofs].T  # (n_i, n_field_modes)
    u_i_lab, u_j_lab, hbar_lab, synthesized_frf_pinv = _lab_synthesis(
        lab_shapes, response_dofs, force_dofs, frequencies, rcond, displacement_derivative)
    n_freq = hbar_lab.shape[0]
    n_lab_modes = u_i_lab.shape[1]
    n_field_modes = u_i_field.shape[1]
    transformation = np.zeros((n_lab_modes, n_field_modes, n_freq), dtype=complex)
    for i in range(n_freq):
        transformation[:, :, i] = (np.diag(hbar_lab[i]) @ u_j_lab.T
                                   @ synthesized_frf_pinv[i] @ u_i_field)
    return transformation


def modal_amplitude_contribution_map(transformation, field_modal_response):
    """Computes the Modal Amplitude Contribution Map (MACM, Eq. 12):
    the laboratory modal response broken down by field mode, before being
    summed together over the field-mode axis.

    Parameters
    ----------
    transformation : ndarray
        `[T^(DL1->DL2)]` as returned by `modal_transformation_matrix`,
        shape (n_lab_modes, n_field_modes, n_freq).
    field_modal_response : ndarray
        Field modal coordinates `{P^(DL1)}`, shape (n_field_modes, n_freq),
        as returned by `project_modal_response`.

    Returns
    -------
    macm : ndarray
        Complex array of shape (n_lab_modes, n_field_modes, n_freq).
        Summing over the field-mode axis (`macm.sum(axis=1)`) recovers the
        laboratory modal response `{P^(DL2)}`.
    """
    field_modal_response = np.asarray(field_modal_response)
    return transformation * field_modal_response[np.newaxis, :, :]


def project_modal_response(shapes, dofs, response, rcond=None):
    """Projects a physical response onto a set of mode shapes to obtain
    modal coordinates -- the inverse of Eq. 6, `{p} = [U]^g {x}`.

    This is exact only if `response` lies in the column space of the mode
    shape matrix at `dofs` (guaranteed if `response` was itself synthesized
    from `shapes`); otherwise it is the least-squares projection.

    Parameters
    ----------
    shapes : ShapeArray
    dofs : CoordinateArray
    response : ndarray
        Physical response, shape (n_dofs, n_freq).
    rcond : float, optional

    Returns
    -------
    modal_response : ndarray
        Shape (n_modes, n_freq).
    """
    mode_shape_matrix = shapes[dofs].T  # (n_dofs, n_modes)
    return _pinv(mode_shape_matrix, rcond) @ np.asarray(response)


def modal_response_from_force(shapes, force_dofs, force, frequencies,
                              displacement_derivative=2):
    """Computes modal coordinates `{P}` directly from a physical force applied
    at `force_dofs`, via modal expansion: `P(mode, freq) = Hbar(mode, freq) *
    [U(force_dofs, mode)]^T {F(force_dofs, freq)}`.

    This is the forward analog of `project_modal_response`, which instead
    recovers modal coordinates from a *response* via a pseudo-inverse and is
    therefore lossy/rank-deficient whenever there are fewer response DOFs
    than modes. This direction needs no inverse at all: it is exact for any
    number of modes, since the force enters linearly through the (known)
    force-DOF mode shapes. Matches Zwink's dissertation (`calculate_results.m`:
    `system1.Pdd = diag(H_modal_dd) * U(jdofs,modes).' * F`).

    Parameters
    ----------
    shapes : ShapeArray
    force_dofs : CoordinateArray
        Physical DOFs at which `force` is applied.
    force : ndarray
        Complex force spectrum, shape (n_force_dofs, n_freq).
    frequencies : ndarray
        Frequency lines (Hz) matching `force`'s second axis.
    displacement_derivative : int, optional
        0 (displacement), 1 (velocity), or 2 (acceleration, default) -- must
        match the convention used downstream (e.g. in
        `modal_transformation_matrix`/`compute_modal_fine`).

    Returns
    -------
    modal_response : ndarray
        Shape (n_modes, n_freq).
    """
    u_j = shapes[force_dofs].T  # (n_force_dofs, n_modes)
    hbar = _modal_frf_diagonal(shapes, frequencies, displacement_derivative)  # (n_freq, n_modes)
    modal_force = u_j.T @ np.asarray(force)  # (n_modes, n_freq)
    return hbar.T * modal_force


@dataclass
class ModalFineResult:
    """Results of a Modal FINE / MACM computation, from `compute_modal_fine`."""
    frequencies: np.ndarray
    transformation: np.ndarray
    macm: np.ndarray
    field_modal_response: np.ndarray
    lab_modal_response: np.ndarray
    lab_forces: np.ndarray


def compute_modal_fine(field_shapes, lab_shapes, response_dofs, force_dofs,
                       field_response, frequencies, rcond=None,
                       displacement_derivative=2):
    """Computes the Modal FINE laboratory forces and the Modal Amplitude
    Contribution Map that reproduce a field DUT response using a
    laboratory fixture attached at the same DUT response DOFs.

    Parameters
    ----------
    field_shapes : ShapeArray
        Mode shapes/frequencies/damping of the field assembly, evaluated at
        least at `response_dofs`.
    lab_shapes : ShapeArray
        Mode shapes/frequencies/damping of the laboratory assembly,
        evaluated at both `response_dofs` and `force_dofs`.
    response_dofs : CoordinateArray
        DUT response ("i") DOFs to be matched between field and lab.
    force_dofs : CoordinateArray
        Candidate laboratory force ("j") DOFs.
    field_response : ndarray
        Target field response `{X_i^(DL1)}`, shape (n_i, n_freq), at the
        same `displacement_derivative` as `displacement_derivative`.
    frequencies : ndarray
        Frequency lines (Hz).
    rcond : float, optional
        Condition number threshold for the pseudo-inverses.
    displacement_derivative : int, optional
        0 (displacement), 1 (velocity), or 2 (acceleration, default).

    Returns
    -------
    ModalFineResult
    """
    frequencies = np.asarray(frequencies, dtype=float)
    field_modal_response = project_modal_response(
        field_shapes, response_dofs, field_response, rcond=rcond)

    u_i_field = field_shapes[response_dofs].T
    u_i_lab, u_j_lab, hbar_lab, synthesized_frf_pinv = _lab_synthesis(
        lab_shapes, response_dofs, force_dofs, frequencies, rcond, displacement_derivative)

    n_freq = frequencies.size
    n_lab_modes = u_i_lab.shape[1]
    n_field_modes = u_i_field.shape[1]
    n_j = u_j_lab.shape[0]
    transformation = np.zeros((n_lab_modes, n_field_modes, n_freq), dtype=complex)
    lab_forces = np.zeros((n_j, n_freq), dtype=complex)
    for i in range(n_freq):
        transformation[:, :, i] = (np.diag(hbar_lab[i]) @ u_j_lab.T
                                   @ synthesized_frf_pinv[i] @ u_i_field)
        lab_forces[:, i] = synthesized_frf_pinv[i] @ (u_i_field @ field_modal_response[:, i])

    macm = modal_amplitude_contribution_map(transformation, field_modal_response)
    lab_modal_response = macm.sum(axis=1)

    return ModalFineResult(frequencies=frequencies, transformation=transformation,
                           macm=macm, field_modal_response=field_modal_response,
                           lab_modal_response=lab_modal_response, lab_forces=lab_forces)


def effective_mode_count(contributions, threshold=0.95):
    """Number of largest-magnitude entries needed for their cumulative
    magnitude to reach `threshold` fraction of the total -- a simple
    diagnostic for how "concentrated" a single MACM slice is. A low count
    means only a few laboratory modes are needed to reconstruct the
    field mode's contribution at that slice (well-conditioned force/response
    DOF placement); a high count (approaching the total number of available
    laboratory modes) means the contribution is smeared across nearly all
    of them (poorly-conditioned placement), matching the qualitative
    "10-15 modes vs. ~190 modes" finding in the paper's Case 1.1 vs. 1.2.

    Parameters
    ----------
    contributions : ndarray
        A single field-mode/frequency slice of a MACM array, e.g.
        `result.macm[:, field_mode_index, freq_index]`.
    threshold : float, optional
        Cumulative fraction of the total magnitude to reach. Default 0.95.

    Returns
    -------
    int
    """
    magnitudes = np.abs(np.asarray(contributions))
    total = magnitudes.sum()
    if total == 0:
        return 0
    order = np.argsort(magnitudes)[::-1]
    cumulative = np.cumsum(magnitudes[order])
    return int(np.searchsorted(cumulative, threshold * total) + 1)


def plot_macm(macm, field_frequencies, lab_frequencies, frequency_index=None, ax=None,
              hide_below=0.001, cmap='jet'):
    """Scatter plot of a MACM slice (or the full MACM summed over frequency)
    in the style of Figs. 2, 12, and 13 of Matthews et al. (2026): one point
    per (laboratory mode, field mode) pair, with marker size proportional to
    the contribution magnitude.

    Parameters
    ----------
    macm : ndarray
        MACM array, shape (n_lab_modes, n_field_modes, n_freq), as returned
        by `modal_amplitude_contribution_map`/`compute_modal_fine`.
    field_frequencies : ndarray
        Natural frequencies (Hz) of the field modes, length n_field_modes.
    lab_frequencies : ndarray
        Natural frequencies (Hz) of the laboratory modes, length n_lab_modes.
    frequency_index : int, optional
        If given, plot only that frequency line. Default is to sum the
        magnitude of the MACM over all frequency lines.
    ax : matplotlib.axes.Axes, optional
    hide_below : float, optional
        Omit points whose magnitude is below this fraction (0-1) of the
        largest magnitude in the plotted slice, decluttering the plot.
        Default is 0.001 (0.1%), matching Brandon Zwink's own MATLAB
        plotting code (`plotmat_3d.m`), which drops points below 0.1% of
        the global maximum before rendering. Pass `None` to plot every
        point. Marker size and
        color are otherwise unaffected: both this module and Zwink's code
        scale marker size by the (unfiltered) global maximum magnitude and
        pass raw, unscaled magnitude as the color value.
    cmap : str or Colormap, optional
        Matplotlib colormap. Default is ``'jet'``, matching the paper's
        figures and Zwink's MATLAB code.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if frequency_index is None:
        magnitude = np.sum(np.abs(macm), axis=-1)
    else:
        magnitude = np.abs(macm[:, :, frequency_index])
    if ax is None:
        _, ax = plt.subplots()
    # 1-indexed mode numbers (mode 1, 2, 3, ...) rather than raw 0-indexed
    # array positions, matching the convention used in the paper's own
    # figures and avoiding an easy-to-misread "mode 3 vs. mode 4" mixup.
    lab_index, field_index = np.meshgrid(
        np.arange(1, len(lab_frequencies) + 1),
        np.arange(1, len(field_frequencies) + 1), indexing='ij')
    max_magnitude = magnitude.max() if magnitude.max() > 0 else 1.0
    lab_points, field_points, sizes, colors = (
        lab_index.ravel(), field_index.ravel(),
        200 * magnitude.ravel() / max_magnitude, magnitude.ravel())
    if hide_below is not None:
        keep = colors >= hide_below * max_magnitude
        lab_points, field_points, sizes, colors = (
            lab_points[keep], field_points[keep], sizes[keep], colors[keep])
    scatter = ax.scatter(field_points, lab_points, s=sizes, c=colors, cmap=cmap)
    ax.set_xlabel('Field Environment System Mode')
    ax.set_ylabel('Laboratory Test System Mode')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # The paper's own Figs. 12/13 plot the laboratory mode axis increasing
    # *downward* (mode 1 at the top) rather than the matplotlib default of
    # increasing upward -- matched here since reproducing those figures is
    # the explicit point of this function.
    ax.invert_yaxis()
    plt.colorbar(scatter, ax=ax)
    return ax
