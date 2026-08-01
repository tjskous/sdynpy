# -*- coding: utf-8 -*-
"""Two-beam analytical models used to demonstrate the Modal FINE / MACM technique.

These models reproduce the "Case 1" and "Case 3" analytical beam systems
from:

    Matthews, J., Zwink, B., Avitabile, P., "Modal Based Fixture
    Neutralization (FINE) Method - Field to Laboratory Considerations,"
    Experimental Techniques (2026). Case 1 is documented there in Appendix
    A.6/A.7 (Tables 7-8); Case 3 in Appendix A.1-A.5 (Tables 4-6).

Both test cases originate from Brandon Zwink's PhD dissertation (Dynamic
Response Matching From Field to Laboratory Replication Methodology, UMass
Lowell, 2020) -- confirmed digit-for-digit against its published natural
frequencies. Case 1 corresponds to dissertation Chapter 6 (and its own,
differently-numbered Appendix A.1-A.7, a single MATLAB script set -- not to
be confused with the paper's Appendix A, which covers both cases); Case 3
corresponds to dissertation Chapter 3. Where the paper's own tables are
silent or ambiguous, this module's parameters were taken from the
dissertation's MATLAB source directly (referenced in the comments below by
appendix section).

In both cases a Device Under Test (DUT) beam is attached to a fixture beam
(the "L" boundary condition) at two points.  The field configuration attaches
the DUT at one location on the fixture; the laboratory configuration shifts
the DUT to a different location, producing a different dynamic boundary
condition for the same DUT.

Geometry, material properties, connection stiffnesses (translational and,
for Case 1, rotational), grounding spring locations, and connection hardware
mass are all taken from the dissertation's MATLAB appendix rather than
guessed, and reproduce the paper's reported natural frequencies to within
~0.5% for Case 1 and ~0.2% for Case 3.
"""

import numpy as _np
from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq
from scipy.signal.windows import hann as _hann

from ..core.sdynpy_coordinate import (
    from_nodelist as _from_nodelist,
    coordinate_array as _coordinate_array,
)
from ..core.sdynpy_geometry import (
    Geometry as _Geometry,
    node_array as _node_array,
    traceline_array as _traceline_array,
    coordinate_system_array as _coordinate_system_array,
)
from ..core.sdynpy_system import System as _System
from ..fem.sdynpy_beam import beamkm as _beamkm


def box_beam_props(E, rho, nu, height, depth, wall_thickness, nelem=None,
                   wall_thickness_bending=None):
    """Return beam keyword dictionary for a thin-walled rectangular box (tube) beam.

    Mirrors the style of `sdynpy.fem.sdynpy_beam.rect_beam_props`, but for a
    hollow rectangular cross-section rather than a solid one.

    Parameters
    ----------
    E : float
        Young's Modulus
    rho : float
        Density
    nu : float
        Poisson's Ratio
    height : float
        Outer dimension of the cross-section in the direction of
        `bend_direction_1`.
    depth : float
        Outer dimension of the cross-section in the direction of
        `bend_direction_2`.
    wall_thickness : float
        Wall thickness used for the cross-sectional area (and hence axial
        stiffness/mass per length and the torsion constant).
    nelem : int, optional
        Number of elements.  This will repeat the values in the dict nelem
        times, as is required by beamkm.  If not specified, only a single
        value will be output for each entry in the dict.
    wall_thickness_bending : float, optional
        Wall thickness used for the bending inertias (and hence rotary
        inertia per length) instead of `wall_thickness`, if given. Zwink's
        dissertation MATLAB (the source of the two-beam test cases this
        function supports) uses two different wall thicknesses for a given
        beam's area vs. its bending inertia -- apparently a copy/paste slip
        between the DUT and fixture beam scripts, since both beams are
        documented elsewhere with one uniform thickness -- but it is what
        generated the published reference natural frequencies, so exact
        reproduction needs to be able to express it. Defaults to
        `wall_thickness` (a single uniform thickness) if not given.

    Returns
    -------
    kwargs : dict
        A dictionary that can be used in beamkm(**kwargs).
    """
    h = height
    d = depth
    t = wall_thickness
    t_bend = wall_thickness_bending if wall_thickness_bending is not None else wall_thickness
    area = h * d - (h - 2 * t) * (d - 2 * t)
    # Area moments of inertia about the two centroidal bending axes
    i1 = (d * h**3 - (d - 2 * t_bend) * (h - 2 * t_bend)**3) / 12
    i2 = (h * d**3 - (h - 2 * t_bend) * (d - 2 * t_bend)**3) / 12
    # Bredt-Batho torsion constant for a thin-walled closed section with
    # uniform wall thickness
    enclosed_area = (h - t) * (d - t)
    midline_perimeter = 2 * ((h - t) + (d - t))
    torsion_constant = 4 * enclosed_area**2 * t / midline_perimeter
    shear_modulus = E / (2 * (1 + nu))
    return_dict = {}
    return_dict['ae'] = area * E
    return_dict['jg'] = torsion_constant * shear_modulus
    return_dict['ei1'] = E * i1
    return_dict['ei2'] = E * i2
    return_dict['mass_per_length'] = rho * area
    return_dict['tmmi_per_length'] = rho * (i1 + i2)
    if nelem is not None:
        for key in return_dict:
            return_dict[key] = return_dict[key] * _np.ones((nelem,), float)
    return return_dict


def _straight_beam_system_and_geometry(length, num_elements, beam_props, color, node_offset=0):
    """Build a straight, uniformly-meshed beam System and Geometry along the x-axis."""
    num_nodes = num_elements + 1
    node_ids = _np.arange(num_nodes) + 1 + node_offset
    coords = _np.zeros((num_nodes, 3))
    coords[:, 0] = _np.linspace(0, length, num_nodes)
    connectivity = _np.array([_np.arange(num_elements), _np.arange(1, num_elements + 1)]).T
    bend_direction_1 = _np.tile([0.0, 0.0, 1.0], (num_elements, 1))

    K, M = _beamkm(coords, connectivity, bend_direction_1, **beam_props)

    geometry = _Geometry(
        _node_array(node_ids, coords, color),
        _coordinate_system_array(),
        _traceline_array(_np.arange(num_elements) + 1, color=color,
                         connectivity=connectivity + 1 + node_offset),
    )
    system = _System(_from_nodelist(node_ids, [1, 2, 3, 4, 5, 6]), M, K)
    return system, geometry


def _add_ground_springs_at_nodes(system, node_ids, stiffness, direction=3):
    """Add soft grounding springs at the given node ids, in place."""
    dofs = _coordinate_array(node=list(node_ids), direction=direction)
    for dof in dofs:
        index = _np.where(system.coordinate == dof)[0][0]
        system.stiffness[index, index] += stiffness


def _add_end_ground_springs(system, num_elements, stiffness, direction=3):
    """Add soft grounding springs at both end nodes of a straight beam system, in place."""
    _add_ground_springs_at_nodes(system, [1, num_elements + 1], stiffness, direction=direction)


def _add_coupling_spring(system, dof_a, dof_b, stiffness):
    """Add a two-DOF coupling spring between `dof_a` and `dof_b`, in place."""
    index_a = _np.where(system.coordinate == dof_a)[0][0]
    index_b = _np.where(system.coordinate == dof_b)[0][0]
    system.stiffness[index_a, index_a] += stiffness
    system.stiffness[index_b, index_b] += stiffness
    system.stiffness[index_a, index_b] -= stiffness
    system.stiffness[index_b, index_a] -= stiffness


def _attach_dut_to_fixture(dut_system, fixture_system, dut_num_elements,
                           fixture_num_elements, fixture_length, attachment_offset,
                           dut_length, translational_stiffness, rotational_stiffness=None,
                           dut_attachment_inset=0, bolt_mass=0.0, node_offset=1000):
    """Couple a DUT system to a fixture system with finite connection springs.

    The DUT nodes nearest the documented attachment offset (measured from
    the left end of the fixture) are tied to the fixture nodes there via an
    explicit Z-translation spring (`translational_stiffness`) and, if
    `rotational_stiffness` is given, an explicit Y-rotation spring as well --
    both plain DOF-to-DOF springs with no offset/lever arm, per the
    dissertation source this test case is drawn from (Zwink 2020).

    `dut_attachment_inset` moves the tie points inward from the DUT's two end
    nodes by that many DUT elements (e.g. Case 1 connects at DUT nodes 2 and
    90, one element in from each end, leaving a short overhang -- visible in
    the source's own FEA figures -- rather than at the literal end nodes 1
    and 91).

    `bolt_mass`, if nonzero, adds a point mass to the fixture-side
    Z-translation DOF at each connection point, representing the connection
    hardware (bolts/brackets) the source model lumps there.

    The DUT's physical length (not its element count) determines how many
    fixture elements its span covers -- the two meshes don't necessarily
    share an element length (e.g. Case 3's 50 in/14-element DUT against a
    140 in/28-element fixture), so indexing the right attachment point by
    DUT element count would misplace it whenever the meshes differ.

    Returns
    -------
    assembled_system : System
    left_fixture_node : int
        Fixture node id (before offset) nearest the left attachment point.
    right_fixture_node : int
        Fixture node id (before offset) nearest the right attachment point.
    """
    fixture_element_length = fixture_length / fixture_num_elements
    dut_element_length = dut_length / dut_num_elements
    left_index = int(round(attachment_offset / fixture_element_length))
    dut_span_elements = int(round(dut_length / fixture_element_length))
    right_index = left_index + dut_span_elements

    fixture_inset = int(round(dut_attachment_inset * dut_element_length / fixture_element_length))
    left_index += fixture_inset
    right_index -= fixture_inset

    combined = _System.concatenate([dut_system, fixture_system],
                                   coordinate_node_offset=node_offset)
    dut_node_offset = node_offset * 1
    fixture_node_offset = node_offset * 2

    dut_nodes = (dut_node_offset + 1 + dut_attachment_inset,
                 dut_node_offset + dut_num_elements + 1 - dut_attachment_inset)
    fixture_nodes = (fixture_node_offset + left_index + 1, fixture_node_offset + right_index + 1)

    for dut_node, fixture_node in zip(dut_nodes, fixture_nodes):
        _add_coupling_spring(
            combined,
            _coordinate_array(node=dut_node, direction=3),
            _coordinate_array(node=fixture_node, direction=3),
            translational_stiffness)
        if rotational_stiffness is not None:
            _add_coupling_spring(
                combined,
                _coordinate_array(node=dut_node, direction=5),
                _coordinate_array(node=fixture_node, direction=5),
                rotational_stiffness)
        if bolt_mass:
            fixture_dof = _coordinate_array(node=fixture_node, direction=3)
            index = _np.where(combined.coordinate == fixture_dof)[0][0]
            combined.mass[index, index] += bolt_mass

    return combined, left_index + 1, right_index + 1


def create_case_1_models(lab_shift=12.0):
    """Build the Case 1 (Appendix A.6/A.7) two-beam analytical models.

    Parameters
    ----------
    lab_shift : float, optional
        Distance (inches) the DUT is shifted to the right on the fixture for
        the laboratory configuration, relative to the symmetric field
        configuration. The paper uses 12 inches.

    Returns
    -------
    dict with keys 'dut', 'fixture', 'field' (DUT-L1), 'lab' (DUT-L2),
    each a System, plus 'dut_geometry' and 'fixture_geometry'.
    """
    # The dissertation's own beam model is pure Euler-Bernoulli bending (no
    # Poisson's ratio, shear modulus, or torsion stiffness at all); nu is
    # only needed here because `beamkm` always builds a full 3D 6-DOF/node
    # element. Confirmed to have zero effect on the in-plane (Z-translation/
    # Y-rotation) frequencies this module and its tests actually check.
    nu = 0.33

    dut_length = 90.0625
    dut_num_elements = 90
    dut_ground_stiffness = 60.0  # lbf/in, at DUT nodes 2 and 90 (Zwink dissertation MATLAB App. A.4)
    # The dissertation's DUT ("modal beam") script uses 0.185in for area but
    # 3/16in (0.1875) for the bending inertias -- see box_beam_props'
    # `wall_thickness_bending` docstring for why this asymmetry is kept.
    dut_props = box_beam_props(E=1e7, rho=2.588e-4, nu=nu,
                               height=1.5, depth=3.0, wall_thickness=0.185,
                               wall_thickness_bending=0.1875,
                               nelem=dut_num_elements)

    fixture_length = 137.9375
    fixture_num_elements = 138
    fixture_ground_stiffness = 60.0  # lbf/in, at fixture nodes 24 and 116 (Zwink dissertation MATLAB App. A.2)
    # The dissertation's fixture ("support beam") script uses the opposite
    # pairing from the DUT: 3/16in (0.1875) for area, 0.185in for bending.
    fixture_props = box_beam_props(E=1.01e7, rho=2.588e-4, nu=nu,
                                   height=1.5, depth=3.0, wall_thickness=0.1875,
                                   wall_thickness_bending=0.185,
                                   nelem=fixture_num_elements)

    dut_system, dut_geometry = _straight_beam_system_and_geometry(
        dut_length, dut_num_elements, dut_props, color=1)
    fixture_system, fixture_geometry = _straight_beam_system_and_geometry(
        fixture_length, fixture_num_elements, fixture_props, color=7)

    _add_ground_springs_at_nodes(dut_system, [2, dut_num_elements], dut_ground_stiffness)
    _add_ground_springs_at_nodes(fixture_system, [24, 116], fixture_ground_stiffness)

    connection_translational_stiffness = 1e6  # lbf/in (Zwink dissertation MATLAB App. A.6)
    connection_rotational_stiffness = 2e4  # in-lbf/rad (Zwink dissertation MATLAB App. A.6)
    connection_bolt_mass = 0.681 / 386.09  # blob, at each fixture-side connection DOF (App. A.6)

    field_offset = (fixture_length - dut_length) / 2
    field_system, field_left, field_right = _attach_dut_to_fixture(
        dut_system, fixture_system, dut_num_elements, fixture_num_elements,
        fixture_length, field_offset, dut_length,
        connection_translational_stiffness, connection_rotational_stiffness,
        dut_attachment_inset=1, bolt_mass=connection_bolt_mass)

    lab_offset = field_offset + lab_shift
    lab_system, lab_left, lab_right = _attach_dut_to_fixture(
        dut_system, fixture_system, dut_num_elements, fixture_num_elements,
        fixture_length, lab_offset, dut_length,
        connection_translational_stiffness, connection_rotational_stiffness,
        dut_attachment_inset=1, bolt_mass=connection_bolt_mass)

    return {
        'dut': dut_system,
        'fixture': fixture_system,
        'field': field_system,
        'lab': lab_system,
        'dut_geometry': dut_geometry,
        'fixture_geometry': fixture_geometry,
        'field_connection_nodes': (field_left, field_right),
        'lab_connection_nodes': (lab_left, lab_right),
    }


def create_case_3_models(lab_shift=25.0):
    """Build the Case 3 (Appendix A.1-A.5) two-beam analytical models.

    Unlike Case 1, the field (`L1`) and laboratory (`L2`) fixtures here are
    physically different components: identical geometry, but `L2` is 10x
    stiffer than `L1` (Table 6 gives `L2`'s Young's Modulus as 10x `L1`'s),
    reproducing the "10xK^(L1)=K^(L2)" boundary condition mismatch shown in
    Fig. 22. The DUT also carries its own (very soft) grounding springs per
    Table 4, in addition to the fixture's grounding springs (Tables 5/6).

    Parameters
    ----------
    lab_shift : float, optional
        Distance (inches) the DUT is shifted to the right on the fixture for
        the laboratory configuration, relative to the symmetric field
        configuration. Not stated numerically in the paper itself, but
        Zwink's dissertation (the source of this test case) states the DUT
        "was translated to the right five elements" -- 5 fixture elements of
        5.0 in each, i.e. 25 in.

    Returns
    -------
    dict with keys 'dut', 'field_fixture' (L1), 'lab_fixture' (L2),
    'field' (DUT-L1), 'lab' (DUT-L2), plus geometries and connection nodes.
    """
    # The dissertation's own beam model is pure Euler-Bernoulli bending (no
    # Poisson's ratio, shear modulus, or torsion stiffness at all); nu is
    # only needed here because `beamkm` always builds a full 3D 6-DOF/node
    # element. Confirmed to have zero effect on the in-plane (Z-translation/
    # Y-rotation) frequencies this module and its tests actually check.
    nu = 0.33

    dut_length = 50.0
    dut_num_elements = 14
    dut_ground_stiffness = 1e-7  # lbf/in, Table 4
    dut_props = box_beam_props(E=1e7, rho=2.588e-4, nu=nu,
                               height=1.5, depth=3.0, wall_thickness=0.1875,
                               nelem=dut_num_elements)

    fixture_length = 140.0
    fixture_num_elements = 28
    fixture_ground_stiffness = 1e3  # lbf/in, Tables 5/6

    field_fixture_props = box_beam_props(E=1e7, rho=2.588e-4, nu=nu,
                                         height=1.5, depth=3.0, wall_thickness=0.1875,
                                         nelem=fixture_num_elements)
    lab_fixture_props = box_beam_props(E=1e8, rho=2.588e-4, nu=nu,
                                       height=1.5, depth=3.0, wall_thickness=0.1875,
                                       nelem=fixture_num_elements)

    dut_system, dut_geometry = _straight_beam_system_and_geometry(
        dut_length, dut_num_elements, dut_props, color=1)
    _add_end_ground_springs(dut_system, dut_num_elements, dut_ground_stiffness)

    field_fixture_system, field_fixture_geometry = _straight_beam_system_and_geometry(
        fixture_length, fixture_num_elements, field_fixture_props, color=7)
    _add_end_ground_springs(field_fixture_system, fixture_num_elements, fixture_ground_stiffness)

    lab_fixture_system, lab_fixture_geometry = _straight_beam_system_and_geometry(
        fixture_length, fixture_num_elements, lab_fixture_props, color=13)
    _add_end_ground_springs(lab_fixture_system, fixture_num_elements, fixture_ground_stiffness)

    connection_translational_stiffness = 1e6  # lbf/in, translation-only (dissertation p.87)

    field_offset = (fixture_length - dut_length) / 2
    field_system, field_left, field_right = _attach_dut_to_fixture(
        dut_system, field_fixture_system, dut_num_elements, fixture_num_elements,
        fixture_length, field_offset, dut_length, connection_translational_stiffness)

    lab_offset = field_offset + lab_shift
    lab_system, lab_left, lab_right = _attach_dut_to_fixture(
        dut_system, lab_fixture_system, dut_num_elements, fixture_num_elements,
        fixture_length, lab_offset, dut_length, connection_translational_stiffness)

    return {
        'dut': dut_system,
        'field_fixture': field_fixture_system,
        'lab_fixture': lab_fixture_system,
        'field': field_system,
        'lab': lab_system,
        'dut_geometry': dut_geometry,
        'field_fixture_geometry': field_fixture_geometry,
        'lab_fixture_geometry': lab_fixture_geometry,
        'field_connection_nodes': (field_left, field_right),
        'lab_connection_nodes': (lab_left, lab_right),
    }


def in_plane_shapes(system, num_modes, minimum_frequency=0.5, in_plane_energy_fraction=0.9):
    """Reduces a full 3D 6-DOF/node beam eigensolution down to the modes
    dominated by the planar (Z-translation / Y-rotation) bending behavior
    the paper's models represent.

    `beamkm` always builds a full 3D beam element (3 translations + 3
    rotations per node), but the paper's two-beam models are effectively
    planar (bending in a single plane only). The extra out-of-plane
    (X/Y-translation, X/Z-rotation) modes this introduces have no
    counterpart in the paper and would otherwise be interleaved with the
    physically relevant modes, so this filters them out by keeping only
    modes whose motion is at least `in_plane_energy_fraction` concentrated
    in the Z-translation/Y-rotation DOFs.

    Parameters
    ----------
    system : System
        A DUT, fixture, or assembled system from `create_case_1_models`/
        `create_case_3_models`.
    num_modes : int
        Number of modes to request from `System.eigensolution` before
        filtering (the filtered result will have fewer modes than this).
    minimum_frequency : float, optional
        Modes at or below this frequency (Hz) are discarded as rigid-body/
        grounding-spring artifacts. Default 0.5 Hz.
    in_plane_energy_fraction : float, optional
        Minimum fraction of a mode's total motion that must be in the
        Z-translation/Y-rotation DOFs for it to be kept. Default 0.9.

    Returns
    -------
    ShapeArray
    """
    shapes = system.eigensolution(num_modes=num_modes).flatten()
    coordinate = shapes[0].coordinate
    in_plane_mask = _np.isin(abs(coordinate.direction), [3, 5])
    modeshape = shapes.modeshape
    energy_total = _np.sum(_np.abs(modeshape)**2, axis=0)
    energy_in_plane = _np.sum(_np.abs(modeshape[in_plane_mask, :])**2, axis=0)
    keep = ((shapes.frequency > minimum_frequency)
           & (energy_in_plane / energy_total > in_plane_energy_fraction))
    return shapes[keep]


def sine_burst_force(frequency, duration=5.0, sample_rate=10000.0, amplitude=7.0):
    """Generates a Hann-windowed sine-burst excitation force and its FFT.

    Reproduces the excitation used for the paper's Case 1.1/1.2 sine
    demonstration, per Zwink's dissertation MATLAB (`generate_excitation_sine.m`):
    a sine wave at `frequency`, windowed by a Hann window covering the first
    half of the time record and zero for the second half. A windowed burst
    (rather than an idealized continuous, single-frequency tone) has spectral
    content spread over a nonzero bandwidth around `frequency` -- this is why
    a MACM built from it shows contributions from field modes near, but not
    only at, the one nominally being excited, matching the paper's own
    figures (see `plan.md` Section 6.8).

    Parameters
    ----------
    frequency : float
        Sine burst frequency (Hz).
    duration : float, optional
        Total signal duration (s). Default 5.0, matching the dissertation.
    sample_rate : float, optional
        Sample rate (Hz). Default 10000.0, matching the dissertation.
    amplitude : float, optional
        Sine amplitude (force units). Default 7.0, matching the dissertation.

    Returns
    -------
    frequencies : ndarray
        One-sided FFT frequency line (Hz), from 0 to the Nyquist frequency.
    spectrum : ndarray
        Complex one-sided FFT of the force signal, using the same convention
        as `sdynpy.core.sdynpy_data.TimeHistoryArray.fft` (`scipy.fft.rfft`,
        `norm="backward"`, unnormalized).
    """
    num_samples = int(round(duration * sample_rate))
    time = _np.arange(num_samples) / sample_rate
    force = amplitude * _np.sin(2 * _np.pi * frequency * time)
    window = _np.zeros(num_samples)
    half = num_samples // 2
    window[:half] = _hann(half)
    force = force * window
    spectrum = _rfft(force, norm="backward")
    frequencies = _rfftfreq(num_samples, d=1.0 / sample_rate)
    return frequencies, spectrum
