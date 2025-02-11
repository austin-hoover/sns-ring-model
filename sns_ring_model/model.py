"""SNS accumulator ring model."""
import math
import os
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any
from pprint import pprint

import numpy as np
import pandas as pd

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.foil import Foil
from orbit.core.impedances import LImpedance
from orbit.core.impedances import TImpedance
from orbit.core.spacecharge import LSpaceChargeCalc
from orbit.core.spacecharge import Boundary2D
from orbit.core.spacecharge import SpaceChargeCalc2p5D
from orbit.core.spacecharge import SpaceChargeCalcSliceBySlice2D
from orbit.aperture import CircleApertureNode
from orbit.aperture import EllipseApertureNode
from orbit.aperture import RectangleApertureNode
from orbit.aperture import TeapotApertureNode
from orbit.aperture.ApertureLatticeModifications import addTeapotApertureNode
from orbit.bumps import bumps
from orbit.bumps import BumpLatticeModifications
from orbit.bumps import TDTeapotSimpleBumpNode
from orbit.bumps import TeapotBumpNode
from orbit.bumps import TeapotSimpleBumpNode
from orbit.collimation import addTeapotCollimatorNode
from orbit.collimation import TeapotCollimatorNode
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import TeapotMomentsNode
from orbit.diagnostics import TeapotStatLatsNode
from orbit.diagnostics import TeapotTuneAnalysisNode
from orbit.foils import addTeapotFoilNode
from orbit.foils import TeapotFoilNode
from orbit.impedances import addImpedanceNode
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_TImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import FreqDep_TImpedance_Node
from orbit.impedances import LImpedance_Node
from orbit.impedances import TImpedance_Node
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import TeapotInjectionNode
from orbit.injection.joho import JohoTransverse
from orbit.injection.joho import JohoLongitudinal
from orbit.injection.distributions import SNSESpreadDist
from orbit.injection.distributions import UniformLongDist
from orbit.kickernodes import flatTopWaveform
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.rf_cavities import RFNode
from orbit.rf_cavities import RFLatticeModifications
from orbit.space_charge import sc2p5d
from orbit.space_charge import sc2dslicebyslice
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc2dslicebyslice.scLatticeModifications import setSC2DSliceBySliceAccNodes
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.time_dep.waveform import ConstantWaveform
from orbit.utils import consts
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light

from sns_ring_model.utils import get_node_by_name_any_case
from sns_ring_model.utils import get_nodes_by_name_any_case
from sns_ring_model.utils import rename_nodes_avoid_duplicates


# Node parts (these are integers 0, 1, 2)
ENTRANCE = AccNode.ENTRANCE
EXIT = AccNode.EXIT
BODY = AccNode.BODY

# Foil boundary relative to injection point [m]
FOIL_XMIN_REL = -0.005
FOIL_XMAX_REL = +0.014722
FOIL_YMIN_REL = -0.005
FOIL_YMAX_REL = +0.020


@dataclass
class CollimatorNodeInfo:
    length: float
    ma: int
    density_fac: float
    shape: int
    a: float
    b: float
    c: float
    d: float
    angle: float
    position: float


def make_aperture_node(
    name: str,
    class_name: str,
    size_x: float,
    size_y: float,
    position: float,
) -> AccNode:
    aperture_node = None
    if class_name == "CircleApertureNode":
        aperture_node = CircleApertureNode(size_x, pos=position, c=0, d=0, name=name)
    elif class_name == "EllipseApertureNode":
        aperture_node = EllipseApertureNode(size_x, size_y, pos=position, c=0, d=0, name=name)
    elif class_name == "RectangleApertureNode":
        aperture_node = RectangleApertureNode(size_x, size_y, pos=position, c=0, d=0, name=name)
    else:
        raise ValueError(f"Unknown aperture node class name {class_name}")
    return aperture_node


def read_longitudinal_impedance_file(path: str) -> list[complex]:
    z = []
    for real, imag in np.loadtxt(path, comments="#"):
        z.append(complex(real, imag))
    return z


def read_transverse_impedance_file(path: str) -> tuple[list[complex]]:
    zp, zm = [], []
    for m, zp_real, zp_imag, zm_real, zm_imag in np.loadtxt(path, comments="#"):
        zp.append(complex(zp_real, -zp_imag))
        zm.append(complex(zm_real, -zm_imag))
    return (zp, zm)


def read_lattice_file(lattice: AccLattice, filename: str, filetype: str, seq: str) -> AccLattice:
    if not os.path.exists(filename):
        raise FileNotFoundError
        
    if filetype.lower() == "madx":
        lattice.readMADX(filename, seq)
    elif filetype.lower() == "mad":
        lattice.readMAD(filename, seq)
    else:
        raise ValueError(f"Invalid file type {filetype}")

    return lattice


class AccModel:
    def __init__(self, verbose: int = 1) -> None:
        self.verbose = verbose


class SNS_RING(AccModel):
    def __init__(
        self,
        lattice_file: str = None,
        lattice_file_type: str = None,
        lattice_seq: str = None,
        inj_x: float = 0.0486,
        inj_y: float = 0.0460,
        inj_xp: float = 0.0,
        inj_yp: float = 0.0,
        foil_xmin_rel: float = None,
        foil_xmax_rel: float = None,
        foil_ymin_rel: float = None,
        foil_ymax_rel: float = None,
        avoid_duplicate_node_names: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.path = pathlib.Path(__file__)
        
        self.lattice = TEAPOT_Ring()

        self.bunch = None
        self.lostbunch = None
        self.params_dict = None

        self.foil_xmin_rel = foil_xmin_rel
        if self.foil_xmin_rel is None:
            self.foil_xmin_rel = FOIL_XMIN_REL

        self.foil_xmax_rel = foil_xmax_rel
        if self.foil_xmax_rel is None:
            self.foil_xmax_rel = FOIL_XMAX_REL

        self.foil_ymin_rel = foil_ymin_rel
        if self.foil_ymin_rel is None:
            self.foil_ymin_rel = FOIL_YMIN_REL

        self.foil_ymax_rel = foil_ymax_rel
        if self.foil_ymax_rel is None:
            self.foil_ymax_rel = FOIL_YMAX_REL

        self.inj_x = inj_x
        self.inj_y = inj_y
        self.inj_xp = inj_xp
        self.inj_yp = inj_yp
        self.foil_xmin = self.inj_x + self.foil_xmin_rel
        self.foil_xmax = self.inj_x + self.foil_xmax_rel
        self.foil_ymin = self.inj_y + self.foil_ymin_rel
        self.foil_ymax = self.inj_y + self.foil_ymax_rel

        self.aperture_nodes = []
        self.collimator_nodes = []
        self.foil_node = None
        self.injection_node = None
        self.transverse_impedance_nodes = []
        self.transverse_spacecharge_nodes = []
        self.longitudinal_impedance_nodes = []
        self.longitudinal_spacecharge_nodes = []
        self.rf_nodes = []

        self.inj_kicker_names = [
            "ikickh_a10",
            "ikickv_a10",
            "ikickh_a11",
            "ikickv_a11",
            "ikickv_a12",
            "ikickh_a12",
            "ikickv_a13",
            "ikickh_a13",
        ]
        self.solenoid_names = ["scbdsol_c13a", "scbdsol_c13b"]

        self.lattice_file = lattice_file
        self.lattice_file_type = lattice_file_type
        self.lattice_seq = lattice_seq

        # Read lattice file
        if self.lattice_file == "default":
            self.lattice_file = self.path.parent.joinpath("madx/sns_ring.lat")
            self.lattice_file_type = "madx"
            self.lattice_seq = "rnginj"
            
        if self.lattice_file is not None:
            self.lattice = read_lattice_file(
                self.lattice,
                self.lattice_file,
                self.lattice_file_type,
                self.lattice_seq,
            )

        self.avoid_duplicate_node_names = avoid_duplicate_node_names
        if self.avoid_duplicate_node_names:
            self.rename_nodes_avoid_duplicates()

    def rename_nodes_avoid_duplicates(self) -> None:
        self.lattice = rename_nodes_avoid_duplicates(self.lattice, self.verbose)

    def initialize(self) -> None:
        """Adds on to `lattice.initialize()`. Call this after the lattice is loaded."""
        self.lattice.initialize()

        self.inj_kicker_nodes = []
        for name in self.inj_kicker_names:
            for node in self.lattice.getNodes():
                if node.getName().lower() == name:
                    self.inj_kicker_nodes.append(node)

        self.solenoid_nodes = []
        for node in self.lattice.getNodes():
            if node.getName().lower() in self.solenoid_names:
                self.solenoid_nodes.append(node)
                # Set strength manually.
                factor = 0.25
                node.setParam("B", factor * 0.6 / (2.0 * node.getLength()))

    def get_lattice(self) -> None:
        return self.lattice

    def set_bunch(self, bunch: Bunch, lostbunch: Bunch = None, params_dict: dict = None) -> None:
        self.bunch = bunch

        self.lostbunch = lostbunch
        if self.lostbunch is None:
            self.lostbunch = Bunch()

        self.params_dict = params_dict
        if self.params_dict is None:
            self.params_dict = {}
            self.params_dict["bunch"] = bunch
            self.params_dict["lostbunch"] = lostbunch

    def set_fringe(self, setting: bool) -> None:
        for node in self.lattice.getNodes():
            try:
                node.setUsageFringeFieldIN(setting)
                node.setUsageFringeFieldOUT(setting)
            except:
                pass

    def add_injection_node(
        self,
        n_parts: int,
        dist_x: Any,
        dist_y: Any,
        dist_z: Any,
        n_parts_max: int = None,
        parent_index: int = 0,
    ) -> AccNode:
        """Add injection node as child node.

        Parameters
        ----------
        n_parts: int
            Number of macroparticles per injection.
        dist_{x, y, z}: Any
            Must implement `(x, xp) = getCoordinates()`, which generates a random
            coordinate from the two-dimensional position-momentum distribution.
        parent_index: int
            Index of parent node. The injection node is added as a child of this node.
            
        Returns
        -------
        orbit.injection.TeapotInjectionNode
        """
        boundary = [
            self.foil_xmin,
            self.foil_xmax,
            self.foil_ymin,
            self.foil_ymax,
        ]
        self.injection_node = TeapotInjectionNode(
            n_parts,
            self.bunch,
            self.lostbunch,
            boundary,
            dist_x,
            dist_y,
            dist_z,
            n_parts_max,
        )
        parent_node = self.lattice.getNodes()[parent_index]
        parent_node.addChildNode(self.injection_node, ENTRANCE)
        return self.injection_node

    def add_foil_node(
        self, thickness: float = 390.0, scatter: str = "full", parent_index: int = 0,
    ) -> AccNode:
        """Add foil scattering node.

        The foil boundaries are defined in the constructor.

        Parameters
        ----------
        thickness: float
            Foil thickness [mm].
        scatter: {"simple", "full"}
            Specifies foil scattering algorithm.
        parent_index: int
            Index of parent node. The injection node is added as a child of this node.
            
        Returns
        -------
        orbit.foils.TeapotFoilNode
        """
        scatter_options = {"full": 0, "simple": 1}
        scatter = scatter_options[scatter]
        
        self.foil_node = TeapotFoilNode(
            self.foil_xmin, 
            self.foil_xmax, 
            self.foil_ymin, 
            self.foil_ymax, 
            thickness
        )
        self.foil_node.setScatterChoice(scatter)

        parent_node = self.lattice.getNodes()[parent_index]
        parent_node.addChildNode(self.foil_node, ENTRANCE)
        return self.foil_node

    def add_transverse_spacecharge_nodes(
        self,
        solver: str = "slicebyslice",
        gridx: int = 128,
        gridy: int = 128,
        gridz: int = 64,
        path_length_min: float = 1.00e-08,
        n_macros_min: int = 1000,
        boundary: bool = True,
        boundary_modes: int = 32,
        boundary_points: int = 128,
        boundary_radius: float = 0.220,
    ) -> list[AccNode]:
        """Add transverse space charge nodes throughout the ring.

        Parameters
        ----------
        solver: str
        gridx, gridy: int
            Transverse x-y grid resolution.
        gridz: int
            Number of longitudinal slices.
        path_length_min: float

        Returns
        -------
        list[orbit.space_charge.SC2p5DAccNodes]
        list[orbit.space_charge.SC2DSliceBySliceAccNodes]
        """
        if boundary:
            boundary = Boundary2D(
                boundary_points, boundary_modes, "Circle", boundary_radius, boundary_radius
            )
        if not boundary:
            boundary = None

        self.transverse_spacecharge_nodes = []
        if solver == "2p5d":
            calculator = SpaceChargeCalc2p5D(gridx, gridy, gridz)
            self.transverse_spacecharge_nodes = setSC2p5DAccNodes(
                self.lattice,
                path_length_min,
                calculator,
                boundary=boundary,
            )
        elif solver in ["slicebyslice", "slice", "sbs"]:
            calculator = SpaceChargeCalcSliceBySlice2D(gridx, gridy, gridz)
            self.transverse_spacecharge_nodes = setSC2DSliceBySliceAccNodes(
                self.lattice,
                path_length_min,
                calculator,
                boundary=boundary
            )
        return self.transverse_spacecharge_nodes

    def add_longitudinal_spacecharge_node(
        self,
        b_over_a: float = (10.0 / 3.0),
        n_macros_min: int = 1000,
        n_bins: int = 64,
        position: float = 124.0,
        impedance: list[float] = None,
    ) -> AccNode:
        """Add a longitudinal space charge at one position in the ring.

        Parameters
        ----------
        ...

        Returns
        -------
        orbit.space_charge.SC1D_AccNode
        """
        sc_node = SC1D_AccNode(b_over_a, self.lattice.getLength(), n_macros_min, 1, n_bins)
        if impedance is None:
            impedance = [complex(0.0, 0.0) for _ in range(n_bins // 2)]
        sc_node.assignImpedance(impedance)
        addLongitudinalSpaceChargeNode(self.lattice, position, sc_node)
        
        self.longitudinal_spacecharge_nodes.append(sc_node)
        return sc_node

    def add_longitudinal_impedance_node(
        self,
        n_macros_min: int = 1000,
        n_bins: int = 128,
        position: float = 124.0,
        Zk: list[complex] = None,
        Zrf: list[complex] = None,
    ) -> AccNode:
        """Add a longitudinal impedance node at one position in the ring.

        Parameters
        ----------
        ...

        Returns
        -------
        orbit.impedances.LImpedance_Node
        """

        if Zk is None:
            Zk = read_longitudinal_impedance_file(
                self.path.parent.joinpath("data/longitudinal_impedance_ekicker.dat")
            )
        if Zrf is None:
            Zrf = read_longitudinal_impedance_file(
                self.path.parent.joinpath("data/longitudinal_impedance_rf.dat")
            )
        
        Z = []
        for zk, zrf in zip(Zk, Zrf):
            z_real = (zk.real / 1.75) + zrf.real
            z_imag = (zk.imag / 1.75) + zrf.imag
            Z.append(complex(z_real, z_imag))

        length = self.lattice.getLength()
        impedance_node = LImpedance_Node(length, n_macros_min, n_bins)
        impedance_node.assignImpedance(Z)
        addImpedanceNode(self.lattice, position, impedance_node)
        
        self.longitudinal_impedance_nodes.append(impedance_node)
        return impedance_node

    def add_transverse_impedance_node(
        self,
        n_macros_min: int = 1000,
        n_bins: int = 64,
        use_x: bool = False,
        use_y: bool = True,
        alpha_x: float = 0.0,
        alpha_y: float = -0.004,
        beta_x: float = 10.191,
        beta_y: float = 10.447,
        tune_x: float = 6.21991,
        tune_y: float = 6.20936,
        ZP: list[complex] = None,
        ZM: list[complex] = None,
        position: float = 124.0,
    ) -> AccNode:
        """Add a transverse impedance node at one position in the ring.

        NOTE: Calculations require Twiss parameters as input. What if the lattice is coupled?
        Also, why were these values hard-coded rather than computed from the linear lattice?

        Parameters
        ----------
        ...

        Returns
        -------
        orbit.impedances.TImpedance_Node
        """
        if (ZP is None) or (ZM is None):
            (ZP, ZM) = read_transverse_impedance_file(
                self.path.parent.joinpath("./data/hahn_impedance.dat")
            )

        impedance_node = TImpedance_Node(
            self.lattice.getLength(), n_macros_min, n_bins, int(use_x), int(use_y)
        )
        impedance_node.assignLatFuncs(tune_x, alpha_x, beta_x, tune_y, alpha_y, beta_y)
        if use_x:
            impedance_node.assignImpedance("X", ZP, ZM)
        if use_y:
            impedance_node.assignImpedance("Y", ZP, ZM)
        addImpedanceNode(self.lattice, position, impedance_node)
        
        self.transverse_impedance_nodes.append(impedance_node)
        return impedance_node

    def add_rf_cavity_nodes(
        self,
        voltage_1: float = 0.0,
        voltage_2: float = 0.0,
        voltage_3: float = 0.0,
        voltage_4: float = 0.0,
        hnum_1: float = 1.0,
        hnum_2: float = 1.0,
        hnum_3: float = 2.0,
        hnum_4: float = 2.0,
        phase_1: float = 0.0,
        phase_2: float = 0.0,
        phase_3: float = 0.0,
        phase_4: float = 0.0,
        position_1: float = 184.273,
        position_2: float = 186.571,
        position_3: float = 188.868,
        position_4: float = 191.165,
        synchronous_de: float = 0.0,
    ) -> list[AccNode]:
        """Add harmonic rf cavity nodes to the lattice.

        Parameters
        ----------
        voltage_{1, 2, 3, 4} : float
            Voltages [GV].
        hnum_{1, 2, 3, 4} : float
            Harmonic numbers.
        phase_{1, 2, 3, 4} : float
            Phases [rad].
        position_{1, 2, 3, 4} : float
            Node positions. We assume zero length.
        synchronous_de : float
            Synchronous particle dE [GeV].
        
        Returns
        -------
        list[orbit.rf_cavities.Harmonic_RFNode]
        """
        z_to_phi = 2.0 * math.pi / self.lattice.getLength()
        length = 0.0

        rf_node_1 = RFNode.Harmonic_RFNode(z_to_phi, synchronous_de, hnum_1, voltage_1, phase_1, length, "RF1")
        rf_node_2 = RFNode.Harmonic_RFNode(z_to_phi, synchronous_de, hnum_2, voltage_2, phase_2, length, "RF2")
        rf_node_3 = RFNode.Harmonic_RFNode(z_to_phi, synchronous_de, hnum_3, voltage_3, phase_3, length, "RF3")
        rf_node_4 = RFNode.Harmonic_RFNode(z_to_phi, synchronous_de, hnum_4, voltage_4, phase_4, length, "RF4")

        RFLatticeModifications.addRFNode(self.lattice, position_1, rf_node_1)
        RFLatticeModifications.addRFNode(self.lattice, position_2, rf_node_2)
        RFLatticeModifications.addRFNode(self.lattice, position_3, rf_node_3)
        RFLatticeModifications.addRFNode(self.lattice, position_4, rf_node_4)

        self.rf_nodes = [rf_node_1, rf_node_2, rf_node_3, rf_node_4]
        return self.rf_nodes

    def add_injection_chicane_aperture_and_displacement_nodes(self) -> list[AccNode]:
        """Add apertures and displacements to injection chicane.

        WARNING: This function currently only works with the MAD lattice file. 

        [To do] Build dictionary of aperture nodes along with parent nodes,
        then add them in a loop and register to self.aperture_nodes.
        """
        bunch = self.bunch
        aperture_nodes = {}

        xcenter = 0.100
        xb10m = 0.0
        xb11m = 0.09677
        xb12m = 0.08899
        xb13m = 0.08484

        ycenter = 0.023

        bumpwave = flatTopWaveform(1.0)

        xb10i = 0.0
        apb10xi = -xb10i
        apb10yi = ycenter
        rb10i = 0.1095375
        appb10i = CircleApertureNode(rb10i, 244.812, apb10xi, apb10yi, name="b10i")

        xb10f = 0.022683
        apb10xf = -xb10f
        apb10yf = ycenter
        rb10f = 0.1095375
        appb10f = CircleApertureNode(rb10f, 245.893, apb10xf, apb10yf, name="b10f")

        mag10x = (xb10i + xb10f) / 2.0 - xb10m
        cdb10i = TDTeapotSimpleBumpNode(bunch, +mag10x, 0.0, -ycenter, 0.0, bumpwave, "mag10bumpi")
        cdb10f = TDTeapotSimpleBumpNode(bunch, -mag10x, 0.0, +ycenter, 0.0, bumpwave, "mag10bumpf")

        xb11i = 0.074468
        apb11xi = xcenter - xb11i
        apb11yi = ycenter
        rb11i = 0.1095375
        appb11i = CircleApertureNode(rb11i, 247.265, apb11xi, apb11yi, name="b11i")

        xfoil = 0.099655
        apfoilx = xcenter - xfoil
        apfoily = ycenter
        rfoil = 0.1095375
        appfoil = CircleApertureNode(rfoil, 248.009, apfoilx, apfoily, name="bfoil")

        mag11ax = (xb11i + xfoil) / 2.0 - xb11m
        cdb11i = TDTeapotSimpleBumpNode(bunch, +mag11ax, 0.0, -ycenter, 0.0, bumpwave, "mag11bumpi")
        cdfoila = TDTeapotSimpleBumpNode(bunch, -mag11ax, 0.0, +ycenter, 0.0, bumpwave, "foilbumpa")

        xb11f = 0.098699
        apb11xf = xcenter - xb11f
        apb11yf = ycenter
        rb11f = 0.1095375
        appb11f = CircleApertureNode(rb11f, 0.195877, apb11xf, apb11yf, name="b11f")

        mag11bx = (xfoil + xb11f) / 2.0 - xb11m
        cdfoilb = TDTeapotSimpleBumpNode(bunch, mag11bx, 0.0, -ycenter, 0.0, bumpwave, "foilbumpb")
        cdb11f  = TDTeapotSimpleBumpNode(bunch, -mag11bx, 0.0, ycenter, 0.0, bumpwave, "mag11bumpf")

        xb12i = 0.093551
        apb12xi = xcenter - xb12i
        apb12yi = ycenter
        rb12i = 0.1095375
        appb12i = CircleApertureNode(rb12i, 1.08593, apb12xi, apb12yi, name="b12i")

        xb12f = 0.05318
        apb12xf = xcenter - xb12f
        apb12yf = ycenter
        rb12f = 0.1174750
        appb12f = CircleApertureNode(rb12f, 1.99425, apb12xf, apb12yf, name="b12f")

        mag12x = (xb12i + xb12f) / 2.0 - xb12m
        cdb12i = TDTeapotSimpleBumpNode(bunch, +mag12x, 0.0, -ycenter, 0.0, bumpwave, "mag12bumpi")
        cdb12f = TDTeapotSimpleBumpNode(bunch, -mag12x, 0.0, ycenter, 0.0, bumpwave, "mag12bumpf")

        xb13i = 0.020774
        apb13xi = xcenter - xb13i
        apb13yi = ycenter
        h13xi = 0.1913
        v13xi = 0.1016
        appb13i = RectangleApertureNode(h13xi, v13xi, 3.11512, apb13xi, apb13yi, name="b13i")

        xb13f = 0.0
        apb13xf = xcenter - xb13f
        apb13yf = ycenter
        h13xf = 0.1913
        v13xf = 0.1016
        appb13f = RectangleApertureNode(h13xf, v13xf, 4.02536, apb13xf, apb13yf, name="b13f")

        mag13x = (xb13i + xb13f) / 2.0 - xb13m
        cdb13i = TDTeapotSimpleBumpNode(bunch, +mag13x, 0.0, -ycenter, 0.0, bumpwave, "mag13bumpi")
        cdb13f = TDTeapotSimpleBumpNode(bunch, -mag13x, 0.0, ycenter, 0.0, bumpwave, "mag13bumpf")

        # Get parent nodes
        inj_chicane_node_names = ["DH_A10", "DH_A11A", "DH_A11B", "DH_A12", "DH_A13"]
        inj_chicane_nodes = get_nodes_by_name_any_case(self.lattice, inj_chicane_node_names)
        (dha10, dha11a, dha11b, dha12, dha13) = inj_chicane_nodes

        # Add apertures as child nodes
        dha10.addChildNode(appb10i, ENTRANCE)
        dha10.addChildNode(cdb10i, ENTRANCE)
        dha10.addChildNode(cdb10f, EXIT)
        dha10.addChildNode(appb10f, EXIT)
        dha11a.addChildNode(appb11i, ENTRANCE)
        dha11a.addChildNode(cdb11i, ENTRANCE)
        dha11a.addChildNode(cdfoila, EXIT)
        dha11a.addChildNode(appfoil, EXIT)
        dha11b.addChildNode(cdfoilb, ENTRANCE)
        dha11b.addChildNode(cdb11f, EXIT)
        dha11b.addChildNode(appb11f, EXIT)
        dha12.addChildNode(appb12i, ENTRANCE)
        dha12.addChildNode(cdb12i, ENTRANCE)
        dha12.addChildNode(cdb12f, EXIT)
        dha12.addChildNode(appb12f, EXIT)
        dha13.addChildNode(appb13i, ENTRANCE)
        dha13.addChildNode(cdb13i, ENTRANCE)
        dha13.addChildNode(cdb13f, EXIT)
        dha13.addChildNode(appb13f, EXIT)

    def add_collimator_nodes(self) -> None:
        # Make info list to avoid writing position twice. (There is no
        # Collimator.getPosition() method. This should probably be added.)
        collimator_node_info_list = [
            CollimatorNodeInfo(0.60000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0, 0.0, 50.3771),
            CollimatorNodeInfo(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 0.0, 51.1921),
            CollimatorNodeInfo(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 0.0, 51.1966),
            CollimatorNodeInfo(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 90.0, 51.2365),
            CollimatorNodeInfo(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 90.0, 51.2410),
            CollimatorNodeInfo(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 45.0, 51.3902),
            CollimatorNodeInfo(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, 45.0, 51.3947),
            CollimatorNodeInfo(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4346),
            CollimatorNodeInfo(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4391),
            CollimatorNodeInfo(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0, 0.0, 51.6228),
            CollimatorNodeInfo(1.80000, 3, 0.65, 2, 0.080, 0.048, 0.0, 0.0, 0.0, 51.9428),
            CollimatorNodeInfo(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0, 0.0, 53.7428),
            CollimatorNodeInfo(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 57.2079),
            CollimatorNodeInfo(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 58.2179),
            CollimatorNodeInfo(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 59.4079),
            CollimatorNodeInfo(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 64.8679),
            CollimatorNodeInfo(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 65.8779),
            CollimatorNodeInfo(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0, 0.0, 67.0679),
        ]
        for collimator_node_info in collimator_node_info_list:
            collimator_node = TeapotCollimatorNode(
                collimator_node_info.length,
                collimator_node_info.ma,
                collimator_node_info.density_fac,
                collimator_node_info.shape,
                collimator_node_info.a,
                collimator_node_info.b,
                collimator_node_info.c,
                collimator_node_info.d,
                collimator_node_info.angle,
                collimator_node_info.position
            )
            position = collimator_node_info.position
            addTeapotCollimatorNode(self.lattice, position, collimator_node)
            self.collimator_nodes.append(collimator_node)

    def add_aperture_nodes_around_ring_by_index(self) -> None:
        """Add aperture and collimator nodes.

        [Accesses nodes by index: works with MAD output lattice file, not MADX.]
        """
        raise NotImplementedError

    def add_aperture_nodes_around_ring(self) -> None:
        """WARNING: This function currently only works with the MAD lattice file."""

        filename = self.path.parent.joinpath("./data/aperture_node_info.txt")

        data = pd.read_csv(filename)
        for i in range(data.shape[0]):
            # Build node
            name = data.loc[i, "name"]
            class_name = data.loc[i, "class_name"]
            size_x = data.loc[i, "size_x"]
            size_y = data.loc[i, "size_y"]
            position = data.loc[i, "position"]

            aperture_node = make_aperture_node(
                name=name,
                class_name=class_name,
                size_x=size_x,
                size_y=size_y,
                position=position
            )

            # Add node as child
            parent_name = data.loc[i, "parent_name"]
            parent_node = get_node_by_name_any_case(self.lattice, parent_name)
            place = data.loc[i, "place"]
            part_index = data.loc[i, "part_index"]

            if parent_node is None:
                raise ValueError(f"Did not find node '{parent_name}'")
            parent_node.addChildNode(aperture_node, place, part_index)

    def add_all_aperture_and_collimator_nodes(self) -> None:
        """WARNING: This function currently only works with the MAD lattice file."""
        self.add_injection_chicane_aperture_and_displacement_nodes()
        self.add_aperture_nodes_around_ring()
        self.add_collimator_nodes()

    def get_solenoid_strengths(self) -> np.ndarray:
        return np.ndarray([node.getParam("B") for node in self.solenoid_nodes])

    def set_solenoid_strengths(self, B: float) -> None:
        """Set solenoid magnet strengths [units]."""
        for node in self.solenoid_nodes:
            node.setParam("B", B)
