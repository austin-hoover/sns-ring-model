"""Library for ring injection control."""
import math
from typing import Any
from typing import Callable

import numpy as np
import scipy.optimize as opt

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.injection import TeapotInjectionNode
from orbit.injection import JohoTransverse
from orbit.injection import JohoLongitudinal
from orbit.injection import SNSESpreadDist
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.utils.consts import speed_of_light


def get_momentum(mass: float, kin_energy: float) -> float:
    """Return momentum from mass [GeV / c^2] and kinetic energy [GeV]."""
    return np.sqrt(kin_energy * (kin_energy + 2.0 * mass))


def get_magnetic_rigidity(mass: float, kin_energy: float) -> float:
    """Return magnetic rigidity (Brho) from mass [GeV / c^2] and kinetic energy [GeV]."""
    momentum = get_momentum(mass=mass, kin_energy=kin_energy)
    return factor * momentum / speed_of_light

def get_node_for_name_any_case(lattice: AccLattice, name: str) -> AccNode:
    """Get node by name, case-insensitive."""
    nodes = lattice.getNodes()
    node_names = [node.getName() for node in nodes]
    if name not in node_names:
        if name.lower() in node_names:
            name = name.lower()
        elif name.upper() in node_names:
            name = name.upper()
        else:
            raise ValueError(f"node {name} not found")
    return lattice.getNodeForName(name)


def get_inj_kicker_angle_limits(mass: float, kin_energy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return limits on injection kicker angles.

    Parameters
    ----------
    mass: float
        Particle mass [GeV].
    kin_energy: float
        Particle kinetic energy [GeV].

    Returns
    -------
    min_angles, max_angles: np.ndarray
        Array of minimum and maximum kicker angles [rad].
    """
    
    # Maximum angles at 1.3 GeV:
    min_angles = 1.15 * np.array([0.0, 0.0, -7.13, -7.13, -7.13, -7.13, 0.0, 0.0])  # [mrad]
    max_angles = 1.15 * np.array([12.84, 12.84, 0.0, 0.0, 0.0, 0.0, 12.84, 12.84])  # [mrad]
    min_angles *= 0.001  # [mrad] -> [rad]
    max_angles *= 0.001  # [mrad] -> [rad]

    # Scale limits to the specified energy
    brho_old = get_magnetic_rigidity(mass=mass, kin_energy=1.300)
    brho_new = get_magnetic_rigidity(mass=mass, kin_energy=kin_energy)
    scale = brho_old / brho_new
    min_angles *= scale
    max_angles *= scale
    return (min_angles, max_angles)


def get_inj_corrector_angle_limits(mass: float, kin_energy: float) -> tuple[float]:
    """Return limits on orbit correctors in injection region.

    Parameters
    ----------
    mass: float
        Particle mass [GeV].
    kin_energy: float
        Particle kinetic energy [GeV].

    Returns
    -------
    min_angles, max_angles: np.ndarray
        Array of minimum and maximum corrector angles [rad].
    """
    max_angle = 1.5e-03  # [rad]
    min_angle = -max_angle
    brho_old = get_magnetic_rigidity(mass=mass, kin_energy=1.300)
    brho_new = get_magnetic_rigidity(mass=mass, kin_energy=kin_energy)
    scale = brho_old / brho_new
    min_angle *= scale
    max_angle *= scale
    return (min_angle, max_angle)


class InjectionController:    
    """Controls ring injection kicker magnets."""
    def __init__(
        self,
        lattice: AccLattice,
        bunch: Bunch,
        inj_start: str = "bpm_a09",
        inj_mid: str = "injm1",
        inj_stop: str = "bpm_b01",
    ) -> None:
        """Constructor.
        
        Parameters
        ----------
        lattice: AccLattice
            The accelerator lattice representing the SNS Ring.
        bunch: Bunch
            Bunch with correct energy and mass used for single particle tracking.
            This bunch will copied, so it does not need to be empty.
        inj_start, inj_stop, inj_mid: str
            First, last, and middle point in injection region. The PyORBIT lattice
            begins from the mid point and circles back to inj_start.
        """
        self.mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        self.mpi_rank = orbit_mpi.MPI_Comm_rank(self.mpi_comm)

        self.lattice = lattice
        
        self.bunch = Bunch()
        bunch.copyEmptyBunchTo(self.bunch)

        self.mass = self.bunch.mass()
        self.kin_energy = self.bunch.getSyncParticle().kinEnergy()
        
        self.kicker_names = [
            "ikickh_a10",
            "ikickv_a10",
            "ikickh_a11",
            "ikickv_a11",
            "ikickv_a12",
            "ikickh_a12",
            "ikickv_a13",
            "ikickh_a13",
        ]
        self.kicker_nodes = [get_node_for_name_any_case(lattice, name) for name in self.kicker_names]
        
        # Get kicker limits (rad)
        self.min_kicker_angles, self.max_kicker_angles = get_inj_kicker_angle_limits(self.mass, self.kin_energy)

        # Identify the horizontal and vertical kicker nodes.
        self.kicker_idx_x = [0, 2, 5, 7]
        self.kicker_idx_y = [1, 3, 4, 6]
        self.kicker_nodes_x = [self.kicker_nodes[i] for i in self.kicker_idx_x]
        self.kicker_nodes_y = [self.kicker_nodes[i] for i in self.kicker_idx_y]
        self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
        self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
        self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
        self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]

        # Identify dipole correctors. These will be used to make a closed bump.
        self.vcorrector_names = ["dmcv_a09", "dchv_a10", "dchv_a13", "dmcv_b01"]
        self.vcorrector_nodes = [get_node_for_name_any_case(lattice, name) for name in self.vcorrector_names]
        self.min_vcorrector_angle, self.max_vcorrector_angle = get_inj_corrector_angle_limits(self.mass, self.kin_energy)

        # Create one sublattice for the first half of the injection region (before the foil)
        # and one sublattice for the second half of the injection region (after the foil).
        lo = lattice.getNodeIndex(get_node_for_name_any_case(lattice, inj_start))
        hi = -1
        self.sublattice1 = self.lattice.getSubLattice(lo, hi)

        lo = lattice.getNodeIndex(get_node_for_name_any_case(lattice, inj_mid))
        hi = lattice.getNodeIndex(get_node_for_name_any_case(lattice, inj_stop))
        self.sublattice2 = self.lattice.getSubLattice(lo, hi)

        self.sublattices = [self.sublattice1, self.sublattice2]
    
        # Initialize the bunch for single-particle tracking.
        self.bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_nodes(self) -> list[AccNode]:
        return self.sublattice1.getNodes() + self.sublattice2.getNodes()

    def get_kicker_angles_x(self) -> np.ndarray:
        return np.array([node.getParam("kx") for node in self.kicker_nodes_x])

    def get_kicker_angles_y(self) -> np.ndarray:
        return np.array([node.getParam("ky") for node in self.kicker_nodes_y])

    def get_kicker_angles(self) -> np.ndarray:
        angles = []
        for node in self.kicker_nodes:
            if node in self.kicker_nodes_x:
                angles.append(node.getParam("kx"))
            elif node in self.kicker_nodes_y:
                angles.append(node.getParam("ky"))
        return np.array(angles)

    def set_kicker_angles_x(self, angles_x: np.ndarray) -> None:
        if angles_x is not None:
            for angle, node in zip(angles_x, self.kicker_nodes_x):
                node.setParam("kx", angle)

    def set_kicker_angles_y(self, angles_y: np.ndarray) -> None:
        if angles_y is not None:
            for angle, node in zip(angles_y, self.kicker_nodes_y):
                node.setParam("ky", angle)

    def set_kicker_angles(self, angles: np.ndarray) -> None:
        angles_x = [angles[i] for i in self.kicker_idx_x]
        angles_y = [angles[i] for i in self.kicker_idx_y]
        self.set_kicker_angles_x(angles_x)
        self.set_kicker_angles_y(angles_y)

    def get_vcorrector_angles(self) -> np.ndarray:
        return np.array([node.getParam("ky") for node in self.vcorrector_nodes])

    def set_vcorrector_angles(self, angles: np.ndarray):
        for angle, node in zip(angles, self.vcorrector_nodes):
            node.setParam("ky", angle)

    def init_part(self, x: float = 0.0, xp: float = 0.0, y: float = 0.0, yp: float = 0.0):
        self.bunch.deleteParticle(0)
        self.bunch.addParticle(x, xp, y, yp, 0.0, 0.0)

    def track_part(self, sublattice: int = 0):
        self.sublattices[sublattice].trackBunch(self.bunch)
        x = self.bunch.x(0)
        y = self.bunch.y(0)
        xp = self.bunch.xp(0)
        yp = self.bunch.yp(0)
        return np.array([x, xp, y, yp])

    def get_inj_coords(self) -> np.array:
        self.init_part(0.0, 0.0, 0.0, 0.0)
        return self.track_part(sublattice=0)

    def set_inj_coords(self, x: float, xp: float, y: float, yp: float, method: str = "bfgs") -> None:
        """Try to set phase space coordinates of the closed orbit at injection."""
        coords = np.array([x, xp, y, yp])
        
        def magnitude(coords: np.ndarray, scale: float = 1.00e+04) -> float:
            return scale * np.sum(np.square(coords))
                            
        def loss_function(angles: np.ndarray) -> float:
            self.set_kicker_angles(angles)
            self.init_part(0.0, 0.0, 0.0, 0.0)
            coords_mid = self.track_part(sublattice=0)
            coords_end = self.track_part(sublattice=1)

            if self.mpi_rank == 0:
                (x_beg, xp_beg, y_beg, yp_beg) = np.zeros(4)
                (x_mid, xp_mid, y_mid, yp_mid) = coords_mid * 1000.0
                (x_end, xp_end, y_end, yp_end) = coords_end * 1000.0
                print(f"x_beg={x_beg:+0.3f} xp_beg={xp_beg:+0.3f} y_beg={y_beg:+0.3f} yp_beg={yp_beg:+0.3f}")
                print(f"x_mid={x_mid:+0.3f} xp_mid={xp_mid:+0.3f} y_mid={y_mid:+0.3f} yp_mid={yp_mid:+0.3f}")
                print(f"x_end={x_end:+0.3f} xp_end={xp_end:+0.3f} y_end={y_end:+0.3f} yp_end={yp_end:+0.3f}")
                print()
            
            loss = magnitude(coords_mid - coords) + magnitude(coords_end)
            return loss
            
        lb = self.min_kicker_angles
        ub = self.max_kicker_angles
        x0 = np.zeros(8)

        result = None
        if method == "bfgs":
            result = opt.minimize(
                loss_function, 
                x0, 
                bounds=opt.Bounds(lb, ub), 
                method="L-BFGS-B", 
                options=dict(disp=1, ftol=1.00e-12, gtol=1.00e-12)
            )
        elif method == "least_squares":
            result = opt.least_squares(
                loss_function, 
                x0, 
                bounds=(lb, ub), 
                max_nfev=10_000,
                verbose=2, 
            )
        else:
            raise ValueError(f"Invalid method {method}")
        return result.x


# Injected distribution

def make_dist_joho(
    order: float,
    alpha: float,
    beta: float,
    emittance: float,
    centerpos: float,
    centermom: float,
) -> JohoTransverse:
    """Utility: make transverse Joho distribution."""
    emitlim = emittance * 2.0 * (order + 1.0)
    dist = JohoTransverse(order, alpha, beta, emitlim, centerpos, centermom)
    return dist


def make_dist_sns_espread(
    bunch: Bunch,
    lattice: AccLattice,
    zlim: float = (40.0 / 64.0),
    tailfraction: float = 0.0,
    esigma: float = 0.0005,
    etrunc: float = 1.0,
    emin: float = -0.0025,
    emax: float = +0.0025,
    ecmean: float = 0.0,
    ecsigma: float = 0.000000001,
    ectrunc: float = 1.0,
    ecmin: float = -0.0035,
    ecmax: float = +0.0035,
    ecdrifti: float = 0.0,
    ecdriftf: float = 0.0,
    esnu: float = 100.0,
    esphase: float = 0.0,
    esmax: float = 0.0, 
) -> SNSESpreadDist:
    """Utility: make longitudinal `SNSESpreadDist` distribution."""

    zlim = np.multiply(zlim, 0.5 * lattice.getLength())
    zmin = -zlim
    zmax = +zlim

    sync_part = bunch.getSyncParticle()
    emean = sync_part.kinEnergy()
    emin += sync_part.kinEnergy()
    emax += sync_part.kinEnergy()

    tturn = lattice.getLength() / (sync_part.beta() * speed_of_light)

    drifttime = 1000.0 * (1000) * tturn
    ecparams = (ecmean, ecsigma, ectrunc, ecmin, ecmax, ecdrifti, ecdriftf, drifttime)
    nulltime = 0.0
    esparams = (esnu, esphase, esmax, nulltime)

    dist = SNSESpreadDist(
        lattice.getLength(),
        zmin,
        zmax,
        tailfraction,
        sync_part,
        emean,
        esigma,
        etrunc,
        emin,
        emax,
        ecparams,
        esparams,
    )
    return dist
