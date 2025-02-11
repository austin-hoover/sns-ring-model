import numpy as np

from orbit.core.bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

from sns_ring_model import SNS_RING
from sns_ring_model import InjectionController


def test_model():
    model = SNS_RING(lattice_file="default")
    model.add_rf_cavity_nodes()


def test_add_apertures():
    model = SNS_RING(
        lattice_file="sns_ring_model/mad/sns_ring_mad.lat", 
        lattice_file_type="mad", 
        lattice_seq="RINGINJ"
    )
    model.add_all_aperture_and_collimator_nodes()


def test_ric():
    model = SNS_RING(
        lattice_file="sns_ring_model/madx/sns_ring.lat", 
        lattice_file_type="madx", 
        lattice_seq="rnginj",
    )
    lattice = model.lattice

    bunch = Bunch()
    bunch.mass(0.938)
    bunch.getSyncParticle().kinEnergy(1.300)
    
    ric = InjectionController(
        lattice=lattice,
        bunch=bunch,
    )

    inj_coords_targ = np.zeros(4)
    inj_coords_targ[0] = model.inj_x
    inj_coords_targ[2] = model.inj_y
    ric.set_inj_coords(*inj_coords_targ)

    inj_coords_calc = ric.get_inj_coords()

    inj_coords_targ *= 1000.0
    inj_coords_calc *= 1000.0

    assert np.all(np.abs(inj_coords_targ - inj_coords_calc) < 0.01)

    
