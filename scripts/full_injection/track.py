"""Full SNS injection simulation.

This script uses hydra to run. Default parameters are defined in config.yaml
and can be changed from the command line. Examples:

    python track.py 
    python track.py lattice.foil=false
    python track.py lattice.rf=true
    python track.py spacecharge.xy.solver=2p5d
    ...

To run with MPI:

    mpirun -n 2 python track.py

Outputs are saved to /outputs/timestamp. The runtime parameters are saved to 
the /config folder within the outputs folder.
"""
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.kickernodes import SquareRootWaveform
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.teapot import TEAPOT_Ring
from orbit.utils.consts import mass_proton

from sns_ring_model import SNS_RING
from sns_ring_model.injection import make_dist_joho
from sns_ring_model.injection import make_dist_sns_espread

# local
from utils import get_intensity


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    sys.stdout.flush()
    
    # Setup
    # --------------------------------------------------------------------------------------
    
    mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    mpi_rank = orbit_mpi.MPI_Comm_rank(mpi_comm)
    mpi_size = orbit_mpi.MPI_Comm_size(mpi_comm)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    
    # Initialize model
    # --------------------------------------------------------------------------------------
    
    model = SNS_RING(
        lattice_file="inputs/sns_ring_mad.lattice",
        lattice_file_type="mad",
        lattice_seq="RINGINJ",
    )
    model.initialize()
    model.set_max_part_length(cfg.lattice.max_part_length)
    
    lattice = model.lattice
    
    
    # Initialize bunch
    # --------------------------------------------------------------------------------------
    
    intensity = get_intensity(
        power=(cfg.bunch.power * 1.00e06),
        energy=(cfg.bunch.energy * 1.00e09),
        frequency=cfg.bunch.frequency,
    )
    
    minipulses_per_pulse = 1044
    minipulse_intensity = intensity / minipulses_per_pulse
    macrosize = minipulse_intensity / cfg.macros_per_turn
    
    bunch = Bunch()
    bunch.mass(mass_proton)
    bunch.macroSize(macrosize)

    sync_part = bunch.getSyncParticle()
    sync_part.kinEnergy(cfg.bunch.energy)
    
    lostbunch = Bunch()
    lostbunch.addPartAttr("LostParticleAttributes")
    
    params_dict = {}
    params_dict["bunch"] = bunch
    params_dict["lostbunch"] = lostbunch
    
    model.set_bunch(bunch, lostbunch=lostbunch, params_dict=params_dict)

    
    # Set up injection kickers
    # --------------------------------------------------------------------------------------
    
    hkick10_node = lattice.getNodeForName("IKICKH_A10")
    vkick10_node = lattice.getNodeForName("IKICKV_A10")
    hkick11_node = lattice.getNodeForName("IKICKH_A11")
    vkick11_node = lattice.getNodeForName("IKICKV_A11")
    hkick12_node = lattice.getNodeForName("IKICKH_A12")
    vkick12_node = lattice.getNodeForName("IKICKV_A12")
    hkick13_node = lattice.getNodeForName("IKICKH_A13")
    vkick13_node = lattice.getNodeForName("IKICKV_A13")
    
    hkick10_node.setParam("kx", cfg.kick.strength.hkick10)
    vkick10_node.setParam("ky", cfg.kick.strength.vkick10)
    hkick11_node.setParam("kx", cfg.kick.strength.hkick11)
    vkick11_node.setParam("ky", cfg.kick.strength.vkick11)
    hkick12_node.setParam("kx", cfg.kick.strength.hkick12)
    vkick12_node.setParam("ky", cfg.kick.strength.vkick12)
    hkick13_node.setParam("kx", cfg.kick.strength.hkick13)
    vkick13_node.setParam("ky", cfg.kick.strength.vkick13)
    
    hkickerwave = SquareRootWaveform(
        sync_part,
        lattice.getLength(),
        cfg.kick.wave.tih,
        cfg.kick.wave.tf,
        cfg.kick.wave.si,
        cfg.kick.wave.sfh,
    )
    vkickerwave = SquareRootWaveform(
        sync_part,
        lattice.getLength(),
        cfg.kick.wave.tiv,
        cfg.kick.wave.tf,
        cfg.kick.wave.si,
        cfg.kick.wave.sfv,
    )
    hkick10_node.setWaveform(hkickerwave)
    vkick10_node.setWaveform(vkickerwave)
    hkick11_node.setWaveform(hkickerwave)
    vkick11_node.setWaveform(vkickerwave)
    hkick12_node.setWaveform(hkickerwave)
    vkick12_node.setWaveform(vkickerwave)
    hkick13_node.setWaveform(hkickerwave)
    vkick13_node.setWaveform(vkickerwave)
    
    
    # Add injection node
    # --------------------------------------------------------------------------------------
    
    inj_dist_x = make_dist_joho(centerpos=model.inj_x, centermom=0.0, **cfg.inj.x)
    inj_dist_y = make_dist_joho(centerpos=model.inj_y, centermom=0.0, **cfg.inj.y)
    inj_dist_z = make_dist_sns_espread(bunch=bunch, lattice=lattice, **cfg.inj.z)
    
    inj_node = model.add_injection_node(
        n_parts=cfg.macros_per_turn,
        dist_x=inj_dist_x,
        dist_y=inj_dist_y,
        dist_z=inj_dist_z,
        n_parts_max=None,
        parent_index=0,
    )
    
    
    # Add additional nodes
    # --------------------------------------------------------------------------------------
    
    if cfg.lattice.foil:
        model.add_foil_node(**cfg.foil)

    if cfg.lattice.rf:
        model.add_rf_cavity_nodes(**cfg.rf)

    if cfg.lattice.apertures:
        model.add_all_aperture_and_collimator_nodes()
    
    if cfg.lattice.impedance.z:
        model.add_longitudinal_impedance_node(**cfg.impedance.z)

    if cfg.lattice.impedance.xy:
        model.add_transverse_impedance_node(**cfg.impedance.xy)

    if cfg.lattice.max_part_length:
        model.set_max_part_length(cfg.lattice.max_part_length)

    if cfg.lattice.spacecharge.z:
        model.add_longitudinal_spacecharge_node(**cfg.spacecharge.z)

    if cfg.lattice.spacecharge.xy:
        model.add_transverse_spacecharge_nodes(**cfg.spacecharge.xy)
        
    
    # Track bunch
    # --------------------------------------------------------------------------------------
    
    start_time = time.time()
    
    for turn_index in range(cfg.turns):
        lattice.trackBunch(bunch, params_dict)
    
        # Diagnostics
        time_ellapsed = time.time() - start_time
        bunch_size = bunch.getSizeGlobal()
    
        # Compute moments
        twiss_calc = BunchTwissAnalysis()
        twiss_calc.computeBunchMoments(bunch, 2, 0, 0)
        x_rms = np.sqrt(twiss_calc.getCorrelation(0, 0)) * 1000.0
        y_rms = np.sqrt(twiss_calc.getCorrelation(2, 2)) * 1000.0
    
        # Print update message
        if mpi_rank == 0:
            message = ""
            message += "turn={:04.0f} ".format(turn_index + 1)
            message += "time={:0.3f} ".format(time_ellapsed)
            message += "size={:0.2e} ".format(bunch_size)
            message += "xrms={:0.2f} ".format(x_rms)
            message += "yrms={:0.2f} ".format(y_rms)
            print(message)
            sys.stdout.flush()
    
        # Write bunch to file
        if (turn_index % cfg.write_bunch_freq == 0) or (turn_index == cfg.turns - 1):
            filename = "bunch_{:04.0f}".format(turn_index)
            filename = os.path.join(output_dir, filename)
            if mpi_rank == 0:
                print(filename)
                sys.stdout.flush()
            bunch.dumpBunch(filename)


if __name__ == "__main__":
    main()
    