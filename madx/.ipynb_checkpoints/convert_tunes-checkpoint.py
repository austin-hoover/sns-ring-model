"""PyORBIT script to synchronize MADX and PyORBIT tunes."""
import argparse
import fileinput
import os
import shutil
import subprocess
import sys
from pprint import pprint

from orbit.core.bunch import Bunch
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton



def run_madx(script_name: str, hide_output: bool = True) -> None:
    """Run madx script."""
    command = None
    if hide_output:
        command = "./madx {} > /dev/null 2>&1".format(script_name)
    else:
        command = "./madx {}".format(script_name)

    subprocess.call(command, shell=True)


def set_node_fringe(node: AccNode, setting: bool) -> AccNode:
    if hasattr(node, "setFringeFieldFunctionIN"):
        node.setUsageFringeFieldIN(setting)
    if hasattr(node, "setFringeFieldFunctionOUT"):
        node.setUsageFringeFieldIN(setting)
    return node


def calculate_tunes(lattice: AccLattice, kin_energy: float, mass: float) -> tuple[float, float]:
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    lattice_params = matrix_lattice.getRingParametersDict()
    nux = lattice_params["fractional tune x"]
    nuy = lattice_params["fractional tune y"]
    return (nux, nuy)


class TuneConverter:
    """Converts MADX tunes to equivalent PyORBIT tunes."""

    def __init__(
        self,
        madx_script_filename: str,
        lattice_filename: str,
        lattice_seq: str,
        mass: float,
        kin_energy: float,
        verbose: int = 1,
        fringe: bool = False,
    ) -> None:
        """Constructor."""
        self.madx_script_filename = madx_script_filename
        self.lattice_filename = lattice_filename
        self.lattice_seq = lattice_seq
        self.mass = mass
        self.kin_energy = kin_energy
        self.verbose = verbose
        self.fringe = fringe

    def set_madx_script_tunes(self, nux_madx: float, nuy_madx: float) -> None:
        """Replace tunes in madx script."""
        for line in fileinput.input([self.madx_script_filename], inplace=True):
            text = line.strip()
            if text.startswith("QH:="):
                line = "QH:={};\n".format(nux_madx)
            elif text.startswith("QV:="):
                line = "QV:={};\n".format(nuy_madx)
            sys.stdout.write(line)

    def set_output_lattice_filename(self, filename: str) -> None:
        """Set the output lattice file name in madx script."""
        self.lattice_filename = lattice_filename

        prefix = "SAVE,sequence=RNGINJ, FILE="
        new_line = "".join([prefix, "'{}',clear;\n".format(filename)])
        for line in fileinput.input([self.madx_script_filename], inplace=True):
            if line.strip().startswith(prefix):
                line = new_line
            sys.stdout.write(line)

    def run_madx(self) -> None:
        return run_madx(self.madx_script_filename, hide_output=(self.verbose < 2))

    def build_lattice(self) -> AccLattice:
        lattice = TEAPOT_Lattice()
        lattice.readMADX(self.lattice_filename, self.lattice_seq)
        lattice.initialize()

        for node in lattice.getNodes():
            set_node_fringe(node, self.fringe)

        return lattice

    def set_pyorbit_tunes(self, nux: float, nuy: float, max_iters: int = 1000, atol: float = 0.001,  rtol: float = 0.001) -> dict[str, float]:
        nux_madx = nux
        nuy_madx = nuy

        converged_x = False
        converged_y = False

        for iteration in range(max_iters):
            self.set_madx_script_tunes(nux_madx, nuy_madx)
            self.run_madx()

            lattice = self.build_lattice()
            nux_calc, nuy_calc = calculate_tunes(
                lattice, kin_energy=self.kin_energy, mass=self.mass
            )

            print("MADX tunes:    {}, {}".format(nux_madx, nuy_madx))
            print("PyORBIT tunes: {}, {}".format(nux_calc, nuy_calc))
            print()

            error_x = (nux % 1.0) - nux_calc
            error_y = (nuy % 1.0) - nuy_calc
            converged_x = abs(error_x) < atol or abs(error_x / nux) < rtol
            converged_y = abs(error_y) < atol or abs(error_y / nuy) < rtol
            if not converged_x:
                nux_madx += error_x
            if not converged_y:
                nuy_madx += error_y
            if converged_x and converged_y:
                return {
                    "nux_madx": nux_madx,
                    "nuy_madx": nuy_madx,
                    "error_x": error_x,
                    "error_y": error_y,
                    "converged_x": converged_x,
                    "converged_y": converged_y,
                    "nux_pyorbit": nux_calc,
                    "nuy_pyorbit": nuy_calc,
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nux", type=float, default=6.24)
    parser.add_argument("--nuy", type=float, default=6.16)
    parser.add_argument("--atol", type=float, default=0.0001)
    parser.add_argument("--rtol", type=float, default=0.0001)
    parser.add_argument("--mass", type=float, default=mass_proton)
    parser.add_argument("--energy", type=float, default=1.300)
    parser.add_argument("--fringe", type=int, default=0)
    args = parser.parse_args()

    madx_script_filename = "sns_ring.madx"
    lattice_filename = "lattice"
    lattice_seq = "rnginj"

    # Remove old output
    subprocess.call('rm ./_outputs/*', shell=True)

    # Find correct madx inputs
    tune_converter = TuneConverter(
        madx_script_filename=madx_script_filename,
        lattice_filename=lattice_filename,
        lattice_seq=lattice_seq,
        mass=args.mass,
        kin_energy=args.energy,
        fringe=args.fringe,
    )
    result = tune_converter.set_pyorbit_tunes(args.nux, args.nuy, atol=args.atol, rtol=args.rtol)
    pprint(result)

    # Save outputs
    nux = result["nux_pyorbit"] + 6.0
    nuy = result["nuy_pyorbit"] + 6.0

    output_dir = "outputs"
    output_dir = os.path.join(output_dir, f"sns_ring_nux-{nux:0.2f}_nuy-{nuy:0.2f}")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    filenames = [
        lattice_filename,
        "madx.ps",
        "optics",
        "optics_for_G4BL",
        "twiss",
    ]
    for filename in filenames:
        shutil.move(filename, output_dir)
