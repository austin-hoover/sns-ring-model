"""Full SNS injection simulation."""
import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from orbit.core import orbit_mpi
from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.spacecharge import Boundary2D
from orbit.core.spacecharge import LSpaceChargeCalc
from orbit.core.spacecharge import SpaceChargeCalc2p5D
from orbit.core.spacecharge import SpaceChargeCalcSliceBySlice2D
from orbit.aperture import TeapotApertureNode
from orbit.aperture import CircleApertureNode
from orbit.aperture import RectangleApertureNode
from orbit.aperture import EllipseApertureNode
from orbit.bumps import TeapotBumpNode
from orbit.bumps import TDTeapotSimpleBumpNode
from orbit.collimation import TeapotCollimatorNode
from orbit.collimation import addTeapotCollimatorNode
from orbit.foils import TeapotFoilNode
from orbit.impedances import addImpedanceNode
from orbit.impedances import LImpedance_Node
from orbit.impedances import TImpedance_Node
from orbit.injection import TeapotInjectionNode
from orbit.injection import JohoTransverse
from orbit.injection import JohoLongitudinal
from orbit.injection import SNSESpreadDist
from orbit.kickernodes import flatTopWaveform
from orbit.kickernodes import SquareRootWaveform
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.rf_cavities import RFNode
from orbit.rf_cavities import addRFNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc2p5d import SC2p5D_AccNode
from orbit.space_charge.sc2p5d import setSC2p5DAccNodes
from orbit.space_charge.sc2dslicebyslice import SC2DSliceBySlice_AccNode
from orbit.space_charge.sc2dslicebyslice import setSC2DSliceBySliceAccNodes
from orbit.teapot import TEAPOT_Ring
from orbit.teapot import DriftTEAPOT
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light

# local
from utils import read_transverse_impedance_file
from utils import read_longitudinal_impedance_file


# Parse arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--apertures", type=int, default=0)
parser.add_argument("--imp-xy", type=int, default=0)
parser.add_argument("--imp-xy-bins", type=int, default=64)
parser.add_argument("--imp-xy-macros-min", type=int, default=1000)
parser.add_argument("--imp-z", type=int, default=0)
parser.add_argument("--imp-z-bins", type=int, default=128)
parser.add_argument("--imp-z-macros-min", type=int, default=1000)
parser.add_argument("--sc-xy", type=int, default=0)
parser.add_argument("--sc-xy-macros-min", type=int, default=1000)
parser.add_argument("--sc-xy-solver", type=str, default="slicebyslice", choices=["slicebyslice", "2p5d"])
parser.add_argument("--sc-xy-gridx", type=int, default=128)
parser.add_argument("--sc-xy-gridy", type=int, default=128)
parser.add_argument("--sc-xy-gridz", type=int, default=64)
parser.add_argument("--sc-xy-boundary", type=int, default=1)
parser.add_argument("--sc-xy-boundary-modes", type=int, default=32)
parser.add_argument("--sc-xy-boundary-points", type=int, default=128)
parser.add_argument("--sc-z", type=int, default=0)
parser.add_argument("--sc-z-bins", type=int, default=64)
parser.add_argument("--sc-z-macros-min", type=int, default=1000)
parser.add_argument("--macros-per-turn", type=int, default=100, help="max 2000")
parser.add_argument("--turns", type=int, default=1044)
parser.add_argument("--write-bunch-freq", type=int, default=250)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

timestamp = time.strftime("%y%m%d%H%M%S")

output_dir = os.path.join("outputs", timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Initialize bunch
# --------------------------------------------------------------------------------------

minipulse_intensity = 2.163e+11
macrosize = minipulse_intensity / args.macros_per_turn

bunch = Bunch()
bunch.mass(mass_proton)
bunch.macroSize(macrosize)
bunch.getSyncParticle().kinEnergy(1.3)  # [GeV]

lostbunch = Bunch()
lostbunch.addPartAttr("LostParticleAttributes")

params_dict = {}
params_dict["bunch"]= bunch
params_dict["lostbunch"] = lostbunch


# Initialize lattice
# --------------------------------------------------------------------------------------

lattice = TEAPOT_Ring()
lattice.readMAD("inputs/sns_ring_mad.lattice", "RINGINJ")
lattice.initialize()

nodes = lattice.getNodes()

for node in nodes:
    print(node.getName(), node)


# Set injection kicker waveforms
# --------------------------------------------------------------------------------------

hkick10_node = nodes[645]
vkick10_node = nodes[647]
hkick11_node = nodes[649]
vkick11_node = nodes[651]
hkick12_node = nodes[18]
vkick12_node = nodes[16]
hkick13_node = nodes[22]
vkick13_node = nodes[20]

strength_hkicker10 = 14.04e-03
strength_vkicker10 =  8.84e-03
strength_hkicker11 = -4.28e-03
strength_vkicker11 = -5.06e-03
strength_hkicker12 = -4.36727974875e-03
strength_vkicker12 = -5.32217284098e-03
strength_hkicker13 = 14.092989681e-03
strength_vkicker13 =  9.0098984536e-03

tih = -0.001
tiv = -0.002
tf =  0.001
si =  1.0
sfh =  0.457
sfv =  0.406

sync_part = bunch.getSyncParticle()
hkickerwave = SquareRootWaveform(sync_part, lattice.getLength(), tih, tf, si, sfh)
vkickerwave = SquareRootWaveform(sync_part, lattice.getLength(), tiv, tf, si, sfv)

hkick10_node.setParam("kx", strength_hkicker10)
vkick10_node.setParam("ky", strength_vkicker10)
hkick11_node.setParam("kx", strength_hkicker11)
vkick11_node.setParam("ky", strength_vkicker11)
hkick12_node.setParam("kx", strength_hkicker12)
vkick12_node.setParam("ky", strength_vkicker12)
hkick13_node.setParam("kx", strength_hkicker13)
vkick13_node.setParam("ky", strength_vkicker13)

hkick10_node.setWaveform(hkickerwave)
vkick10_node.setWaveform(vkickerwave)
hkick11_node.setWaveform(hkickerwave)
vkick11_node.setWaveform(vkickerwave)
hkick12_node.setWaveform(hkickerwave)
vkick12_node.setWaveform(vkickerwave)
hkick13_node.setWaveform(hkickerwave)
vkick13_node.setWaveform(vkickerwave)


# Define minipulse distribution function
# --------------------------------------------------------------------------------------

lattice_length = lattice.getLength()
sync_part = bunch.getSyncParticle()

order = 9.0
alphax = 0.064
betax = 10.056
alphay = 0.063
betay = 10.815
emitlim = 0.221 * 2 * (order + 1) * 1.0e-6
xcenterpos = 0.0486
xcentermom = 0.0
ycenterpos = 0.046
ycentermom = 0.0

zlim = 139.68 * lattice_length / 360.0
zmin = -zlim
zmax = zlim
tailfraction = 0.0
emean = sync_part.kinEnergy()
esigma = 0.0005
etrunc = 1.0
emin = sync_part.kinEnergy() - 0.0025
emax = sync_part.kinEnergy() + 0.0025
ecmean = 0.0
ecsigma = 0.000000001
ectrunc = 1.0
ecmin = -0.0035
ecmax = 0.0035
ecdrifti = 0.0
ecdriftf = 0.0
time_per_turn = lattice_length / (sync_part.beta() * speed_of_light)
drifttime = 1000.0 * args.turns * time_per_turn
ecparams = (ecmean, ecsigma, ectrunc, ecmin, ecmax, ecdrifti, ecdriftf, drifttime)
esnu = 100.0
esphase = 0.0
esmax = 0.0
nulltime = 0.0
esparams = (esnu, esphase, esmax, nulltime)

inj_dist_x = JohoTransverse(order, alphax, betax, emitlim, xcenterpos, xcentermom)
inj_dist_y = JohoTransverse(order, alphay, betay, emitlim, ycenterpos, ycentermom)
inj_dist_z = SNSESpreadDist(lattice_length, zmin, zmax, tailfraction, sync_part, emean, esigma, etrunc, emin, emax, ecparams, esparams)


# Add injection and foil nodes
# --------------------------------------------------------------------------------------

foil_xmin = xcenterpos - 0.005
foil_xmax = xcenterpos + 0.014722
foil_ymin = ycenterpos - 0.005
foil_ymax = ycenterpos + 0.020
foil_thickness = 390.0
foil_node = TeapotFoilNode(foil_xmin, foil_xmax, foil_ymin, foil_ymax, foil_thickness)
foil_node.setScatterChoice(0)

inj_params = (foil_xmin, foil_xmax, foil_ymin, foil_ymax)
inj_node = TeapotInjectionNode(args.macros_per_turn, bunch, lostbunch, inj_params, inj_dist_x, inj_dist_y, inj_dist_z)

start_node = nodes[0]
start_node.addChildNode(inj_node, AccNode.ENTRANCE)
start_node.addChildNode(foil_node, AccNode.ENTRANCE)


# Add apertures and displacements for injection chicane magnets
# --------------------------------------------------------------------------------------

xcenter = 0.100
xb10m   = 0.0
xb11m   = 0.09677
xb12m   = 0.08899
xb13m   = 0.08484

ycenter = 0.023

bumpwave = flatTopWaveform(1.0)

xb10i   = 0.0
apb10xi = -xb10i
apb10yi = ycenter
rb10i   = 0.1095375
appb10i = CircleApertureNode(rb10i, 244.812, apb10xi, apb10yi, name = "b10i")

xb10f   = 0.022683
apb10xf = -xb10f
apb10yf = ycenter
rb10f   = 0.1095375
appb10f = CircleApertureNode(rb10f, 245.893, apb10xf, apb10yf, name = "b10f")

mag10x  = (xb10i + xb10f) / 2.0 - xb10m
cdb10i  = TDTeapotSimpleBumpNode(bunch,  mag10x, 0.0, -ycenter, 0.0, bumpwave, "mag10bumpi")
cdb10f  = TDTeapotSimpleBumpNode(bunch, -mag10x, 0.0,  ycenter, 0.0, bumpwave, "mag10bumpf")

xb11i   = 0.074468
apb11xi = xcenter - xb11i
apb11yi = ycenter
rb11i   = 0.1095375
appb11i = CircleApertureNode(rb11i, 247.265, apb11xi, apb11yi, name = "b11i")

xfoil   = 0.099655
apfoilx = xcenter - xfoil
apfoily = ycenter
rfoil   = 0.1095375
appfoil = CircleApertureNode(rfoil, 248.009, apfoilx, apfoily, name = "bfoil")

mag11ax = (xb11i + xfoil) / 2.0 - xb11m
cdb11i  = TDTeapotSimpleBumpNode(bunch,  mag11ax, 0.0, -ycenter, 0.0, bumpwave, "mag11bumpi")
cdfoila = TDTeapotSimpleBumpNode(bunch, -mag11ax, 0.0,  ycenter, 0.0, bumpwave, "foilbumpa")

xb11f   = 0.098699
apb11xf = xcenter - xb11f
apb11yf = ycenter
rb11f   = 0.1095375
appb11f = CircleApertureNode(rb11f, 0.195877, apb11xf, apb11yf, name = "b11f")

mag11bx = (xfoil + xb11f) / 2.0 - xb11m
cdfoilb = TDTeapotSimpleBumpNode(bunch,  mag11bx, 0.0, -ycenter, 0.0, bumpwave, "foilbumpb")
cdb11f  = TDTeapotSimpleBumpNode(bunch, -mag11bx, 0.0,  ycenter, 0.0, bumpwave, "mag11bumpf")

xb12i   = 0.093551
apb12xi = xcenter - xb12i
apb12yi = ycenter
rb12i   = 0.1095375
appb12i = CircleApertureNode(rb12i, 1.08593, apb12xi, apb12yi, name = "b12i")

xb12f   = 0.05318
apb12xf = xcenter - xb12f
apb12yf = ycenter
rb12f   = 0.1174750
appb12f = CircleApertureNode(rb12f, 1.99425, apb12xf, apb12yf, name = "b12f")

mag12x  = (xb12i + xb12f) / 2.0 - xb12m
cdb12i  = TDTeapotSimpleBumpNode(bunch,  mag12x, 0.0, -ycenter, 0.0, bumpwave, "mag12bumpi")
cdb12f  = TDTeapotSimpleBumpNode(bunch, -mag12x, 0.0,  ycenter, 0.0, bumpwave, "mag12bumpf")

xb13i   = 0.020774
apb13xi = xcenter - xb13i
apb13yi = ycenter
h13xi   =  0.1913
v13xi   =  0.1016
appb13i = RectangleApertureNode(h13xi, v13xi, 3.11512, apb13xi, apb13yi, name = "b13i")

xb13f   = 0.0
apb13xf = xcenter - xb13f
apb13yf = ycenter
h13xf   =  0.1913
v13xf   =  0.1016
appb13f = RectangleApertureNode(h13xf, v13xf, 4.02536, apb13xf, apb13yf, name = "b13f")

mag13x  = (xb13i + xb13f) / 2.0 - xb13m
cdb13i  = TDTeapotSimpleBumpNode(bunch,  mag13x, 0.0, -ycenter, 0.0, bumpwave, "mag13bumpi")
cdb13f  = TDTeapotSimpleBumpNode(bunch, -mag13x, 0.0,  ycenter, 0.0, bumpwave, "mag13bumpf")

nodes = lattice.getNodes()
dha10   = nodes[663]
dha11a  = nodes[665]
dha11b  = nodes[0]
dha12   = nodes[2]
dha13   = nodes[4]

if args.apertures:
    dha10.addChildNode(appb10i, AccNode.ENTRANCE)
    dha10.addChildNode(cdb10i, AccNode.ENTRANCE)
    dha10.addChildNode(cdb10f, AccNode.EXIT)
    dha10.addChildNode(appb10f, AccNode.EXIT)
    dha11a.addChildNode(appb11i, AccNode.ENTRANCE)
    dha11a.addChildNode(cdb11i , AccNode.ENTRANCE)
    dha11a.addChildNode(cdfoila, AccNode.EXIT)
    dha11a.addChildNode(appfoil, AccNode.EXIT)
    dha11b.addChildNode(cdfoilb, AccNode.ENTRANCE)
    dha11b.addChildNode(cdb11f , AccNode.EXIT)
    dha11b.addChildNode(appb11f, AccNode.EXIT)
    dha12.addChildNode(appb12i, AccNode.ENTRANCE)
    dha12.addChildNode(cdb12i, AccNode.ENTRANCE)
    dha12.addChildNode(cdb12f, AccNode.EXIT)
    dha12.addChildNode(appb12f, AccNode.EXIT)
    dha13.addChildNode(appb13i, AccNode.ENTRANCE)
    dha13.addChildNode(cdb13i, AccNode.ENTRANCE)
    dha13.addChildNode(cdb13f, AccNode.EXIT)
    dha13.addChildNode(appb13f, AccNode.EXIT)


# Add aperture and collimator nodes around the ring
# --------------------------------------------------------------------------------------

a115p8 = 0.1158
b078p7 = 0.0787
a080p0 = 0.0800
b048p0 = 0.0480

r062 = 0.0625
r100 = 0.1000
r120 = 0.1200
r125 = 0.1250
r130 = 0.1300
r140 = 0.1400

app06200 = CircleApertureNode(r062,  58.21790, 0.0, 0.0, name = "s1")
app06201 = CircleApertureNode(r062,  65.87790, 0.0, 0.0, name = "s2")

app10000 = CircleApertureNode(r100,  10.54740, 0.0, 0.0, name = "bp100")
app10001 = CircleApertureNode(r100,  10.97540, 0.0, 0.0, name = "bp100")
app10002 = CircleApertureNode(r100,  13.57190, 0.0, 0.0, name = "bp100")
app10003 = CircleApertureNode(r100,  15.39140, 0.0, 0.0, name = "21cmquad")
app10004 = CircleApertureNode(r100,  19.39150, 0.0, 0.0, name = "21cmquad")
app10005 = CircleApertureNode(r100,  23.39170, 0.0, 0.0, name = "21cmquad")
app10006 = CircleApertureNode(r100,  31.39210, 0.0, 0.0, name = "21cmquad")
app10007 = CircleApertureNode(r100,  39.39250, 0.0, 0.0, name = "21cmquad")
app10008 = CircleApertureNode(r100,  43.39270, 0.0, 0.0, name = "21cmquad")
app10009 = CircleApertureNode(r100,  47.39290, 0.0, 0.0, name = "21cmquad")
app10010 = CircleApertureNode(r100,  48.86630, 0.0, 0.0, name = "bp100")
app10011 = CircleApertureNode(r100,  50.37710, 0.0, 0.0, name = "P1a")
app10012 = CircleApertureNode(r100,  51.19660, 0.0, 0.0, name = "scr1c")
app10013 = CircleApertureNode(r100,  51.24100, 0.0, 0.0, name = "scr2c")
app10014 = CircleApertureNode(r100,  51.39470, 0.0, 0.0, name = "scr3c")
app10015 = CircleApertureNode(r100,  51.43910, 0.0, 0.0, name = "scr4c")
app10016 = CircleApertureNode(r100,  51.62280, 0.0, 0.0, name = "p2shield")
app10017 = CircleApertureNode(r100,  53.74280, 0.0, 0.0, name = "p2shield")
app10018 = CircleApertureNode(r100,  54.75640, 0.0, 0.0, name = "bp100")
app10019 = CircleApertureNode(r100,  71.16320, 0.0, 0.0, name = "bp100")
app10020 = CircleApertureNode(r100,  73.35170, 0.0, 0.0, name = "bp100")
app10021 = CircleApertureNode(r100,  75.60170, 0.0, 0.0, name = "bp100")
app10022 = CircleApertureNode(r100,  76.79720, 0.0, 0.0, name = "bp100")
app10023 = CircleApertureNode(r100,  77.39290, 0.0, 0.0, name = "21cmquad")
app10024 = CircleApertureNode(r100,  81.39310, 0.0, 0.0, name = "21cmquad")
app10025 = CircleApertureNode(r100,  85.39330, 0.0, 0.0, name = "21cmquad")
app10026 = CircleApertureNode(r100,  93.39370, 0.0, 0.0, name = "21cmquad")
app10027 = CircleApertureNode(r100, 101.39400, 0.0, 0.0, name = "21cmquad")
app10028 = CircleApertureNode(r100, 105.39400, 0.0, 0.0, name = "21cmquad")
app10029 = CircleApertureNode(r100, 109.39400, 0.0, 0.0, name = "21cmquad")
app10030 = CircleApertureNode(r100, 110.49000, 0.0, 0.0, name = "bp100")
app10031 = CircleApertureNode(r100, 112.69100, 0.0, 0.0, name = "bp100")
app10032 = CircleApertureNode(r100, 114.82200, 0.0, 0.0, name = "bp100")
app10033 = CircleApertureNode(r100, 118.38600, 0.0, 0.0, name = "bp100")
app10034 = CircleApertureNode(r100, 120.37900, 0.0, 0.0, name = "bp100")
app10035 = CircleApertureNode(r100, 122.21700, 0.0, 0.0, name = "bp100")
app10036 = CircleApertureNode(r100, 124.64400, 0.0, 0.0, name = "bp100")
app10037 = CircleApertureNode(r100, 127.77400, 0.0, 0.0, name = "bp100")
app10038 = CircleApertureNode(r100, 132.53100, 0.0, 0.0, name = "bp100")
app10039 = CircleApertureNode(r100, 136.10400, 0.0, 0.0, name = "bp100")
app10040 = CircleApertureNode(r100, 138.79900, 0.0, 0.0, name = "bp100")
app10041 = CircleApertureNode(r100, 139.39400, 0.0, 0.0, name = "21cmquad")
app10042 = CircleApertureNode(r100, 143.39500, 0.0, 0.0, name = "21cmquad")
app10043 = CircleApertureNode(r100, 147.39500, 0.0, 0.0, name = "21cmquad")
app10044 = CircleApertureNode(r100, 155.39500, 0.0, 0.0, name = "21cmquad")
app10045 = CircleApertureNode(r100, 163.39600, 0.0, 0.0, name = "21cmquad")
app10046 = CircleApertureNode(r100, 167.39600, 0.0, 0.0, name = "21cmquad")
app10047 = CircleApertureNode(r100, 171.39600, 0.0, 0.0, name = "21cmquad")
app10048 = CircleApertureNode(r100, 173.70900, 0.0, 0.0, name = "bp100")
app10049 = CircleApertureNode(r100, 175.93900, 0.0, 0.0, name = "bp100")
app10050 = CircleApertureNode(r100, 180.38700, 0.0, 0.0, name = "bp100")
app10051 = CircleApertureNode(r100, 182.12700, 0.0, 0.0, name = "bp100")
app10052 = CircleApertureNode(r100, 184.27300, 0.0, 0.0, name = "bp100")
app10053 = CircleApertureNode(r100, 186.57100, 0.0, 0.0, name = "bp100")
app10054 = CircleApertureNode(r100, 188.86800, 0.0, 0.0, name = "bp100")
app10055 = CircleApertureNode(r100, 191.16500, 0.0, 0.0, name = "bp100")
app10056 = CircleApertureNode(r100, 194.53200, 0.0, 0.0, name = "bp100")
app10057 = CircleApertureNode(r100, 196.61400, 0.0, 0.0, name = "bp100")
app10058 = CircleApertureNode(r100, 199.47500, 0.0, 0.0, name = "bp100")
app10059 = CircleApertureNode(r100, 201.39600, 0.0, 0.0, name = "21cmquad")
app10060 = CircleApertureNode(r100, 205.39600, 0.0, 0.0, name = "21cmquad")
app10061 = CircleApertureNode(r100, 209.39600, 0.0, 0.0, name = "21cmquad")
app10062 = CircleApertureNode(r100, 217.39700, 0.0, 0.0, name = "21cmquad")
app10063 = CircleApertureNode(r100, 225.39700, 0.0, 0.0, name = "21cmquad")
app10064 = CircleApertureNode(r100, 229.39700, 0.0, 0.0, name = "21cmquad")
app10065 = CircleApertureNode(r100, 233.39700, 0.0, 0.0, name = "21cmquad")
app10066 = CircleApertureNode(r100, 234.87800, 0.0, 0.0, name = "bp100")
app10067 = CircleApertureNode(r100, 236.87700, 0.0, 0.0, name = "bp100")
app10068 = CircleApertureNode(r100, 238.74100, 0.0, 0.0, name = "bp100")
app10069 = CircleApertureNode(r100, 242.38900, 0.0, 0.0, name = "bp100")

app12000 = CircleApertureNode(r120,   6.89986, 0.0, 0.0, name = "bp120")
app12001 = CircleApertureNode(r120,   8.52786, 0.0, 0.0, name = "bp120")
app12002 = CircleApertureNode(r120,  57.20790, 0.0, 0.0, name = "s1shield")
app12003 = CircleApertureNode(r120,  59.40790, 0.0, 0.0, name = "s1shield")
app12004 = CircleApertureNode(r120,  64.86790, 0.0, 0.0, name = "s2shield")
app12005 = CircleApertureNode(r120,  67.06790, 0.0, 0.0, name = "s2shield")
app12006 = CircleApertureNode(r120, 116.75800, 0.0, 0.0, name = "bp120")
app12007 = CircleApertureNode(r120, 130.90300, 0.0, 0.0, name = "bp120")
app12008 = CircleApertureNode(r120, 178.75900, 0.0, 0.0, name = "bp120")
app12009 = CircleApertureNode(r120, 192.90400, 0.0, 0.0, name = "bp120")
app12010 = CircleApertureNode(r120, 240.76100, 0.0, 0.0, name = "bp120")

app12500 = CircleApertureNode(r125,  27.37140, 0.0, 0.0, name = "26cmquad")
app12501 = CircleApertureNode(r125,  35.37180, 0.0, 0.0, name = "26cmquad")
app12502 = CircleApertureNode(r125,  89.37300, 0.0, 0.0, name = "26cmquad")
app12503 = CircleApertureNode(r125,  97.37330, 0.0, 0.0, name = "26cmquad")
app12504 = CircleApertureNode(r125, 151.37400, 0.0, 0.0, name = "26cmquad")
app12505 = CircleApertureNode(r125, 159.37500, 0.0, 0.0, name = "26cmquad")
app12506 = CircleApertureNode(r125, 213.37600, 0.0, 0.0, name = "26cmquad")
app12507 = CircleApertureNode(r125, 221.37600, 0.0, 0.0, name = "26cmquad")

app13000 = CircleApertureNode(r130,  60.41790, 0.0, 0.0, name = "bp130")
app13001 = CircleApertureNode(r130,  64.42290, 0.0, 0.0, name = "bp130")
app13002 = CircleApertureNode(r130,  68.07790, 0.0, 0.0, name = "bp130")
app13003 = CircleApertureNode(r130,  68.90140, 0.0, 0.0, name = "bp130")
app13004 = CircleApertureNode(r130,  70.52940, 0.0, 0.0, name = "bp130")

app14000 = CircleApertureNode(r140,   7.43286, 0.0, 0.0, name = "30cmquad")
app14001 = CircleApertureNode(r140,   7.85486, 0.0, 0.0, name = "30cmquad")
app14002 = CircleApertureNode(r140,  55.42940, 0.0, 0.0, name = "30cmquad")
app14003 = CircleApertureNode(r140,  55.85140, 0.0, 0.0, name = "30cmquad")
app14004 = CircleApertureNode(r140,  56.38440, 0.0, 0.0, name = "30cmquad")
app14005 = CircleApertureNode(r140,  60.86290, 0.0, 0.0, name = "bp140")
app14006 = CircleApertureNode(r140,  62.64290, 0.0, 0.0, name = "bp140")
app14007 = CircleApertureNode(r140,  63.97790, 0.0, 0.0, name = "bp140")
app14008 = CircleApertureNode(r140,  69.43440, 0.0, 0.0, name = "30cmquad")
app14009 = CircleApertureNode(r140,  69.85640, 0.0, 0.0, name = "30cmquad")
app14010 = CircleApertureNode(r140, 117.43100, 0.0, 0.0, name = "30cmquad")
app14011 = CircleApertureNode(r140, 117.85300, 0.0, 0.0, name = "30cmquad")
app14012 = CircleApertureNode(r140, 131.43600, 0.0, 0.0, name = "30cmquad")
app14013 = CircleApertureNode(r140, 131.85800, 0.0, 0.0, name = "30cmquad")
app14014 = CircleApertureNode(r140, 179.43200, 0.0, 0.0, name = "30cmquad")
app14015 = CircleApertureNode(r140, 179.85400, 0.0, 0.0, name = "30cmquad")
app14016 = CircleApertureNode(r140, 193.43700, 0.0, 0.0, name = "30cmquad")
app14017 = CircleApertureNode(r140, 193.85900, 0.0, 0.0, name = "30cmquad")
app14018 = CircleApertureNode(r140, 241.43400, 0.0, 0.0, name = "30cmquad")
app14019 = CircleApertureNode(r140, 241.85600, 0.0, 0.0, name = "30cmquad")

appell00 = EllipseApertureNode(a115p8, b078p7,  16.9211, 0.0, 0.0, name = "arcdipole")
appell01 = EllipseApertureNode(a115p8, b078p7,  20.9213, 0.0, 0.0, name = "arcdipole")
appell02 = EllipseApertureNode(a115p8, b078p7,  24.9215, 0.0, 0.0, name = "arcdipole")
appell03 = EllipseApertureNode(a115p8, b078p7,  28.9217, 0.0, 0.0, name = "arcdipole")
appell04 = EllipseApertureNode(a115p8, b078p7,  32.9219, 0.0, 0.0, name = "arcdipole")
appell05 = EllipseApertureNode(a115p8, b078p7,  36.9221, 0.0, 0.0, name = "arcdipole")
appell06 = EllipseApertureNode(a115p8, b078p7,  40.9222, 0.0, 0.0, name = "arcdipole")
appell07 = EllipseApertureNode(a115p8, b078p7,  44.9224, 0.0, 0.0, name = "arcdipole")
appell08 = EllipseApertureNode(a080p0, b048p0,  51.9428, 0.0, 0.0, name = "p2")
appell09 = EllipseApertureNode(a115p8, b078p7,  78.9226, 0.0, 0.0, name = "arcdipole")
appell10 = EllipseApertureNode(a115p8, b078p7,  82.9228, 0.0, 0.0, name = "arcdipole")
appell11 = EllipseApertureNode(a115p8, b078p7,  86.9230, 0.0, 0.0, name = "arcdipole")
appell12 = EllipseApertureNode(a115p8, b078p7,  90.9232, 0.0, 0.0, name = "arcdipole")
appell13 = EllipseApertureNode(a115p8, b078p7,  94.9234, 0.0, 0.0, name = "arcdipole")
appell14 = EllipseApertureNode(a115p8, b078p7,  98.9236, 0.0, 0.0, name = "arcdipole")
appell15 = EllipseApertureNode(a115p8, b078p7, 102.9240, 0.0, 0.0, name = "arcdipole")
appell16 = EllipseApertureNode(a115p8, b078p7, 106.9240, 0.0, 0.0, name = "arcdipole")
appell17 = EllipseApertureNode(a115p8, b078p7, 140.9240, 0.0, 0.0, name = "arcdipole")
appell18 = EllipseApertureNode(a115p8, b078p7, 144.9240, 0.0, 0.0, name = "arcdipole")
appell19 = EllipseApertureNode(a115p8, b078p7, 148.9250, 0.0, 0.0, name = "arcdipole")
appell20 = EllipseApertureNode(a115p8, b078p7, 152.9250, 0.0, 0.0, name = "arcdipole")
appell21 = EllipseApertureNode(a115p8, b078p7, 156.9250, 0.0, 0.0, name = "arcdipole")
appell22 = EllipseApertureNode(a115p8, b078p7, 160.9250, 0.0, 0.0, name = "arcdipole")
appell23 = EllipseApertureNode(a115p8, b078p7, 164.9250, 0.0, 0.0, name = "arcdipole")
appell24 = EllipseApertureNode(a115p8, b078p7, 168.9260, 0.0, 0.0, name = "arcdipole")
appell25 = EllipseApertureNode(a115p8, b078p7, 202.9260, 0.0, 0.0, name = "arcdipole")
appell26 = EllipseApertureNode(a115p8, b078p7, 206.9260, 0.0, 0.0, name = "arcdipole")
appell27 = EllipseApertureNode(a115p8, b078p7, 210.9260, 0.0, 0.0, name = "arcdipole")
appell28 = EllipseApertureNode(a115p8, b078p7, 214.9260, 0.0, 0.0, name = "arcdipole")
appell29 = EllipseApertureNode(a115p8, b078p7, 218.9260, 0.0, 0.0, name = "arcdipole")
appell30 = EllipseApertureNode(a115p8, b078p7, 222.9270, 0.0, 0.0, name = "arcdipole")
appell31 = EllipseApertureNode(a115p8, b078p7, 226.9270, 0.0, 0.0, name = "arcdipole")
appell32 = EllipseApertureNode(a115p8, b078p7, 230.9270, 0.0, 0.0, name = "arcdipole")

p1    = TeapotCollimatorNode(0.60000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0,   0.0, 50.3771)
scr1t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0,   0.0, 51.1921)
scr1c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0,   0.0, 51.1966)
scr2t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  90.0, 51.2365)
scr2c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  90.0, 51.2410)
scr3t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  45.0, 51.3902)
scr3c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  45.0, 51.3947)
scr4t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4346)
scr4c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4391)
p2sh1 = TeapotCollimatorNode(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0,   0.0, 51.6228)
p2    = TeapotCollimatorNode(1.80000, 3, 0.65, 2, 0.080, 0.048, 0.0, 0.0,   0.0, 51.9428)
p2sh2 = TeapotCollimatorNode(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0,   0.0, 53.7428)
s1sh1 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 57.2079)
s1    = TeapotCollimatorNode(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 58.2179)
s1sh2 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 59.4079)
s2sh1 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 64.8679)
s2    = TeapotCollimatorNode(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 65.8779)
s2sh2 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 67.0679)

nodes = lattice.getNodes()

ap06200 = nodes[173]
ap06201 = nodes[186]

ap10000 = nodes[15]
ap10001 = nodes[16]
ap10002 = nodes[21]
ap10003 = nodes[31]
ap10004 = nodes[45]
ap10005 = nodes[58]
ap10006 = nodes[78]
ap10007 = nodes[103]
ap10008 = nodes[116]
ap10009 = nodes[130]
ap10010 = nodes[140]
ap10011 = nodes[144]
ap10012 = nodes[147]
ap10013 = nodes[150]
ap10014 = nodes[153]
ap10015 = nodes[156]
ap10016 = nodes[158]
ap10017 = nodes[160]
ap10018 = nodes[168]
ap10019 = nodes[198]
ap10020 = nodes[199]
ap10021 = nodes[202]
ap10022 = nodes[203]
ap10023 = nodes[211]
ap10024 = nodes[225]
ap10025 = nodes[238]
ap10026 = nodes[258]
ap10027 = nodes[283]
ap10028 = nodes[296]
ap10029 = nodes[310]
ap10030 = nodes[319]
ap10031 = nodes[322]
ap10032 = nodes[330]
ap10033 = nodes[344]
ap10034 = nodes[351]
ap10035 = nodes[358]
ap10036 = nodes[359]
ap10037 = nodes[360]
ap10038 = nodes[364]
ap10039 = nodes[371]
ap10040 = nodes[372]
ap10041 = nodes[380]
ap10042 = nodes[394]
ap10043 = nodes[407]
ap10044 = nodes[427]
ap10045 = nodes[452]
ap10046 = nodes[465]
ap10047 = nodes[479]
ap10048 = nodes[489]
ap10049 = nodes[491]
ap10050 = nodes[502]
ap10051 = nodes[503]
ap10052 = nodes[504]
ap10053 = nodes[506]
ap10054 = nodes[508]
ap10055 = nodes[510]
ap10056 = nodes[514]
ap10057 = nodes[521]
ap10058 = nodes[524]
ap10059 = nodes[535]
ap10060 = nodes[549]
ap10061 = nodes[562]
ap10062 = nodes[582]
ap10063 = nodes[607]
ap10064 = nodes[620]
ap10065 = nodes[634]
ap10066 = nodes[644]
ap10067 = nodes[647]
ap10068 = nodes[651]
ap10069 = nodes[661]

ap12000 = nodes[5]
ap12001 = nodes[8]
ap12002 = nodes[172]
ap12003 = nodes[174]
ap12004 = nodes[185]
ap12005 = nodes[187]
ap12006 = nodes[341]
ap12007 = nodes[361]
ap12008 = nodes[498]
ap12009 = nodes[511]
ap12010 = nodes[658]

ap12500 = nodes[70]
ap12501 = nodes[91]
ap12502 = nodes[250]
ap12503 = nodes[271]
ap12504 = nodes[419]
ap12505 = nodes[440]
ap12506 = nodes[574]
ap12507 = nodes[595]

ap13000 = nodes[175]
ap13001 = nodes[184]
ap13002 = nodes[188]
ap13003 = nodes[189]
ap13004 = nodes[192]

ap14000 = nodes[6]
ap14001 = nodes[7]
ap14002 = nodes[169]
ap14003 = nodes[170]
ap14004 = nodes[171]
ap14005 = nodes[176]
ap14006 = nodes[180]
ap14007 = nodes[183]
ap14008 = nodes[190]
ap14009 = nodes[191]
ap14010 = nodes[342]
ap14011 = nodes[343]
ap14012 = nodes[362]
ap14013 = nodes[363]
ap14014 = nodes[500]
ap14015 = nodes[501]
ap14016 = nodes[512]
ap14017 = nodes[513]
ap14018 = nodes[659]
ap14019 = nodes[660]

apell00 = nodes[35]
apell01 = nodes[49]
apell02 = nodes[62]
apell03 = nodes[74]
apell04 = nodes[87]
apell05 = nodes[99]
apell06 = nodes[112]
apell07 = nodes[126]
apell08 = nodes[159]
apell09 = nodes[215]
apell10 = nodes[229]
apell11 = nodes[242]
apell12 = nodes[254]
apell13 = nodes[267]
apell14 = nodes[279]
apell15 = nodes[292]
apell16 = nodes[306]
apell17 = nodes[384]
apell18 = nodes[398]
apell19 = nodes[411]
apell20 = nodes[423]
apell21 = nodes[436]
apell22 = nodes[448]
apell23 = nodes[461]
apell24 = nodes[475]
apell25 = nodes[539]
apell26 = nodes[553]
apell27 = nodes[566]
apell28 = nodes[578]
apell29 = nodes[591]
apell30 = nodes[603]
apell31 = nodes[616]
apell32 = nodes[630]

if args.apertures:
    ap06200.addChildNode(app06200, AccNode.EXIT)
    ap06201.addChildNode(app06201, AccNode.EXIT)

    ap10000.addChildNode(app10000, AccNode.EXIT)
    ap10001.addChildNode(app10001, AccNode.EXIT)
    ap10002.addChildNode(app10002, AccNode.EXIT)
    ap10003.addChildNode(app10003, AccNode.EXIT)
    ap10004.addChildNode(app10004, AccNode.EXIT)
    ap10005.addChildNode(app10005, AccNode.EXIT)
    ap10006.addChildNode(app10006, AccNode.EXIT)
    ap10007.addChildNode(app10007, AccNode.EXIT)
    ap10008.addChildNode(app10008, AccNode.EXIT)
    ap10009.addChildNode(app10009, AccNode.EXIT)
    ap10010.addChildNode(app10010, AccNode.EXIT)
    ap10011.addChildNode(app10011, AccNode.EXIT)
    ap10012.addChildNode(app10012, AccNode.EXIT)
    ap10013.addChildNode(app10013, AccNode.EXIT)
    ap10014.addChildNode(app10014, AccNode.EXIT)
    ap10015.addChildNode(app10015, AccNode.EXIT)
    ap10016.addChildNode(app10016, AccNode.EXIT)
    ap10017.addChildNode(app10017, AccNode.EXIT)
    ap10018.addChildNode(app10018, AccNode.EXIT)
    ap10019.addChildNode(app10019, AccNode.EXIT)
    ap10020.addChildNode(app10020, AccNode.EXIT)
    ap10021.addChildNode(app10021, AccNode.EXIT)
    ap10022.addChildNode(app10022, AccNode.EXIT)
    ap10023.addChildNode(app10023, AccNode.EXIT)
    ap10024.addChildNode(app10024, AccNode.EXIT)
    ap10025.addChildNode(app10025, AccNode.EXIT)
    ap10026.addChildNode(app10026, AccNode.EXIT)
    ap10027.addChildNode(app10027, AccNode.EXIT)
    ap10028.addChildNode(app10028, AccNode.EXIT)
    ap10029.addChildNode(app10029, AccNode.EXIT)
    ap10030.addChildNode(app10030, AccNode.EXIT)
    ap10031.addChildNode(app10031, AccNode.EXIT)
    ap10032.addChildNode(app10032, AccNode.EXIT)
    ap10033.addChildNode(app10033, AccNode.EXIT)
    ap10034.addChildNode(app10034, AccNode.EXIT)
    ap10035.addChildNode(app10035, AccNode.EXIT)
    ap10036.addChildNode(app10036, AccNode.EXIT)
    ap10037.addChildNode(app10037, AccNode.EXIT)
    ap10038.addChildNode(app10038, AccNode.EXIT)
    ap10039.addChildNode(app10039, AccNode.EXIT)
    ap10040.addChildNode(app10040, AccNode.EXIT)
    ap10041.addChildNode(app10041, AccNode.EXIT)
    ap10042.addChildNode(app10042, AccNode.EXIT)
    ap10043.addChildNode(app10043, AccNode.EXIT)
    ap10044.addChildNode(app10044, AccNode.EXIT)
    ap10045.addChildNode(app10045, AccNode.EXIT)
    ap10046.addChildNode(app10046, AccNode.EXIT)
    ap10047.addChildNode(app10047, AccNode.EXIT)
    ap10048.addChildNode(app10048, AccNode.EXIT)
    ap10049.addChildNode(app10049, AccNode.EXIT)
    ap10050.addChildNode(app10050, AccNode.EXIT)
    ap10051.addChildNode(app10051, AccNode.EXIT)
    ap10052.addChildNode(app10052, AccNode.EXIT)
    ap10053.addChildNode(app10053, AccNode.EXIT)
    ap10054.addChildNode(app10054, AccNode.EXIT)
    ap10055.addChildNode(app10055, AccNode.EXIT)
    ap10056.addChildNode(app10056, AccNode.EXIT)
    ap10057.addChildNode(app10057, AccNode.EXIT)
    ap10058.addChildNode(app10058, AccNode.EXIT)
    ap10059.addChildNode(app10059, AccNode.EXIT)
    ap10060.addChildNode(app10060, AccNode.EXIT)
    ap10061.addChildNode(app10061, AccNode.EXIT)
    ap10062.addChildNode(app10062, AccNode.EXIT)
    ap10063.addChildNode(app10063, AccNode.EXIT)
    ap10064.addChildNode(app10064, AccNode.EXIT)
    ap10065.addChildNode(app10065, AccNode.EXIT)
    ap10066.addChildNode(app10066, AccNode.EXIT)
    ap10067.addChildNode(app10067, AccNode.EXIT)
    ap10068.addChildNode(app10068, AccNode.EXIT)
    ap10069.addChildNode(app10069, AccNode.EXIT)

    ap12000.addChildNode(app12000, AccNode.EXIT)
    ap12001.addChildNode(app12001, AccNode.EXIT)
    ap12002.addChildNode(app12002, AccNode.EXIT)
    ap12003.addChildNode(app12003, AccNode.EXIT)
    ap12004.addChildNode(app12004, AccNode.EXIT)
    ap12005.addChildNode(app12005, AccNode.EXIT)
    ap12006.addChildNode(app12006, AccNode.EXIT)
    ap12007.addChildNode(app12007, AccNode.EXIT)
    ap12008.addChildNode(app12008, AccNode.EXIT)
    ap12009.addChildNode(app12009, AccNode.EXIT)
    ap12010.addChildNode(app12010, AccNode.EXIT)

    ap12500.addChildNode(app12500, AccNode.EXIT)
    ap12501.addChildNode(app12501, AccNode.EXIT)
    ap12502.addChildNode(app12502, AccNode.EXIT)
    ap12503.addChildNode(app12503, AccNode.EXIT)
    ap12504.addChildNode(app12504, AccNode.EXIT)
    ap12505.addChildNode(app12505, AccNode.EXIT)
    ap12506.addChildNode(app12506, AccNode.EXIT)
    ap12507.addChildNode(app12507, AccNode.EXIT)

    ap13000.addChildNode(app13000, AccNode.EXIT)
    ap13001.addChildNode(app13001, AccNode.EXIT)
    ap13002.addChildNode(app13002, AccNode.EXIT)
    ap13003.addChildNode(app13003, AccNode.EXIT)
    ap13004.addChildNode(app13004, AccNode.EXIT)

    ap14000.addChildNode(app14000, AccNode.EXIT)
    ap14001.addChildNode(app14001, AccNode.EXIT)
    ap14002.addChildNode(app14002, AccNode.EXIT)
    ap14003.addChildNode(app14003, AccNode.EXIT)
    ap14004.addChildNode(app14004, AccNode.EXIT)
    ap14005.addChildNode(app14005, AccNode.EXIT)
    ap14006.addChildNode(app14006, AccNode.EXIT)
    ap14007.addChildNode(app14007, AccNode.EXIT)
    ap14008.addChildNode(app14008, AccNode.EXIT)
    ap14009.addChildNode(app14009, AccNode.EXIT)
    ap14010.addChildNode(app14010, AccNode.EXIT)
    ap14011.addChildNode(app14011, AccNode.EXIT)
    ap14012.addChildNode(app14012, AccNode.EXIT)
    ap14013.addChildNode(app14013, AccNode.EXIT)
    ap14014.addChildNode(app14014, AccNode.EXIT)
    ap14015.addChildNode(app14015, AccNode.EXIT)
    ap14016.addChildNode(app14016, AccNode.EXIT)
    ap14017.addChildNode(app14017, AccNode.EXIT)
    ap14018.addChildNode(app14018, AccNode.EXIT)
    ap14019.addChildNode(app14019, AccNode.EXIT)

    apell00.addChildNode(appell00, AccNode.EXIT)
    apell01.addChildNode(appell01, AccNode.EXIT)
    apell02.addChildNode(appell02, AccNode.EXIT)
    apell03.addChildNode(appell03, AccNode.EXIT)
    apell04.addChildNode(appell04, AccNode.EXIT)
    apell05.addChildNode(appell05, AccNode.EXIT)
    apell06.addChildNode(appell06, AccNode.EXIT)
    apell07.addChildNode(appell07, AccNode.EXIT)
    apell08.addChildNode(appell08, AccNode.EXIT)
    apell09.addChildNode(appell09, AccNode.EXIT)
    apell10.addChildNode(appell10, AccNode.EXIT)
    apell11.addChildNode(appell11, AccNode.EXIT)
    apell12.addChildNode(appell12, AccNode.EXIT)
    apell13.addChildNode(appell13, AccNode.EXIT)
    apell14.addChildNode(appell14, AccNode.EXIT)
    apell15.addChildNode(appell15, AccNode.EXIT)
    apell16.addChildNode(appell16, AccNode.EXIT)
    apell17.addChildNode(appell17, AccNode.EXIT)
    apell18.addChildNode(appell18, AccNode.EXIT)
    apell19.addChildNode(appell19, AccNode.EXIT)
    apell20.addChildNode(appell20, AccNode.EXIT)
    apell21.addChildNode(appell21, AccNode.EXIT)
    apell22.addChildNode(appell22, AccNode.EXIT)
    apell23.addChildNode(appell23, AccNode.EXIT)
    apell24.addChildNode(appell24, AccNode.EXIT)
    apell25.addChildNode(appell25, AccNode.EXIT)
    apell26.addChildNode(appell26, AccNode.EXIT)
    apell27.addChildNode(appell27, AccNode.EXIT)
    apell28.addChildNode(appell28, AccNode.EXIT)
    apell29.addChildNode(appell29, AccNode.EXIT)
    apell30.addChildNode(appell30, AccNode.EXIT)
    apell31.addChildNode(appell31, AccNode.EXIT)
    apell32.addChildNode(appell32, AccNode.EXIT)

if args.apertures:
    addTeapotCollimatorNode(lattice, 50.3771, p1)
    addTeapotCollimatorNode(lattice, 51.1921, scr1t)
    addTeapotCollimatorNode(lattice, 51.1966, scr1c)
    addTeapotCollimatorNode(lattice, 51.2365, scr2t)
    addTeapotCollimatorNode(lattice, 51.2410, scr2c)
    addTeapotCollimatorNode(lattice, 51.3902, scr3t)
    addTeapotCollimatorNode(lattice, 51.3947, scr3c)
    addTeapotCollimatorNode(lattice, 51.4346, scr4t)
    addTeapotCollimatorNode(lattice, 51.4391, scr4c)
    addTeapotCollimatorNode(lattice, 51.6228, p2sh1)
    addTeapotCollimatorNode(lattice, 51.9428, p2)
    addTeapotCollimatorNode(lattice, 53.7428, p2sh2)
    addTeapotCollimatorNode(lattice, 57.2079, s1sh1)
    addTeapotCollimatorNode(lattice, 58.2179, s1)
    addTeapotCollimatorNode(lattice, 59.4079, s1sh2)
    addTeapotCollimatorNode(lattice, 64.8679, s2sh1)
    addTeapotCollimatorNode(lattice, 65.8779, s2)
    addTeapotCollimatorNode(lattice, 67.0679, s2sh2)


# Add RF Cavities
# --------------------------------------------------------------------------------------

z_to_phi = 2.0 * math.pi / lattice.getLength()
de_sync = 0.0
rf1_hnum = 1.0
rf1_voltage = 0.000002
rf1_phase = 0.0
rf2_hnum = 2.0
rf2_voltage = -0.000004
rf2_phase = 0.0
rf_length = 0.0

rf1_nodea = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf1_hnum, rf1_voltage, rf1_phase, rf_length, "RF1")
rf1_nodeb = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf1_hnum, rf1_voltage, rf1_phase, rf_length, "RF1")
rf1_nodec = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf1_hnum, rf1_voltage, rf1_phase, rf_length, "RF1")
rf2_node  = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf2_hnum, rf2_voltage, rf2_phase, rf_length, "RF2")

addRFNode(lattice, 184.273, rf1_nodea)
addRFNode(lattice, 186.571, rf1_nodeb)
addRFNode(lattice, 188.868, rf1_nodec)
addRFNode(lattice, 191.165, rf2_node)


# Add longitudinal impedance Node
# --------------------------------------------------------------------------------------

impedances_ekicker = read_longitudinal_impedance_file("inputs/sns_ring_zl_ekicker.dat")
impedances_rf = read_longitudinal_impedance_file("inputs/sns_ring_zl_rf.dat")

impedances = []
for z_kick, z_rf in zip(impedances_ekicker, impedances_rf):
    z_real = (z_kick.real / 1.75) + z_rf.real
    z_imag = (z_kick.imag / 1.75) + z_rf.imag
    impedance = complex(z_real, z_imag)
    impedances.append(impedance)

n_macros_min = args.imp_z_macros_min
n_bins = args.imp_z_bins
position = 124.0

impedance_node = LImpedance_Node(lattice.getLength(), n_macros_min, n_bins)
impedance_node.assignImpedance(impedances)

if args.imp_z:
    addImpedanceNode(lattice, position, impedance_node)


# Add transverse impedance Node
# --------------------------------------------------------------------------------------

n_macros_min = args.imp_xy_macros_min
n_bins = args.imp_xy_bins
use_x = False
use_y = True
alpha_x = 0.0
alpha_y = -0.004
beta_x = 10.191
beta_y = 10.447
tune_x = 6.21991
tune_y = 6.20936
position = 124.0

ZP, ZM = read_transverse_impedance_file("inputs/sns_ring_hahn_impedance.dat")

impedance_node = TImpedance_Node(lattice.getLength(), n_macros_min, n_bins, int(use_x), int(use_y))
impedance_node.assignLatFuncs(tune_x, alpha_x, beta_x, tune_y, alpha_y, beta_y)
if use_x:
    impedance_node.assignImpedance("X", ZP, ZM)
if use_y:
    impedance_node.assignImpedance("Y", ZP, ZM)

if args.imp_xy:
    addImpedanceNode(lattice, position, impedance_node)



# Add longitudinal space charge node
# --------------------------------------------------------------------------------------

b_over_a = 10.0 / 3.0
n_bins = args.sc_z_bins
n_macros_min = args.sc_z_macros_min
position = 124.0

impedance = [complex(0.0, 0.0) for _ in range(n_bins // 2)]

sc_node_long = SC1D_AccNode(b_over_a, lattice.getLength(), n_macros_min, 1, n_bins)
sc_node_long.assignImpedance(impedance)

if args.sc_z:
    addLongitudinalSpaceChargeNode(lattice, position, sc_node_long)


# Add transverse space charge nodes
# --------------------------------------------------------------------------------------

boundary = None
if args.sc_xy_boundary:
    boundary_type = "Circle"
    boundary_radius = 0.220
    boundary = Boundary2D(
        args.sc_xy_boundary_points,
        args.sc_xy_boundary_modes,
        boundary_type,
        boundary_radius,
        boundary_radius
    )

sc_path_length_min = 1.00e-08

sc_nodes = []
if args.sc_xy:
    if args.sc_xy_solver == "2p5d":
        sc_calc = SpaceChargeCalc2p5D(args.sc_xy_gridx, args.sc_xy_gridy, args.sc_xy_gridz)
        sc_nodes = setSC2p5DAccNodes(
            lattice,
            sc_path_length_min,
            sc_calc,
            boundary=boundary,
        )
    elif args.sc_xy_solver in ["slicebyslice", "slice", "sbs"]:
        sc_calc = SpaceChargeCalcSliceBySlice2D(args.sc_xy_gridx, args.sc_xy_gridy, args.sc_xy_gridz)
        sc_nodes = setSC2DSliceBySliceAccNodes(
            lattice,
            sc_path_length_min,
            sc_calc,
            boundary=boundary
        )


# Add diagnostic nodes
# --------------------------------------------------------------------------------------


# Track bunch
# --------------------------------------------------------------------------------------

start_time = time.time()

for turn_index in range(args.turns):
    lattice.trackBunch(bunch, params_dict)

    # Diagnostics
    time_ellapsed = time.time() - start_time
    bunch_size = bunch.getSizeGlobal()

    # Compute moments
    twiss_calc = BunchTwissAnalysis()
    twiss_calc.computeBunchMoments(bunch, 2, 0, 0)
    x_rms = np.sqrt(twiss_calc.getCorrelation(0, 0)) * 1000.0
    y_rms = np.sqrt(twiss_calc.getCorrelation(2, 2)) * 1000.0

    if _mpi_rank == 0:
        message = ""
        message += "turn={:04.0f} ".format(turn_index + 1)
        message += "time={:0.3f} ".format(time_ellapsed)
        message += "size={:0.2e} ".format(bunch_size)
        message += "xrms={:0.2f} ".format(x_rms)
        message += "yrms={:0.2f} ".format(y_rms)
        print(message)
        sys.stdout.flush()

    if (turn_index % args.write_bunch_freq == 0) or (turn_index == args.turns - 1):
        filename = "bunch_{:04.0f}".format(turn_index)
        filename = os.path.join(output_dir, filename)
        if _mpi_rank == 0:
            print(filename)
            sys.stdout.flush()
        bunch.dumpBunch(filename)
 