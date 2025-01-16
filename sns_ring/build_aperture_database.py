"""Manually add aperture nodes to SNS ring lattice (read from MAD file)."""
import os
from pprint import pprint

import pandas

from orbit.core.aperture import Aperture
from orbit.core.bunch import Bunch
from orbit.core.collimator import Collimator
from orbit.core.foil import Foil
from orbit.aperture import ApertureLatticeModifications
from orbit.aperture import ApertureLatticeRangeModifications
from orbit.aperture import CircleApertureNode
from orbit.aperture import EllipseApertureNode
from orbit.aperture import RectangleApertureNode
from orbit.aperture import TeapotApertureNode
from orbit.collimation import addTeapotCollimatorNode
from orbit.collimation import TeapotCollimatorNode
from orbit.bumps import TeapotBumpNode
from orbit.bumps import TDTeapotSimpleBumpNode
from orbit.collimation import TeapotCollimatorNode
from orbit.kickernodes import flatTopWaveform
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.foils import TeapotFoilNode
from orbit.rf_cavities import RFNode
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.teapot import DriftTEAPOT

from sns_orbit_models.ring.model import SNS_RING
from sns_orbit_models.ring.model import rename_nodes_avoid_duplicates
from sns_orbit_models.ring.utils import get_node_by_name_any_case


ENTRANCE = AccNode.ENTRANCE
EXIT = AccNode.EXIT
BODY = AccNode.BODY


class ApertureChildNodeInfo:
    def __init__(
        self,
        node: AccNode,
        parent_node: AccNode,
        place: int,
        part_index: int, **kwargs
    ) -> None:
        self.node = node
        self.name = node.getName()
        self.type= type(node)
        self.class_name = node.__class__.__name__

        self.parent_node = parent_node
        self.parent_name = parent_node.getName()
        self.parent_index = lattice.getNodeIndex(parent_node)

        self.place = place
        self.part_index = part_index

        self.shape = self.node.shape
        self.size_x = self.size_y = None
        if self.shape == 1:
            self.size_x = node.a
            self.size_y = self.size_x
        else:
            self.size_x = node.a
            self.size_y = node.b
        self.position = node.pos


class ApertureChildNodeInfoWriter:
    def __init__(self, filename: str, verbose: bool = True) -> None:
        self.filename = filename
        self.verbose = verbose

        self.keys = [
            "name",
            "class_name",
            "parent_name",
            "place",
            "part_index",
            "size_x",
            "size_y",
            "position",
        ]
        self.header = ",".join(self.keys)
        if self.verbose:
            print(self.header)

        self.file = open(self.filename, "w")
        self.file.write(self.header + "\n")

    def get_line(self, aperture_child_node_info: ApertureChildNodeInfo) -> str:
        line = []
        for key in self.keys:
            line.append(str(getattr(aperture_child_node_info, key)))
        line = ",".join(line)
        return line

    def write_line(self, aperture_child_node_info: ApertureChildNodeInfo) -> None:
        line = self.get_line(aperture_child_node_info)
        if self.verbose:
            print(line)
        self.file.write(line + "\n")

    def close(self) -> None:
        self.file.close()


def is_aperture_node(node: AccNode) -> bool:
    aperture_node_classes = [
        CircleApertureNode,
        EllipseApertureNode,
        RectangleApertureNode,
    ]
    return type(node) in aperture_node_classes


def collect_aperture_node_info(lattice: AccLattice) -> list[ApertureChildNodeInfo]:
    aperture_node_info_list = []
    for index, node in enumerate(lattice.getNodes()):
        for place in [ENTRANCE, BODY, EXIT]:
            for part_index, child_node in enumerate(node.getChildNodes(place)):
                if is_aperture_node(child_node):
                    child_node_info = ApertureChildNodeInfo(
                        node=child_node,
                        parent_node=node,
                        place=place,
                        part_index=part_index,
                        lattice=lattice,
                    )
                    aperture_node_info_list.append(child_node_info)
    return aperture_node_info_list


def save_aperture_node_info(lattice: AccLattice, filename: str) -> None:
    writer = ApertureChildNodeInfoWriter(filename)
    for aperture_node_info in collect_aperture_node_info(lattice):
        writer.write_line(aperture_node_info)
    writer.close()


def add_aperture_nodes_by_index(lattice: AccLattice) -> AccLattice:
    """Add aperture and collimator nodes around the ring.

    [Accesses nodes by index: works with MAD output lattice file, not MADX.]
    """
    nodes = lattice.getNodes()

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

    app06200 = CircleApertureNode(r062, 58.21790, 0.0, 0.0, name="s1")
    app06201 = CircleApertureNode(r062, 65.87790, 0.0, 0.0, name="s2")

    app10000 = CircleApertureNode(r100, 10.54740, 0.0, 0.0, name="bp100")
    app10001 = CircleApertureNode(r100, 10.97540, 0.0, 0.0, name="bp100")
    app10002 = CircleApertureNode(r100, 13.57190, 0.0, 0.0, name="bp100")
    app10003 = CircleApertureNode(r100, 15.39140, 0.0, 0.0, name="21cmquad")
    app10004 = CircleApertureNode(r100, 19.39150, 0.0, 0.0, name="21cmquad")
    app10005 = CircleApertureNode(r100, 23.39170, 0.0, 0.0, name="21cmquad")
    app10006 = CircleApertureNode(r100, 31.39210, 0.0, 0.0, name="21cmquad")
    app10007 = CircleApertureNode(r100, 39.39250, 0.0, 0.0, name="21cmquad")
    app10008 = CircleApertureNode(r100, 43.39270, 0.0, 0.0, name="21cmquad")
    app10009 = CircleApertureNode(r100, 47.39290, 0.0, 0.0, name="21cmquad")
    app10010 = CircleApertureNode(r100, 48.86630, 0.0, 0.0, name="bp100")
    app10011 = CircleApertureNode(r100, 50.37710, 0.0, 0.0, name="P1a")
    app10012 = CircleApertureNode(r100, 51.19660, 0.0, 0.0, name="scr1c")
    app10013 = CircleApertureNode(r100, 51.24100, 0.0, 0.0, name="scr2c")
    app10014 = CircleApertureNode(r100, 51.39470, 0.0, 0.0, name="scr3c")
    app10015 = CircleApertureNode(r100, 51.43910, 0.0, 0.0, name="scr4c")
    app10016 = CircleApertureNode(r100, 51.62280, 0.0, 0.0, name="p2shield")
    app10017 = CircleApertureNode(r100, 53.74280, 0.0, 0.0, name="p2shield")
    app10018 = CircleApertureNode(r100, 54.75640, 0.0, 0.0, name="bp100")
    app10019 = CircleApertureNode(r100, 71.16320, 0.0, 0.0, name="bp100")
    app10020 = CircleApertureNode(r100, 73.35170, 0.0, 0.0, name="bp100")
    app10021 = CircleApertureNode(r100, 75.60170, 0.0, 0.0, name="bp100")
    app10022 = CircleApertureNode(r100, 76.79720, 0.0, 0.0, name="bp100")
    app10023 = CircleApertureNode(r100, 77.39290, 0.0, 0.0, name="21cmquad")
    app10024 = CircleApertureNode(r100, 81.39310, 0.0, 0.0, name="21cmquad")
    app10025 = CircleApertureNode(r100, 85.39330, 0.0, 0.0, name="21cmquad")
    app10026 = CircleApertureNode(r100, 93.39370, 0.0, 0.0, name="21cmquad")
    app10027 = CircleApertureNode(r100, 101.39400, 0.0, 0.0, name="21cmquad")
    app10028 = CircleApertureNode(r100, 105.39400, 0.0, 0.0, name="21cmquad")
    app10029 = CircleApertureNode(r100, 109.39400, 0.0, 0.0, name="21cmquad")
    app10030 = CircleApertureNode(r100, 110.49000, 0.0, 0.0, name="bp100")
    app10031 = CircleApertureNode(r100, 112.69100, 0.0, 0.0, name="bp100")
    app10032 = CircleApertureNode(r100, 114.82200, 0.0, 0.0, name="bp100")
    app10033 = CircleApertureNode(r100, 118.38600, 0.0, 0.0, name="bp100")
    app10034 = CircleApertureNode(r100, 120.37900, 0.0, 0.0, name="bp100")
    app10035 = CircleApertureNode(r100, 122.21700, 0.0, 0.0, name="bp100")
    app10036 = CircleApertureNode(r100, 124.64400, 0.0, 0.0, name="bp100")
    app10037 = CircleApertureNode(r100, 127.77400, 0.0, 0.0, name="bp100")
    app10038 = CircleApertureNode(r100, 132.53100, 0.0, 0.0, name="bp100")
    app10039 = CircleApertureNode(r100, 136.10400, 0.0, 0.0, name="bp100")
    app10040 = CircleApertureNode(r100, 138.79900, 0.0, 0.0, name="bp100")
    app10041 = CircleApertureNode(r100, 139.39400, 0.0, 0.0, name="21cmquad")
    app10042 = CircleApertureNode(r100, 143.39500, 0.0, 0.0, name="21cmquad")
    app10043 = CircleApertureNode(r100, 147.39500, 0.0, 0.0, name="21cmquad")
    app10044 = CircleApertureNode(r100, 155.39500, 0.0, 0.0, name="21cmquad")
    app10045 = CircleApertureNode(r100, 163.39600, 0.0, 0.0, name="21cmquad")
    app10046 = CircleApertureNode(r100, 167.39600, 0.0, 0.0, name="21cmquad")
    app10047 = CircleApertureNode(r100, 171.39600, 0.0, 0.0, name="21cmquad")
    app10048 = CircleApertureNode(r100, 173.70900, 0.0, 0.0, name="bp100")
    app10049 = CircleApertureNode(r100, 175.93900, 0.0, 0.0, name="bp100")
    app10050 = CircleApertureNode(r100, 180.38700, 0.0, 0.0, name="bp100")
    app10051 = CircleApertureNode(r100, 182.12700, 0.0, 0.0, name="bp100")
    app10052 = CircleApertureNode(r100, 184.27300, 0.0, 0.0, name="bp100")
    app10053 = CircleApertureNode(r100, 186.57100, 0.0, 0.0, name="bp100")
    app10054 = CircleApertureNode(r100, 188.86800, 0.0, 0.0, name="bp100")
    app10055 = CircleApertureNode(r100, 191.16500, 0.0, 0.0, name="bp100")
    app10056 = CircleApertureNode(r100, 194.53200, 0.0, 0.0, name="bp100")
    app10057 = CircleApertureNode(r100, 196.61400, 0.0, 0.0, name="bp100")
    app10058 = CircleApertureNode(r100, 199.47500, 0.0, 0.0, name="bp100")
    app10059 = CircleApertureNode(r100, 201.39600, 0.0, 0.0, name="21cmquad")
    app10060 = CircleApertureNode(r100, 205.39600, 0.0, 0.0, name="21cmquad")
    app10061 = CircleApertureNode(r100, 209.39600, 0.0, 0.0, name="21cmquad")
    app10062 = CircleApertureNode(r100, 217.39700, 0.0, 0.0, name="21cmquad")
    app10063 = CircleApertureNode(r100, 225.39700, 0.0, 0.0, name="21cmquad")
    app10064 = CircleApertureNode(r100, 229.39700, 0.0, 0.0, name="21cmquad")
    app10065 = CircleApertureNode(r100, 233.39700, 0.0, 0.0, name="21cmquad")
    app10066 = CircleApertureNode(r100, 234.87800, 0.0, 0.0, name="bp100")
    app10067 = CircleApertureNode(r100, 236.87700, 0.0, 0.0, name="bp100")
    app10068 = CircleApertureNode(r100, 238.74100, 0.0, 0.0, name="bp100")
    app10069 = CircleApertureNode(r100, 242.38900, 0.0, 0.0, name="bp100")

    app12000 = CircleApertureNode(r120, 6.89986, 0.0, 0.0, name="bp120")
    app12001 = CircleApertureNode(r120, 8.52786, 0.0, 0.0, name="bp120")
    app12002 = CircleApertureNode(r120, 57.20790, 0.0, 0.0, name="s1shield")
    app12003 = CircleApertureNode(r120, 59.40790, 0.0, 0.0, name="s1shield")
    app12004 = CircleApertureNode(r120, 64.86790, 0.0, 0.0, name="s2shield")
    app12005 = CircleApertureNode(r120, 67.06790, 0.0, 0.0, name="s2shield")
    app12006 = CircleApertureNode(r120, 116.75800, 0.0, 0.0, name="bp120")
    app12007 = CircleApertureNode(r120, 130.90300, 0.0, 0.0, name="bp120")
    app12008 = CircleApertureNode(r120, 178.75900, 0.0, 0.0, name="bp120")
    app12009 = CircleApertureNode(r120, 192.90400, 0.0, 0.0, name="bp120")
    app12010 = CircleApertureNode(r120, 240.76100, 0.0, 0.0, name="bp120")

    app12500 = CircleApertureNode(r125, 27.37140, 0.0, 0.0, name="26cmquad")
    app12501 = CircleApertureNode(r125, 35.37180, 0.0, 0.0, name="26cmquad")
    app12502 = CircleApertureNode(r125, 89.37300, 0.0, 0.0, name="26cmquad")
    app12503 = CircleApertureNode(r125, 97.37330, 0.0, 0.0, name="26cmquad")
    app12504 = CircleApertureNode(r125, 151.37400, 0.0, 0.0, name="26cmquad")
    app12505 = CircleApertureNode(r125, 159.37500, 0.0, 0.0, name="26cmquad")
    app12506 = CircleApertureNode(r125, 213.37600, 0.0, 0.0, name="26cmquad")
    app12507 = CircleApertureNode(r125, 221.37600, 0.0, 0.0, name="26cmquad")

    app13000 = CircleApertureNode(r130, 60.41790, 0.0, 0.0, name="bp130")
    app13001 = CircleApertureNode(r130, 64.42290, 0.0, 0.0, name="bp130")
    app13002 = CircleApertureNode(r130, 68.07790, 0.0, 0.0, name="bp130")
    app13003 = CircleApertureNode(r130, 68.90140, 0.0, 0.0, name="bp130")
    app13004 = CircleApertureNode(r130, 70.52940, 0.0, 0.0, name="bp130")

    app14000 = CircleApertureNode(r140, 7.43286, 0.0, 0.0, name="30cmquad")
    app14001 = CircleApertureNode(r140, 7.85486, 0.0, 0.0, name="30cmquad")
    app14002 = CircleApertureNode(r140, 55.42940, 0.0, 0.0, name="30cmquad")
    app14003 = CircleApertureNode(r140, 55.85140, 0.0, 0.0, name="30cmquad")
    app14004 = CircleApertureNode(r140, 56.38440, 0.0, 0.0, name="30cmquad")
    app14005 = CircleApertureNode(r140, 60.86290, 0.0, 0.0, name="bp140")
    app14006 = CircleApertureNode(r140, 62.64290, 0.0, 0.0, name="bp140")
    app14007 = CircleApertureNode(r140, 63.97790, 0.0, 0.0, name="bp140")
    app14008 = CircleApertureNode(r140, 69.43440, 0.0, 0.0, name="30cmquad")
    app14009 = CircleApertureNode(r140, 69.85640, 0.0, 0.0, name="30cmquad")
    app14010 = CircleApertureNode(r140, 117.43100, 0.0, 0.0, name="30cmquad")
    app14011 = CircleApertureNode(r140, 117.85300, 0.0, 0.0, name="30cmquad")
    app14012 = CircleApertureNode(r140, 131.43600, 0.0, 0.0, name="30cmquad")
    app14013 = CircleApertureNode(r140, 131.85800, 0.0, 0.0, name="30cmquad")
    app14014 = CircleApertureNode(r140, 179.43200, 0.0, 0.0, name="30cmquad")
    app14015 = CircleApertureNode(r140, 179.85400, 0.0, 0.0, name="30cmquad")
    app14016 = CircleApertureNode(r140, 193.43700, 0.0, 0.0, name="30cmquad")
    app14017 = CircleApertureNode(r140, 193.85900, 0.0, 0.0, name="30cmquad")
    app14018 = CircleApertureNode(r140, 241.43400, 0.0, 0.0, name="30cmquad")
    app14019 = CircleApertureNode(r140, 241.85600, 0.0, 0.0, name="30cmquad")

    appell00 = EllipseApertureNode(a115p8, b078p7, 16.9211, 0.0, 0.0, name="arcdipole")
    appell01 = EllipseApertureNode(a115p8, b078p7, 20.9213, 0.0, 0.0, name="arcdipole")
    appell02 = EllipseApertureNode(a115p8, b078p7, 24.9215, 0.0, 0.0, name="arcdipole")
    appell03 = EllipseApertureNode(a115p8, b078p7, 28.9217, 0.0, 0.0, name="arcdipole")
    appell04 = EllipseApertureNode(a115p8, b078p7, 32.9219, 0.0, 0.0, name="arcdipole")
    appell05 = EllipseApertureNode(a115p8, b078p7, 36.9221, 0.0, 0.0, name="arcdipole")
    appell06 = EllipseApertureNode(a115p8, b078p7, 40.9222, 0.0, 0.0, name="arcdipole")
    appell07 = EllipseApertureNode(a115p8, b078p7, 44.9224, 0.0, 0.0, name="arcdipole")
    appell08 = EllipseApertureNode(a080p0, b048p0, 51.9428, 0.0, 0.0, name="p2")
    appell09 = EllipseApertureNode(a115p8, b078p7, 78.9226, 0.0, 0.0, name="arcdipole")
    appell10 = EllipseApertureNode(a115p8, b078p7, 82.9228, 0.0, 0.0, name="arcdipole")
    appell11 = EllipseApertureNode(a115p8, b078p7, 86.9230, 0.0, 0.0, name="arcdipole")
    appell12 = EllipseApertureNode(a115p8, b078p7, 90.9232, 0.0, 0.0, name="arcdipole")
    appell13 = EllipseApertureNode(a115p8, b078p7, 94.9234, 0.0, 0.0, name="arcdipole")
    appell14 = EllipseApertureNode(a115p8, b078p7, 98.9236, 0.0, 0.0, name="arcdipole")
    appell15 = EllipseApertureNode(a115p8, b078p7, 102.9240, 0.0, 0.0, name="arcdipole")
    appell16 = EllipseApertureNode(a115p8, b078p7, 106.9240, 0.0, 0.0, name="arcdipole")
    appell17 = EllipseApertureNode(a115p8, b078p7, 140.9240, 0.0, 0.0, name="arcdipole")
    appell18 = EllipseApertureNode(a115p8, b078p7, 144.9240, 0.0, 0.0, name="arcdipole")
    appell19 = EllipseApertureNode(a115p8, b078p7, 148.9250, 0.0, 0.0, name="arcdipole")
    appell20 = EllipseApertureNode(a115p8, b078p7, 152.9250, 0.0, 0.0, name="arcdipole")
    appell21 = EllipseApertureNode(a115p8, b078p7, 156.9250, 0.0, 0.0, name="arcdipole")
    appell22 = EllipseApertureNode(a115p8, b078p7, 160.9250, 0.0, 0.0, name="arcdipole")
    appell23 = EllipseApertureNode(a115p8, b078p7, 164.9250, 0.0, 0.0, name="arcdipole")
    appell24 = EllipseApertureNode(a115p8, b078p7, 168.9260, 0.0, 0.0, name="arcdipole")
    appell25 = EllipseApertureNode(a115p8, b078p7, 202.9260, 0.0, 0.0, name="arcdipole")
    appell26 = EllipseApertureNode(a115p8, b078p7, 206.9260, 0.0, 0.0, name="arcdipole")
    appell27 = EllipseApertureNode(a115p8, b078p7, 210.9260, 0.0, 0.0, name="arcdipole")
    appell28 = EllipseApertureNode(a115p8, b078p7, 214.9260, 0.0, 0.0, name="arcdipole")
    appell29 = EllipseApertureNode(a115p8, b078p7, 218.9260, 0.0, 0.0, name="arcdipole")
    appell30 = EllipseApertureNode(a115p8, b078p7, 222.9270, 0.0, 0.0, name="arcdipole")
    appell31 = EllipseApertureNode(a115p8, b078p7, 226.9270, 0.0, 0.0, name="arcdipole")
    appell32 = EllipseApertureNode(a115p8, b078p7, 230.9270, 0.0, 0.0, name="arcdipole")

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

    ap06200.addChildNode(app06200, EXIT)
    ap06201.addChildNode(app06201, EXIT)

    ap10000.addChildNode(app10000, EXIT)
    ap10001.addChildNode(app10001, EXIT)
    ap10002.addChildNode(app10002, EXIT)
    ap10003.addChildNode(app10003, EXIT)
    ap10004.addChildNode(app10004, EXIT)
    ap10005.addChildNode(app10005, EXIT)
    ap10006.addChildNode(app10006, EXIT)
    ap10007.addChildNode(app10007, EXIT)
    ap10008.addChildNode(app10008, EXIT)
    ap10009.addChildNode(app10009, EXIT)
    ap10010.addChildNode(app10010, EXIT)
    ap10011.addChildNode(app10011, EXIT)
    ap10012.addChildNode(app10012, EXIT)
    ap10013.addChildNode(app10013, EXIT)
    ap10014.addChildNode(app10014, EXIT)
    ap10015.addChildNode(app10015, EXIT)
    ap10016.addChildNode(app10016, EXIT)
    ap10017.addChildNode(app10017, EXIT)
    ap10018.addChildNode(app10018, EXIT)
    ap10019.addChildNode(app10019, EXIT)
    ap10020.addChildNode(app10020, EXIT)
    ap10021.addChildNode(app10021, EXIT)
    ap10022.addChildNode(app10022, EXIT)
    ap10023.addChildNode(app10023, EXIT)
    ap10024.addChildNode(app10024, EXIT)
    ap10025.addChildNode(app10025, EXIT)
    ap10026.addChildNode(app10026, EXIT)
    ap10027.addChildNode(app10027, EXIT)
    ap10028.addChildNode(app10028, EXIT)
    ap10029.addChildNode(app10029, EXIT)
    ap10030.addChildNode(app10030, EXIT)
    ap10031.addChildNode(app10031, EXIT)
    ap10032.addChildNode(app10032, EXIT)
    ap10033.addChildNode(app10033, EXIT)
    ap10034.addChildNode(app10034, EXIT)
    ap10035.addChildNode(app10035, EXIT)
    ap10036.addChildNode(app10036, EXIT)
    ap10037.addChildNode(app10037, EXIT)
    ap10038.addChildNode(app10038, EXIT)
    ap10039.addChildNode(app10039, EXIT)
    ap10040.addChildNode(app10040, EXIT)
    ap10041.addChildNode(app10041, EXIT)
    ap10042.addChildNode(app10042, EXIT)
    ap10043.addChildNode(app10043, EXIT)
    ap10044.addChildNode(app10044, EXIT)
    ap10045.addChildNode(app10045, EXIT)
    ap10046.addChildNode(app10046, EXIT)
    ap10047.addChildNode(app10047, EXIT)
    ap10048.addChildNode(app10048, EXIT)
    ap10049.addChildNode(app10049, EXIT)
    ap10050.addChildNode(app10050, EXIT)
    ap10051.addChildNode(app10051, EXIT)
    ap10052.addChildNode(app10052, EXIT)
    ap10053.addChildNode(app10053, EXIT)
    ap10054.addChildNode(app10054, EXIT)
    ap10055.addChildNode(app10055, EXIT)
    ap10056.addChildNode(app10056, EXIT)
    ap10057.addChildNode(app10057, EXIT)
    ap10058.addChildNode(app10058, EXIT)
    ap10059.addChildNode(app10059, EXIT)
    ap10060.addChildNode(app10060, EXIT)
    ap10061.addChildNode(app10061, EXIT)
    ap10062.addChildNode(app10062, EXIT)
    ap10063.addChildNode(app10063, EXIT)
    ap10064.addChildNode(app10064, EXIT)
    ap10065.addChildNode(app10065, EXIT)
    ap10066.addChildNode(app10066, EXIT)
    ap10067.addChildNode(app10067, EXIT)
    ap10068.addChildNode(app10068, EXIT)
    ap10069.addChildNode(app10069, EXIT)

    ap12000.addChildNode(app12000, EXIT)
    ap12001.addChildNode(app12001, EXIT)
    ap12002.addChildNode(app12002, EXIT)
    ap12003.addChildNode(app12003, EXIT)
    ap12004.addChildNode(app12004, EXIT)
    ap12005.addChildNode(app12005, EXIT)
    ap12006.addChildNode(app12006, EXIT)
    ap12007.addChildNode(app12007, EXIT)
    ap12008.addChildNode(app12008, EXIT)
    ap12009.addChildNode(app12009, EXIT)
    ap12010.addChildNode(app12010, EXIT)

    ap12500.addChildNode(app12500, EXIT)
    ap12501.addChildNode(app12501, EXIT)
    ap12502.addChildNode(app12502, EXIT)
    ap12503.addChildNode(app12503, EXIT)
    ap12504.addChildNode(app12504, EXIT)
    ap12505.addChildNode(app12505, EXIT)
    ap12506.addChildNode(app12506, EXIT)
    ap12507.addChildNode(app12507, EXIT)

    ap13000.addChildNode(app13000, EXIT)
    ap13001.addChildNode(app13001, EXIT)
    ap13002.addChildNode(app13002, EXIT)
    ap13003.addChildNode(app13003, EXIT)
    ap13004.addChildNode(app13004, EXIT)

    ap14000.addChildNode(app14000, EXIT)
    ap14001.addChildNode(app14001, EXIT)
    ap14002.addChildNode(app14002, EXIT)
    ap14003.addChildNode(app14003, EXIT)
    ap14004.addChildNode(app14004, EXIT)
    ap14005.addChildNode(app14005, EXIT)
    ap14006.addChildNode(app14006, EXIT)
    ap14007.addChildNode(app14007, EXIT)
    ap14008.addChildNode(app14008, EXIT)
    ap14009.addChildNode(app14009, EXIT)
    ap14010.addChildNode(app14010, EXIT)
    ap14011.addChildNode(app14011, EXIT)
    ap14012.addChildNode(app14012, EXIT)
    ap14013.addChildNode(app14013, EXIT)
    ap14014.addChildNode(app14014, EXIT)
    ap14015.addChildNode(app14015, EXIT)
    ap14016.addChildNode(app14016, EXIT)
    ap14017.addChildNode(app14017, EXIT)
    ap14018.addChildNode(app14018, EXIT)
    ap14019.addChildNode(app14019, EXIT)

    apell00.addChildNode(appell00, EXIT)
    apell01.addChildNode(appell01, EXIT)
    apell02.addChildNode(appell02, EXIT)
    apell03.addChildNode(appell03, EXIT)
    apell04.addChildNode(appell04, EXIT)
    apell05.addChildNode(appell05, EXIT)
    apell06.addChildNode(appell06, EXIT)
    apell07.addChildNode(appell07, EXIT)
    apell08.addChildNode(appell08, EXIT)
    apell09.addChildNode(appell09, EXIT)
    apell10.addChildNode(appell10, EXIT)
    apell11.addChildNode(appell11, EXIT)
    apell12.addChildNode(appell12, EXIT)
    apell13.addChildNode(appell13, EXIT)
    apell14.addChildNode(appell14, EXIT)
    apell15.addChildNode(appell15, EXIT)
    apell16.addChildNode(appell16, EXIT)
    apell17.addChildNode(appell17, EXIT)
    apell18.addChildNode(appell18, EXIT)
    apell19.addChildNode(appell19, EXIT)
    apell20.addChildNode(appell20, EXIT)
    apell21.addChildNode(appell21, EXIT)
    apell22.addChildNode(appell22, EXIT)
    apell23.addChildNode(appell23, EXIT)
    apell24.addChildNode(appell24, EXIT)
    apell25.addChildNode(appell25, EXIT)
    apell26.addChildNode(appell26, EXIT)
    apell27.addChildNode(appell27, EXIT)
    apell28.addChildNode(appell28, EXIT)
    apell29.addChildNode(appell29, EXIT)
    apell30.addChildNode(appell30, EXIT)
    apell31.addChildNode(appell31, EXIT)
    apell32.addChildNode(appell32, EXIT)

    return lattice


if __name__ == "__main__":
    # Create lattice
    lattice_file = "./mad/sns_ring_mad.lattice"
    lattice_file_seq = "RINGINJ"

    lattice = TEAPOT_Ring()
    lattice.readMAD(lattice_file, lattice_file_seq)
    lattice = rename_nodes_avoid_duplicates(lattice)
    lattice.initialize()

    # Add aperture nodes by index
    lattice = add_aperture_nodes_by_index(lattice)

    # Build and save aperture database
    filename = "./data/aperture_node_info.txt"
    save_aperture_node_info(lattice, filename)

    # Test loading apertures from database
    model = SNS_RING()
    model.add_all_aperture_and_collimator_nodes()