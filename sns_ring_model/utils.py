from collections import Counter
from pprint import pprint

from orbit.lattice import AccLattice
from orbit.lattice import AccNode


def get_node_by_name_any_case(lattice: AccLattice, name: str) -> AccNode:
    nodes = {node.getName(): node for node in lattice.getNodes()}
    node = nodes.get(name, None)
    if node is None:
        node = nodes.get(name.upper(), None)
    if node is None:
        node = nodes.get(name.lower(), None)
    if node is None:
        raise ValueError(f"Could not find node name {name}")
    return node


def get_nodes_by_name_any_case(lattice: AccLattice, names: list[str]) -> list[AccNode]:
    return [get_node_by_name_any_case(lattice, name) for name in names]


def rename_nodes_avoid_duplicates(
    lattice: AccLattice,
    verbose: bool = False,
    delimiter="_",
) -> AccLattice:
    node_names = [node.getName() for node in lattice.getNodes()]
    counter = Counter(node_names)
    if verbose:
        pprint(counter)
    for name, count in counter.items():
        if count > 1:
            for index, node in enumerate(lattice.getNodes()):
                if node.getName() == name:
                    old_name = node.getName()
                    new_name = f"{old_name}_{index}"
                    if verbose:
                        print(f"index={index} {old_name} -> {new_name}")
                    node.setName(new_name)
    return lattice