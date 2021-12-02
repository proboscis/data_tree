from loguru import logger

from data_tree.auto_data.solver import DEFAULT_ASTAR_DESIGN, AstarSolver


def test_solve():
    solver = DEFAULT_ASTAR_DESIGN.provide(AstarSolver)
    res = solver.solve("image,RGB,RGB","widget")
    # ok,
    # I just need intra dict conversions?
    logger.info(res)