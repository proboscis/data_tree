from data_tree.coconut.auto_data import AutoData, AutoSolver
from data_tree.auto_data.default_rules import DEFAULT_RULES
SOLVER = None





def solver_getter():
    return SOLVER


def auto_img(format):
    def _get_value(value):
        return AutoData(value, format, solver_getter)

    return _get_value


def get_init_solver():
    from data_tree.coconut.omni_converter import tuple_widget_heuristics, tuple_edge_cutter, debug
    return AutoSolver(
        rules=list(DEFAULT_RULES.rules),
        smart_rules=[],
        heuristics=tuple_widget_heuristics,
        edge_cutter=tuple_edge_cutter,
        debug_hook=debug
    )

def reset_solver():
    global SOLVER
    SOLVER = get_init_solver()


reset_solver()