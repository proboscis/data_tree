import PIL
from IPython import embed
from loguru import logger

from data_tree.auto_data.omni_solver import RuleResult, ParameterizedAstarSolver, eval_solution, Conversion
from data_tree.auto_data.widget_rules import provide_conversion_widget


def _rule(q):
    if q == "A":
        return [RuleResult(
            ["B"],
            lambda b: f"{b}->A",
            1,
            "A if B exists"
        )]
    if q == "B":
        return [RuleResult(
            ["C1", "C2"],
            lambda c1, c2: f"({c1},{c2})->B",
            1,
            "fact B"
        )]
    if q == "C1" or q == "C2":
        return [
            RuleResult(
                [],
                lambda: q,
                1,
                f"fact {q}"
            )
        ]
    return []


def _h(pos):
    return 0


def test_solve():
    solver = ParameterizedAstarSolver()
    res = solver.solve("A", _rule, _h)
    assert eval_solution(res) == '(C1,C2)->B->A'

def test_widget_rule():
    solver = ParameterizedAstarSolver()
    res = solver.solve(Conversion(PIL.Image.Image,"widget"),provide_conversion_widget,_h)
    logger.info(eval_solution(res))