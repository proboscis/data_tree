from typing import Iterable


def tabulate_dicts(dicts: Iterable[dict]):
    import tabulate
    import pandas as pd
    df = pd.DataFrame(dicts)
    s = tabulate.tabulate(df, headers=df.columns)
    return s