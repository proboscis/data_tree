import pandas as pd
import numpy as np
from data_tree.table import from_df


def test_load_df():
    df = pd.DataFrame(np.arange(100).reshape(50,2),columns="a b".split())
    t = from_df(df)
    print(t[-1])