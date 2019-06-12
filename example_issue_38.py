"""Reproducing issue 38 for tigramite."""

import os

import pandas as pd
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import (
    CMIknn,
)
from tigramite.data_processing import DataFrame


def issue38():
    dpath = os.path.dirname(os.path.abspath(__file__))
    fname = 'tigramite_issue_38_input_example.csv'
    fpath = os.path.join(dpath, fname)
    df = pd.read_csv(fpath, index_col=0)
    print(df)
    data = df.values

    tdf = DataFrame(
        data=data,
        mask=None,
        missing_flag=None,
        var_names=df.columns,
        datatime=None,
    )

    indp_test = CMIknn(
        # knn=None,
        # shuffle_neighbors=None,
        # transform=None,
        # significance=None,
    )

    selected_variables = [
        col_lbl for col_lbl in df.columns
        if 'i' in col_lbl
    ]
    selected_variables_ix = [
        df.columns.get_loc(lbl) for lbl in selected_variables]
    print((
        "Init PCMCI with:"
        f"dataframe={tdf},"
        f"cond_ind_test={indp_test},"
        f"selected_variables={selected_variables_ix},"
        f"verbosity=10,"
    ))
    pcmci = PCMCI(
        dataframe=tdf,
        cond_ind_test=indp_test,
        selected_variables=selected_variables_ix,
        verbosity=10,
    )

    max_lag = 24
    alpha = 0.1

    print("Running PCMCI...")
    pcmci.run_pcmci(tau_max=max_lag, pc_alpha=alpha)

    print("Done successfully! No errors!")


if __name__ == '__main__':
    issue38()
