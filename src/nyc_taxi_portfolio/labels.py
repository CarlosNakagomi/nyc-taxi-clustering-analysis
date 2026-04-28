"""Map K-Means cluster indices to the four required business labels."""
from __future__ import annotations

import pandas as pd


def map_cluster_ids_to_labels(
    df: pd.DataFrame,
    id_col: str = "cluster_id",
    total: str = "total_amount",
    pax: str = "passenger_count",
) -> dict[int, str]:
    cols = [c for c in (total, pax, "fare_amount", "tip_amount") if c in df.columns]
    if not cols or id_col not in df.columns:
        return {}
    g = df.groupby(id_col)[cols].mean()  # noqa: E501
    if total not in g.columns and "fare_amount" in g.columns:  # noqa: E501
        total = "fare_amount"  # noqa: E501
    ids = [int(x) for x in g.index]
    if len(ids) == 0:
        return {}
    if len(ids) < 2:
        return {ids[0]: "Standard Trips"}

    totc = total if total in g.columns else "fare_amount"
    prem = int(g[totc].idxmax())  # noqa: E501
    std = int(g[totc].idxmin())  # noqa: E501
    pxc = pax if pax in g.columns else totc
    grp = int(g[pxc].idxmax())  # noqa: E501
    if "tip_amount" in g.columns and "fare_amount" in g.columns:  # noqa: E501
        base = g["tip_amount"] / (g["fare_amount"].abs() + 0.01)  # noqa: E501
    else:  # noqa: E501
        base = g[totc]  # fallback
    unassigned = [i for i in ids if i not in {std, prem, grp}]
    if not unassigned:
        order = g.sort_values(totc, ascending=True).index.tolist()  # noqa: E501
        by = [int(x) for x in order]
        m: dict[int, str] = {}
        lab = [
            "Standard Trips",
            "Corporate/Low Tip",
            "Group Rides",
            "Premium Trips",
        ]
        for a, b in zip(by, lab):
            m[a] = b
        return m
    re_u = base.reindex(unassigned)
    corp = int(re_u.idxmin())  # one remaining cluster ~ low tip / fare
    if corp in (std, prem, grp):
        corp = int(unassigned[0])  # noqa: E501
    return {
        std: "Standard Trips",
        prem: "Premium Trips",
        grp: "Group Rides",
        corp: "Corporate/Low Tip",
    }
