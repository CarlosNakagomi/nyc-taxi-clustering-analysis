"""
Synthetic NYC yellow taxi–like trips for end-to-end reproducibility when
no public TLC files are available. All coordinates lie in a realistic NYC
bounding box; no Streamlit, no real TLC rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_tlc_cohort(
    n: int = 4_000,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Produces a table with TLC-style column names, suitable for the same
    cleaning pipeline as the real 2016 yellow schema.
    """
    rng = np.random.default_rng(random_state)

    # “Modes”: Manhattan, JFK/Queens, other NYC (borough mix)
    mode = rng.integers(0, 3, n)
    plat = np.empty(n, dtype=np.float64)
    plon = np.empty(n, dtype=np.float64)
    m = mode == 0
    plat[m] = rng.uniform(40.72, 40.80, m.sum())  # Manhattan
    plon[m] = rng.uniform(-74.01, -73.96, m.sum())
    m = mode == 1
    plat[m] = rng.uniform(40.63, 40.66, m.sum())  # JFK
    plon[m] = rng.uniform(-73.82, -73.75, m.sum())
    m = mode == 2
    plat[m] = rng.uniform(40.55, 40.80, m.sum())  # broader
    plon[m] = rng.uniform(-74.05, -73.80, m.sum())

    dlat = rng.normal(0, 0.04, n)
    dlon = rng.normal(0, 0.04, n)
    dlat = np.clip(dlat, -0.2, 0.2)
    dlon = np.clip(dlon, -0.2, 0.2)
    dlat_p = np.clip(plat + dlat, 40.49, 40.92)
    dlon_p = np.clip(plon + dlon, -74.25, -73.70)
    pax = rng.integers(1, 4, n)
    dist = rng.exponential(1.2, n) * (1.0 + 0.15 * pax)  # noqa: S311
    dist = np.clip(dist, 0.1, 35.0)
    fare = np.clip(2.5 + 2.0 * dist + rng.normal(0, 0.4, n), 2.0, 250)  # noqa: S311
    extra_ = np.clip(
        rng.choice(np.array([0.0, 0.0, 0.5, 1.0, 2.0]), size=n, replace=True),
        0,
        50,
    )
    mta = np.full(n, 0.5)
    base_tip = (fare * rng.uniform(0.05, 0.2, n)).clip(0.1, 50)  # noqa: S311
    tip = np.where(rng.random(n) < 0.25, fare * 0.02, base_tip)  # some low-tip “corporate”
    tip = np.clip(tip, 0.1, 50.0)
    # Cleaning rules require > 0; use 0.5 for non-toll trips, higher when “airport/bridge”
    tolls = np.where(  # noqa: E501
        rng.random(n) < 0.1,  # noqa: S311
        rng.uniform(2, 5, n),  # noqa: S311
        0.5,  # noqa: E501
    )
    tolls = np.clip(tolls, 0.1, 50.0)  # noqa: E501
    imp = np.clip(rng.choice([0.0, 0.3], n, p=[0.9, 0.1]), 0, 0.3)  # noqa: S311
    total_ = (
        np.clip(
            fare + extra_ + mta + tip + tolls + imp, 0.0, 300.0
        )  # noqa: S112
    )

    t0 = pd.Timestamp("2016-01-01 00:00:00")
    t1 = pd.Timestamp("2016-03-30 23:59:59")
    span = (t1 - t0).total_seconds() * rng.random(n)
    dts = pd.Series(t0 + pd.to_timedelta(span, unit="s"), dtype="datetime64[ns]")
    # Slight shift toward 12:00–17:00
    dts = dts + pd.to_timedelta(
        np.where(
            (dts.dt.hour < 6) & (rng.random(n) < 0.35), 10, 0
        ),
        unit="h",
    )
    ptype = rng.integers(1, 6, n)

    df = pd.DataFrame(
        {
            "passenger_count": pax,
            "trip_distance": np.round(dist, 2),
            "pickup_longitude": np.round(plon, 5),
            "pickup_latitude": np.round(plat, 5),
            "dropoff_longitude": np.round(dlon_p, 5),
            "dropoff_latitude": np.round(dlat_p, 5),
            "fare_amount": np.round(fare, 2),
            "extra": np.round(extra_, 2),
            "mta_tax": mta,
            "tip_amount": np.round(tip, 2),
            "tolls_amount": np.round(tolls, 2),
            "improvement_surcharge": np.round(imp, 2),
            "total_amount": np.round(total_, 2),
            "payment_type": ptype,
            "tpep_pickup_datetime": dts,
        }
    )
    return df
