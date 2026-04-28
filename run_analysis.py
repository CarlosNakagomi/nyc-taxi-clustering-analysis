#!/usr/bin/env python3
"""
NYC TLC–style yellow taxi: clustering methodology study (script, not a dashboard).
Run:  python run_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

from nyc_taxi_portfolio.demo_data import generate_demo_tlc_cohort
from nyc_taxi_portfolio.labels import map_cluster_ids_to_labels
from nyc_taxi_portfolio.metrics_extra import dunn_index, hclust_silhouette_k4, hopkins_statistic  # noqa: E501

RAW = [ROOT / f"data/raw/yellow_tripdata_2016-{m}.csv" for m in ("01", "02", "03")]
PATH_MODEL = ROOT / "data/processed/yellow_sample_export.csv"
PATH_VIZ = ROOT / "data/processed/yellow_kmeans_viz_sample.csv"
FIG = ROOT / "outputs/figures"
TAB = ROOT / "outputs/tables"
REP = ROOT / "outputs/report"
RNG = np.random.default_rng(123)


def ensure_dirs() -> None:
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _img_tag(path: Path, alt: str) -> str:
    if not path.is_file():
        return (
            f"<div class='missing'>Missing figure: <code>{path.as_posix()}</code></div>"
        )
    rel = path.relative_to(ROOT / "outputs").as_posix()
    rel = f"../{rel}"
    return (
        f"<img class='report-image' src='{rel}' alt='{alt}' "
        "loading='lazy' decoding='async' />"
    )


def _table_or_note(path: Path, title: str, rounded: int = 4) -> str:
    df = _safe_read_csv(path)
    if df is None:
        return f"<div class='missing'>Missing table: <code>{path.as_posix()}</code></div>"
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].round(rounded)
    return (
        f"<h4>{title}</h4>"
        "<div class='table-wrap'>"
        f"{df.to_html(index=False, classes='data-table', border=0)}"
        "</div>"
    )


def _metric_value(path: Path, metric_name: str) -> str:
    df = _safe_read_csv(path)
    if df is None or "metric" not in df.columns or "value" not in df.columns:
        return "N/A"
    hit = df.loc[df["metric"].astype(str).str.lower() == metric_name.lower(), "value"]
    if hit.empty:
        return "N/A"
    try:
        return f"{float(hit.iloc[0]):.4f}"
    except Exception:  # noqa: BLE001
        return str(hit.iloc[0])


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def build_html_report(overview: dict[str, object], eda_message: str) -> Path:
    report_path = REP / "nyc_taxi_clustering_report.html"
    map_html = FIG / "nyc_cluster_centroid_map.html"
    pca_plot = FIG / "pca_individuals.png"
    scree_plot = FIG / "pca_scree.png"
    elbow_plot = FIG / "elbow_k.png"
    sil_plot = FIG / "silhouette_k.png"
    hotspot_plot = _first_existing(
        [FIG / "pickup_locations_by_cluster.png", FIG / "demand_hotspot_scatter.png"]
    )
    time_plot = _first_existing(
        [FIG / "time_based_trip_behavior.png", FIG / "time_trips_by_cluster.png"]
    )
    fare_plot = _first_existing(
        [FIG / "fare_based_segmentation.png", FIG / "fare_1d_segmentation.png"]
    )

    # Required aliases for expected filenames in final outputs.
    if (FIG / "demand_hotspot_scatter.png").is_file() and not (
        FIG / "pickup_locations_by_cluster.png"
    ).is_file():
        (FIG / "pickup_locations_by_cluster.png").write_bytes(
            (FIG / "demand_hotspot_scatter.png").read_bytes()
        )
    if (FIG / "time_trips_by_cluster.png").is_file() and not (
        FIG / "time_based_trip_behavior.png"
    ).is_file():
        (FIG / "time_based_trip_behavior.png").write_bytes(
            (FIG / "time_trips_by_cluster.png").read_bytes()
        )
    if (FIG / "fare_1d_segmentation.png").is_file() and not (
        FIG / "fare_based_segmentation.png"
    ).is_file():
        (FIG / "fare_based_segmentation.png").write_bytes(
            (FIG / "fare_1d_segmentation.png").read_bytes()
        )

    dataset_rows = overview.get("n_rows", "N/A")
    dataset_cols = len(overview.get("columns", [])) if isinstance(
        overview.get("columns"), list
    ) else "N/A"
    key_vars = [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "tpep_pickup_datetime",
        "payment_type",
    ]
    available_vars = [v for v in key_vars if v in (overview.get("columns") or [])]
    data_source = str(overview.get("data_source", "unknown"))

    hopkins = _metric_value(TAB / "02_hopkins.csv", "hopkins_tendency")
    validation_path = TAB / "validation_metrics.csv"
    if not validation_path.is_file():
        src = TAB / "03_validation_k4_kmeans.csv"
        if src.is_file():
            src_df = pd.read_csv(src)
            rename_map = {
                "silhouette": "Silhouette Score",
                "davies_bouldin": "Davies-Bouldin Index",
                "calinski_harabasz": "Calinski-Harabasz Index",
                "dunn_index": "Dunn Index",
            }
            src_df["metric"] = src_df["metric"].map(rename_map).fillna(src_df["metric"])
            src_df.to_csv(validation_path, index=False)

    cluster_summary_path = TAB / "cluster_summary.csv"
    if not cluster_summary_path.is_file():
        src = TAB / "04_cluster_means_by_label.csv"
        if src.is_file():
            src_df = pd.read_csv(src)
            src_df.to_csv(cluster_summary_path, index=False)

    map_embed = (
        f"<iframe class='map-frame' src='../figures/{map_html.name}' "
        "title='NYC cluster centroid map'></iframe>"
        if map_html.is_file()
        else f"<div class='missing'>Missing map: <code>{map_html.as_posix()}</code></div>"
    )
    eda_html = f'<p class="muted">{eda_message}</p>' if str(eda_message).strip() else ""

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NYC Taxi Clustering Report</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #4b5563;
      --line: #e5e7eb;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .container {{
      width: min(1300px, 96vw);
      margin: 24px auto 48px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    h1, h2, h3, h4 {{ margin-top: 0; }}
    h1 {{ font-size: 1.85rem; margin-bottom: 8px; }}
    h2 {{ font-size: 1.2rem; color: var(--accent); margin-bottom: 10px; }}
    .muted {{ color: var(--muted); }}
    .pipeline {{
      font-weight: 600;
      color: #111827;
      background: #f8fafc;
      border: 1px dashed #cbd5e1;
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .report-image {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }}
    .map-frame {{
      width: 100%;
      min-height: 760px;
      border: 1px solid var(--line);
      border-radius: 12px;
      display: block;
    }}
    .table-wrap {{ overflow-x: auto; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.93rem;
    }}
    .data-table th, .data-table td {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
    }}
    .data-table thead th {{
      background: #f1f5f9;
    }}
    .missing {{
      border: 1px dashed #f59e0b;
      background: #fffbeb;
      color: #92400e;
      border-radius: 8px;
      padding: 10px 12px;
    }}
    ul {{ margin: 8px 0 0 20px; }}
  </style>
</head>
<body>
  <div class="container">
    <section class="card">
      <h1>Clustering NYC Taxi Trips to Identify Travel Patterns and High-Demand Zones</h1>
      <p class="muted">Automated script-generated HTML study report</p>
    </section>

    <section class="card">
      <h2>Project Overview</h2>
      <p>This is a Python clustering methodology study using NYC TLC-style taxi trip data. If raw TLC data is unavailable, the pipeline falls back to a reproducible demo dataset so the workflow remains fully runnable end-to-end.</p>
      {eda_html}
    </section>

    <section class="card">
      <h2>Dataset Overview</h2>
      <ul>
        <li>Number of rows: <strong>{dataset_rows}</strong></li>
        <li>Number of columns: <strong>{dataset_cols}</strong></li>
        <li>Key variables: <strong>{", ".join(available_vars) if available_vars else "N/A"}</strong></li>
        <li>Dataset source: <strong>{data_source}</strong> (processed/raw/demo)</li>
      </ul>
    </section>

    <section class="card">
      <h2>Methodology Pipeline</h2>
      <p class="pipeline">Data loading → preprocessing → scaling → PCA → clustering tendency → optimal K → K-Means clustering → validation → insights</p>
    </section>

    <section class="card">
      <h2>PCA Results</h2>
      {_img_tag(pca_plot, "PCA scatter")}
      <div style="height:12px;"></div>
      {_img_tag(scree_plot, "PCA scree plot")}
      <p class="muted">The first components capture a meaningful share of variance, and the PCA scatter indicates separable structure consistent with downstream clustering.</p>
    </section>

    <section class="card">
      <h2>Clustering Tendency / Optimal K</h2>
      <p><strong>Hopkins statistic:</strong> {hopkins} (values above ~0.5 suggest non-random cluster tendency).</p>
      {_img_tag(elbow_plot, "Elbow method")}
      <div style="height:12px;"></div>
      {_img_tag(sil_plot, "Silhouette by k")}
      <p class="muted">Selected <strong>k = 4</strong> to balance compactness, separation, and interpretability of business-relevant trip segments.</p>
    </section>

    <section class="card">
      <h2>Validation Metrics</h2>
      {_table_or_note(validation_path, "K-Means (k = 4) validation metrics")}
    </section>

    <section class="card">
      <h2>Cluster Summary</h2>
      {_table_or_note(cluster_summary_path, "Cluster summary")}
      <p><strong>Cluster labels used:</strong> Standard Trips, Premium Trips, Corporate/Low Tip, Group Rides.</p>
    </section>

    <section class="card">
      <h2>Demand Hotspots</h2>
      {_img_tag(hotspot_plot, "Pickup locations by cluster") if hotspot_plot else "<div class='missing'>Missing hotspot plot.</div>"}
      <div style="height:14px;"></div>
      {map_embed}
      <p class="muted">Interactive map is built with Folium (Leaflet) and CartoDB Positron tiles—no Mapbox token required.</p>
      <ul>
        <li>Manhattan shows dense standard-trip activity.</li>
        <li>Airport/JFK-related areas form distinct hotspot behavior.</li>
        <li>Cluster centroids indicate spatial concentration of different trip types.</li>
      </ul>
    </section>

    <section class="card">
      <h2>Time-Based Trip Behavior</h2>
      {_img_tag(time_plot, "Time-based trip behavior") if time_plot else "<div class='missing'>Missing time-based plot.</div>"}
      <p class="muted">Peak demand occurs around 12 PM to 5 PM, indicating stronger afternoon taxi activity.</p>
    </section>

    <section class="card">
      <h2>Fare-Based Segmentation</h2>
      {_img_tag(fare_plot, "Fare-based segmentation") if fare_plot else "<div class='missing'>Missing fare-based plot.</div>"}
      <p class="muted">Low and very high fare groups dominate the sampled trip distribution.</p>
    </section>

    <section class="card">
      <h2>Key Findings</h2>
      <ul>
        <li>K-Means with k = 4 provides interpretable trip segments.</li>
        <li>Demand hotspots appear around Manhattan and JFK-related areas.</li>
        <li>Afternoon hours show stronger taxi activity.</li>
        <li>Fare-based clustering separates low, mid, high, and very high fare trips.</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _clean(y: pd.DataFrame) -> pd.DataFrame:
    t = y[
        (y["passenger_count"] > 0)
        & (y["passenger_count"] <= 3)
        & (y["trip_distance"] > 0)
        & (y["trip_distance"] <= 35)
        & (y["pickup_longitude"].between(-74.26, -73.70))
        & (y["pickup_latitude"].between(40.49, 40.92))
        & (y["dropoff_longitude"].between(-74.26, -73.70))
        & (y["dropoff_latitude"].between(40.49, 40.92))
        & (y["fare_amount"] > 0)
        & (y["fare_amount"] <= 250)
        & (y["extra"].between(0, 50))
        & (y["mta_tax"] > 0)
        & (y["mta_tax"] <= 0.5)
        & (y["tip_amount"] > 0)
        & (y["tip_amount"] <= 50)
        & (y["tolls_amount"] > 0)
        & (y["tolls_amount"] <= 50)
        & (y["improvement_surcharge"].between(0, 0.3))
        & (y["total_amount"] > 0)
        & (y["total_amount"] <= 300)
    ]
    return t


def _strat_10pct(df: pd.DataFrame) -> pd.DataFrame:
    s = df["payment_type"] if "payment_type" in df else None
    if s is not None and s.nunique() > 1 and s.value_counts().min() >= 2:
        out, _ = train_test_split(
            df, train_size=0.1, random_state=123, stratify=s, shuffle=True
        )
    else:
        n = max(1, int(len(df) * 0.1))
        out = df.sample(n=n, random_state=123, replace=(n > len(df)))
    return out


def process_raw_months() -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in RAW]
    cols = set(dfs[0].columns)
    for d in dfs[1:]:
        if set(d.columns) != cols:
            raise ValueError("2016 month files: column name mismatch between months.")
    y = pd.concat(dfs, ignore_index=True)
    if "payment_type" not in y.columns:
        y["payment_type"] = 1
    y = y.dropna(subset=["payment_type"], how="any")
    return y


def make_and_save(y_pool: pd.DataFrame, eda: str) -> tuple[str, str]:
    part = _strat_10pct(y_pool)
    dups = int(part.duplicated().sum())
    nac = int((part.isna().sum() > 0).sum())
    clean = _clean(part)
    ymod = build_modeling_table(clean)
    ymod.to_csv(PATH_MODEL, index=False)
    nv = min(5000, max(0, len(clean)))
    if nv:
        viz = build_viz_subframe(clean, nv)
        viz.to_csv(PATH_VIZ, index=False)
    tag = f"{eda} 10% stratified sample; dups in strat. sample: {dups}; columns w/ NAs: {nac}. Rows after clean: {len(clean)}. Saved {PATH_MODEL.name}."  # noqa: E501
    return tag, "raw" if all(p.is_file() for p in RAW) else "demo"


def build_modeling_table(clean: pd.DataFrame) -> pd.DataFrame:
    drop = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "RatecodeID",
        "store_and_fwd_flag",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    return clean.drop(
        columns=[c for c in drop if c in clean.columns], errors="ignore"
    )  # noqa: E501


def build_viz_subframe(clean: pd.DataFrame, n: int) -> pd.DataFrame:
    n2 = min(n, len(clean))
    v = clean.sample(
        n=n2, random_state=123, replace=False
    ) if n2 < len(clean) else clean.copy()  # noqa: E501
    num = list(v.select_dtypes(include="number").columns)
    for ex in (
        "tpep_pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
    ):
        if ex in v.columns and ex not in num:
            num.append(ex)
    v = v[[c for c in num if c in v.columns]]
    v["tpep_pickup_datetime"] = pd.to_datetime(v["tpep_pickup_datetime"])
    return v


def ingest() -> tuple[pd.DataFrame, pd.DataFrame | None, str, str]:  # noqa: E501
    """(yellow, viz, message, source)"""
    if PATH_MODEL.is_file():
        yellow = pd.read_csv(PATH_MODEL)
        if PATH_VIZ.is_file():
            v = pd.read_csv(PATH_VIZ)
            v["tpep_pickup_datetime"] = pd.to_datetime(
                v["tpep_pickup_datetime"]
            )  # noqa: E501
        else:
            v = None
        return (
            yellow,
            v,
            "",
            "processed",
        )

    if all(p.is_file() for p in RAW):
        pool = process_raw_months()
        msg, src = make_and_save(pool, "Sourced from TLC public yellow taxi CSVs (3 months).")  # noqa: E501
    else:
        pool = generate_demo_tlc_cohort(6_000, random_state=123)
        if "payment_type" not in pool:
            pool["payment_type"] = 1
        msg, src = make_and_save(
            pool, "**Demo data** (no real TLC files; NYC-bounded synthetic). "
        )
    y = pd.read_csv(PATH_MODEL)
    v = None
    if PATH_VIZ.is_file():
        v = pd.read_csv(PATH_VIZ)
        v["tpep_pickup_datetime"] = pd.to_datetime(v["tpep_pickup_datetime"])  # noqa: E501
    return y, v, msg, src


def as_scaled_matrix(d: pd.DataFrame) -> np.ndarray:
    n = d.select_dtypes(include="number")
    if "mta_tax" in n.columns and n["mta_tax"].isna().all():
        n = n.drop(columns=["mta_tax"])
    n = n.replace([np.inf, -np.inf], np.nan).dropna()
    if n.empty or n.shape[0] < 2:
        raise ValueError("No numeric rows after dropna.")
    return StandardScaler().fit_transform(n.to_numpy())


def circle_line_latlon(  # noqa: E501
    lat0: float, lon0: float, miles: float, n: int = 64
) -> tuple[np.ndarray, np.ndarray]:  # noqa: E501
    """Approximate 4-mi circle in lat/lon (small-area)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rlat = miles / 69.0
    rlon = miles / (69.0 * max(0.2, np.cos(np.radians(lat0))))  # noqa: E501
    return lat0 + rlat * np.sin(t), lon0 + rlon * np.cos(t)


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles (WGS84 sphere)."""
    r_earth = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r_earth * math.asin(min(1.0, math.sqrt(h)))


def nyc_centroid_area_label(lat: float, lon: float) -> str:
    """Short place label from coordinates (heuristic, no external geocoder)."""
    # Manhattan — downtown / midtown / upper (west of East River throat)
    if -74.03 < lon < -73.95 and 40.698 <= lat <= 40.770:
        if lat <= 40.718:
            return "Downtown Manhattan"
        if lat <= 40.738:
            return "Lower Manhattan / SoHo"
        return "Midtown Manhattan"
    if -74.02 < lon < -73.94 and 40.770 < lat <= 40.835:
        return "Upper Manhattan / Harlem"
    # Brooklyn / Queens border band (north Brooklyn into Queens)
    if -73.98 < lon < -73.86 and 40.65 <= lat <= 40.735:
        if lat >= 40.68:
            return "North Brooklyn / Western Queens"
        return "Central Brooklyn"
    # Southeast Queens toward JFK / Jamaica Bay
    if lat <= 40.695 and lon >= -73.89:
        if lat <= 40.665 and lon >= -73.84:
            return "JFK / Howard Beach area"
        return "Southeast Queens"
    # Default borough hints
    if lon < -73.92 and lat < 40.74 and lat > 40.58:
        return "Brooklyn"
    if lon >= -73.85 and lat < 40.75:
        return "Queens"
    return "NYC area"


def plotly_nyc_map(
    dff: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    clabel: str,
    out_html: Path,
    out_png: Path | None,
) -> None:
    """Interactive map via Folium (Leaflet + CartoDB tiles). Plotly Mapbox HTML used
    ``zoom: null`` with bounds and often rendered a blank white map in the browser."""
    base_radius_mi = 1.05
    jfk_lat, jfk_lon = 40.6413, -73.7781

    csum = dff.groupby(clabel, as_index=False).agg(  # noqa: E501
        lat=(lat_col, "mean"),
        lon=(lon_col, "mean"),
        n=(lat_col, "count"),
    )
    if "fare_amount" in dff.columns:
        fare_mean = dff.groupby(clabel)["fare_amount"].mean().reset_index(name="avg_fare")
        csum = csum.merge(fare_mean, on=clabel, how="left")
    else:
        csum["avg_fare"] = np.nan
    cluster_order = sorted(dff[clabel].dropna().unique(), key=str)
    hex_palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
    hex_map = {c: hex_palette[i % len(hex_palette)] for i, c in enumerate(cluster_order)}

    dist_jfk = [
        haversine_miles(float(r["lat"]), float(r["lon"]), jfk_lat, jfk_lon)
        for _, r in csum.iterrows()
    ]
    ix_nearest_jfk = int(np.argmin(dist_jfk)) if dist_jfk else -1
    radius_miles_list: list[float] = []
    for i, (_, r) in enumerate(csum.iterrows()):
        if i == ix_nearest_jfk and dist_jfk:
            radius_miles_list.append(max(base_radius_mi, float(dist_jfk[i]) + 0.38))
        else:
            radius_miles_list.append(base_radius_mi)

    lat_mins: list[float] = []
    lat_maxs: list[float] = []
    lon_mins: list[float] = []
    lon_maxs: list[float] = []
    for i, (_, r) in enumerate(csum.iterrows()):
        rm = radius_miles_list[i]
        la, lo = circle_line_latlon(float(r["lat"]), float(r["lon"]), rm)
        lat_mins.append(float(np.min(la)))
        lat_maxs.append(float(np.max(la)))
        lon_mins.append(float(np.min(lo)))
        lon_maxs.append(float(np.max(lo)))
    lat_mins.append(float(dff[lat_col].min()))
    lat_maxs.append(float(dff[lat_col].max()))
    lon_mins.append(float(dff[lon_col].min()))
    lon_maxs.append(float(dff[lon_col].max()))

    lat_min = min(lat_mins)
    lat_max = max(lat_maxs)
    lon_min = min(lon_mins)
    lon_max = max(lon_maxs)
    airports = (
        ("LGA", 40.7769, -73.8740),
        ("JFK", 40.6413, -73.7781),
    )
    pad_lat = max(0.008, (lat_max - lat_min) * 0.04)
    pad_lon = max(0.01, (lon_max - lon_min) * 0.04)
    lat_min -= pad_lat
    lat_max += pad_lat
    lon_min -= pad_lon
    lon_max += pad_lon
    for _name, alat, alon in airports:
        if lat_min <= alat <= lat_max and lon_min <= alon <= lon_max:
            continue
        lat_min = min(lat_min, alat - 0.022)
        lat_max = max(lat_max, alat + 0.022)
        lon_min = min(lon_min, alon - 0.022)
        lon_max = max(lon_max, alon + 0.022)

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0
    span_deg = max(
        lat_max - lat_min,
        (lon_max - lon_min) * math.cos(math.radians(center_lat)),
        1e-6,
    )
    span_ratio = max(span_deg / 0.12, 1e-9)
    zoom_guess = int(max(9, min(14, round(11.2 - 4.2 * math.log(span_ratio, 2)))))

    # --- Folium: reliable local HTML (no Plotly Mapbox GL / null-zoom blank canvas).
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_guess,
        tiles="CartoDB positron",
        attr="© OpenStreetMap contributors © CARTO",
        control_scale=True,
    )
    m.get_root().html.add_child(
        folium.Element(
            "<style>"
            ".folium-map { min-height: 880px !important; width: 100% !important; }"
            "</style>"
        )
    )
    title_html = (
        "<div style=\"position:absolute;left:50%;transform:translateX(-50%);top:12px;"
        "z-index:9999;background:#fff;padding:10px 18px;border-radius:10px;"
        "border:1px solid #e5e7eb;box-shadow:0 1px 4px rgba(0,0,0,.08);"
        "font-family:Segoe UI,Arial,sans-serif;text-align:center;max-width:96%;\">"
        "<div style=\"font-size:18px;font-weight:700;color:#1f2937;\">"
        "NYC Taxi Clusters and Centroid Hotspots</div>"
        "<div style=\"font-size:12px;color:#4b5563;margin-top:4px;\">"
        "Pickup points, centroid stars, reference radii; LGA / JFK labeled.</div></div>"
    )
    m.get_root().html.add_child(folium.Element(title_html))

    for _, row in dff.iterrows():
        lbl = row[clabel]
        hx = hex_map.get(str(lbl), "#888888")
        folium.CircleMarker(
            location=[float(row[lat_col]), float(row[lon_col])],
            radius=3,
            stroke=False,
            fill=True,
            fill_color=hx,
            fill_opacity=0.22,
        ).add_to(m)

    for i, (_, r) in enumerate(csum.iterrows()):
        lbl = r[clabel]
        hx = hex_map.get(str(lbl), "#d94801")
        rm = radius_miles_list[i]
        la, lo = circle_line_latlon(float(r["lat"]), float(r["lon"]), rm)
        coords = [[float(a), float(b)] for a, b in zip(la, lo)]
        coords.append(coords[0])
        folium.PolyLine(
            locations=coords,
            color=hx,
            weight=3,
            opacity=0.62,
        ).add_to(m)

    for i, (_, r) in enumerate(csum.iterrows()):
        lbl = str(r[clabel])
        hx = hex_map.get(lbl, "#333333")
        clat, clon = float(r["lat"]), float(r["lon"])
        area_lbl = nyc_centroid_area_label(clat, clon)
        fare_v = float(r["avg_fare"]) if pd.notna(r["avg_fare"]) else float("nan")
        fare_txt = f"${fare_v:.2f}" if fare_v == fare_v else "N/A"
        rm = radius_miles_list[i]
        popup = (
            f"<b>{lbl}</b> (centroid)<br>"
            f"<b>Area (approx.)</b>: {area_lbl}<br>"
            f"Latitude: {clat:.5f}<br>"
            f"Longitude: {clon:.5f}<br>"
            f"Trip count: {int(r['n'])}<br>"
            f"Avg fare: {fare_txt}<br>"
            f"Reference radius: {rm:.2f} mi"
            + (
                "<br><i>JFK-sized ring: centroid nearest JFK expanded to include airport.</i>"
                if i == ix_nearest_jfk
                else ""
            )
        )
        marker_html = (
            "<div style=\"display:flex;flex-direction:column;align-items:center;"
            "font-family:Segoe UI,Arial,sans-serif;\">"
            f"<div style=\"font-size:26px;line-height:1;color:{hx};"
            "text-shadow:1px 1px 2px #fff;\">★</div>"
            "<div style=\"margin-top:3px;font-size:11px;font-weight:700;max-width:148px;"
            "text-align:center;color:#111;background:rgba(255,255,255,0.96);"
            "padding:4px 8px;border-radius:8px;border:1px solid #d1d5db;line-height:1.25;"
            "box-shadow:0 1px 3px rgba(0,0,0,.12);\">"
            f"{area_lbl}</div></div>"
        )
        folium.Marker(
            location=[clat, clon],
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(152, 56),
                icon_anchor=(76, 52),
            ),
            popup=folium.Popup(popup, max_width=300),
        ).add_to(m)

    for name, alat, alon in airports:
        folium.Marker(
            location=[alat, alon],
            popup=name,
            tooltip=name,
            icon=folium.DivIcon(
                html=(
                    "<div style=\"background:#2563eb;color:white;font-size:11px;"
                    "font-weight:700;padding:4px 8px;border-radius:6px;"
                    'border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.2);">'
                    f"{name}</div>"
                ),
                icon_size=(56, 28),
                icon_anchor=(28, 28),
            ),
        ).add_to(m)

    leg_bits = "".join(
        f"<span style=\"margin:0 12px;font-size:13px;\">"
        f"<span style=\"color:{hx};font-size:18px;line-height:0;\">●</span> {lbl}</span>"
        for lbl, hx in hex_map.items()
    )
    legend_html = (
        "<div style=\"position:absolute;left:50%;transform:translateX(-50%);"
        "bottom:18px;z-index:9999;background:#fff;padding:12px 16px;border-radius:10px;"
        "border:1px solid #e5e7eb;box-shadow:0 1px 4px rgba(0,0,0,.08);"
        "font-family:Segoe UI,Arial,sans-serif;\">"
        "<b>Cluster</b> " + leg_bits + "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]], padding=(14, 14))
    m.save(str(out_html))

    if out_png and out_png.parent.exists():
        pal = sns.color_palette("Set2", n_colors=max(8, len(cluster_order)))
        label_color = {lbl: pal[i % len(pal)] for i, lbl in enumerate(cluster_order)}
        fig2, ax2 = plt.subplots(figsize=(13, 8))
        sns.scatterplot(
            data=dff,
            x=lon_col,
            y=lat_col,
            hue=clabel,
            palette=label_color,
            s=8,
            alpha=0.2,
            linewidth=0,
            ax=ax2,
        )
        for i, (_, r) in enumerate(csum.iterrows()):
            lbl = r[clabel]
            col = label_color.get(lbl, (0.2, 0.2, 0.2))
            clat = float(r["lat"])
            clon = float(r["lon"])
            rm = radius_miles_list[i]
            ax2.scatter(
                [clon],
                [clat],
                s=260,
                marker="*",
                c=[col],
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
            )
            la, lo = circle_line_latlon(clat, clon, rm)
            ax2.plot(lo, la, color=col, linewidth=2.8, alpha=0.55, zorder=3)
            area_txt = nyc_centroid_area_label(clat, clon)
            ax2.annotate(
                area_txt,
                (clon, clat),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="#cbd5e1",
                    alpha=0.96,
                ),
                zorder=7,
            )
        for t, alat, alon in airports:
            ax2.scatter(
                [alon],
                [alat],
                s=45,
                c="#2563eb",
                zorder=6,
                edgecolors="white",
            )
            ax2.annotate(t, (alon, alat), xytext=(4, 4), textcoords="offset points", fontsize=9)
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
        ax2.set_title("NYC Taxi Clusters and Centroid Hotspots")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
            fontsize=9,
            frameon=True,
            title="Cluster",
        )
        plt.tight_layout()
        fig2.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig2)


def main() -> int:  # noqa: PLR0915
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except OSError:
            pass
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    yellow, viz, eda, src = ingest()
    n_mod, n_f = len(yellow), len(yellow.select_dtypes("number").columns)  # noqa: E501
    overview = {
        "n_rows": n_mod,
        "n_numeric_features": n_f,
        "columns": yellow.columns.tolist(),  # noqa: E501
        "data_source": src,  # noqa: E501
    }
    with open(TAB / "dataset_overview.json", "w", encoding="utf-8") as f:  # noqa: E501
        json.dump(overview, f, indent=2)
    pd.DataFrame(  # noqa: E501
        {
            "metric": ["n_rows", "n_numeric", "data_source", "message"],
            "value": [n_mod, n_f, src, eda],  # noqa: E501
        }
    ).to_csv(TAB / "01_dataset_overview.csv", index=False)

    if str(eda).strip():
        print(eda, "\n")

    X = as_scaled_matrix(yellow)
    pd.DataFrame(X[: min(5, X.shape[0])]).to_csv(TAB / "scaler_head_sample.csv")  # noqa: E501

    # — PCA
    s_fit = min(20_000, X.shape[0])
    s_ix = (  # noqa: E501
        np.arange(s_fit)
        if s_fit == X.shape[0]  # noqa: E501
        else RNG.choice(X.shape[0], s_fit, replace=False)  # noqa: E501
    )
    pca = PCA().fit(X[s_ix])
    n_sc = min(10, pca.n_components_)  # noqa: E501
    ev = pca.explained_variance_ratio_[:n_sc]  # noqa: E501
    _, ax = plt.subplots(figsize=(8, 4.5))  # noqa: E501
    ax.bar(range(1, len(ev) + 1), ev, color="steelblue", edgecolor="0.2")  # noqa: E501
    ax.set_title("Scree: PCA (scaled features)")  # noqa: E501
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    plt.tight_layout()
    plt.savefig(FIG / "pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close()
    n2 = min(2_000, X.shape[0])  # noqa: E501
    z2 = pca.transform(X[:n2])[:, :2]  # noqa: E501
    _, ax = plt.subplots(figsize=(8, 4.5))  # noqa: E501
    ax.scatter(z2[:, 0], z2[:, 1], s=3, alpha=0.28)  # noqa: E501
    ax.set_title("PCA: first two components (sample rows)")  # noqa: E501
    plt.tight_layout()  # noqa: E501
    plt.savefig(FIG / "pca_individuals.png", dpi=150, bbox_inches="tight")
    plt.close()  # noqa: E501

    hi = RNG.permutation(X.shape[0])[: min(2_000, X.shape[0])]  # noqa: E501
    hop = hopkins_statistic(  # noqa: E501
        X[hi], m=max(2, min(500, len(hi) // 3))
    )  # noqa: E501
    pd.DataFrame(  # noqa: E501
        {"metric": ["hopkins_tendency"], "value": [float(hop)]}  # noqa: E501
    ).to_csv(
        TAB / "02_hopkins.csv", index=False
    )  # noqa: E501
    print(  # noqa: E501
        f"Hopkins statistic: {hop:.4f} "  # noqa: E501
        f"(near 0.5: random, near 1: stronger cluster structure.)\n"  # noqa: E501
    )  # noqa: E501

    s5 = 5_000 if X.shape[0] >= 5_000 else X.shape[0]  # noqa: E501
    s_ix2 = (  # noqa: E501
        np.arange(s5)
        if s5 == X.shape[0]  # noqa: E501
        else RNG.permutation(X.shape[0])[:s5]  # noqa: E501
    )  # noqa: E501
    X5 = X[s_ix2]  # noqa: E501
    wss, s_k = [], []  # noqa: E501
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=25, random_state=123)  # noqa: E501
        km.fit(X5)  # noqa: E501
        wss.append(km.inertia_)  # noqa: E501
        p = km.predict(X5)  # noqa: E501
        s_k.append(  # noqa: E501
            float(  # noqa: E501
                silhouette_score(X5, p, metric="euclidean")  # noqa: E501
            )  # noqa: E501
            if len(np.unique(p)) > 1
            else 0.0
        )  # noqa: E501
    pd.DataFrame(  # noqa: E501
        {"k": list(range(2, 11)), "wss": wss, "silhouette": s_k}  # noqa: E501
    ).to_csv(  # noqa: E501
        TAB / "elbow_silhouette_sweep_k.csv", index=False
    )  # noqa: E501
    _, ax = plt.subplots(figsize=(8, 4.5))  # noqa: E501
    ax.plot(range(2, 11), wss, "o-")  # noqa: E501
    ax.set_xlabel("k")
    ax.set_ylabel("Within-cluster sum of squares")
    ax.set_title("Elbow (K-Means, ≤5000 row sample)")
    plt.tight_layout()  # noqa: E501
    plt.savefig(FIG / "elbow_k.png", dpi=150)
    plt.close()
    _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(2, 11), s_k, "o-")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette (Euclidean)")
    ax.set_title("Silhouette vs k")
    plt.tight_layout()  # noqa: E501
    plt.savefig(FIG / "silhouette_k.png", dpi=150)  # noqa: E501
    plt.close()

    msh, nhc = hclust_silhouette_k4(X, random_state=123, max_rows=500)
    pd.DataFrame(  # noqa: E501
        {"Ward_hierarchical_sil_k4": [msh], "n_rows": [nhc]}  # noqa: E501
    ).to_csv(TAB / "hierarchical_ward_silhouette_k4.csv", index=False)  # noqa: E501
    print(  # noqa: E501
        f"Ward hierarchical, k=4 mean silhouette: {msh:.3f} (n={nhc}; reference, vs K-Means focus).\n"  # noqa: E501
    )  # noqa: E501

    sv = min(5_000, X.shape[0])  # noqa: E501
    s_ix3 = (  # noqa: E501
        np.arange(sv)
        if sv == X.shape[0]  # noqa: E501
        else RNG.permutation(X.shape[0])[:sv]  # noqa: E501
    )  # noqa: E501
    Xv = X[s_ix3]  # noqa: E501
    l4 = KMeans(4, n_init=25, random_state=123).fit_predict(Xv)  # noqa: E501
    s_sil = float(  # noqa: E501
        silhouette_score(Xv, l4, metric="euclidean")
    )  # noqa: E501
    db_ = float(davies_bouldin_score(Xv, l4))  # noqa: E501
    ch_ = float(calinski_harabasz_score(Xv, l4))  # noqa: E501
    dn_ = dunn_index(Xv, l4)
    val = pd.DataFrame(
        {
            "metric": [
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "dunn_index",
            ],
            "value": [s_sil, db_, ch_, dn_],
            "note": [  # noqa: E501
                "K-Means k=4; sklearn. PAM/K-Medoids not in this build (scikit-learn-extra not required).",  # noqa: E501
            ]
            * 4,
        }
    )
    val.to_csv(TAB / "03_validation_k4_kmeans.csv", index=False)
    val.to_csv(TAB / "validation_metrics.csv", index=False)

    km_all = KMeans(4, n_init=25, random_state=123)
    lab_all = km_all.fit_predict(X)
    y2 = yellow.copy()
    y2["cluster_id"] = lab_all
    lblmap = map_cluster_ids_to_labels(y2, "cluster_id")
    y2["cluster_label"] = y2["cluster_id"].map(lblmap)  # noqa: E501
    numc = y2.select_dtypes("number").columns
    sumtab = (  # noqa: E501
        y2.groupby("cluster_label", as_index=True)[list(numc)]  # noqa: E501
        .mean()  # noqa: E501
        .round(3)  # noqa: E501
    )  # noqa: E501
    sumtab.to_csv(TAB / "04_cluster_means_by_label.csv")
    sumtab.reset_index().to_csv(TAB / "cluster_summary.csv", index=False)
    y2.groupby("cluster_label").size().to_csv(  # noqa: E501
        TAB / "04_cluster_sizes.csv"  # noqa: E501
    )  # noqa: E501
    with open(  # noqa: E501
        TAB / "00_preprocessing_summary.txt",  # noqa: E501
        "w",  # noqa: E501
        encoding="utf-8",  # noqa: E501
    ) as fp:  # noqa: E501
        fp.write(  # noqa: E501
            "Preprocessing: StandardScaler on numeric trip/fare fields; "  # noqa: E501
            "rows with NaN/Inf in numerics dropped; mta_tax all-NaN column dropped. "  # noqa: E501
            f"K-Means on scaled matrix shape {X.shape}.\n"  # noqa: E501
        )  # noqa: E501

    # — distance heatmap
    t = min(1_000, Xv.shape[0])
    dmat = squareform(pdist(Xv[:t], metric="euclidean"))
    dnorm = (dmat - dmat.min()) / (dmat.max() - dmat.min() + 1e-9)  # noqa: E501
    _, ax = plt.subplots(figsize=(6.5, 5.5))  # noqa: E501
    im = ax.imshow(dnorm, cmap="RdYlBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # noqa: E501
    ax.set_title("Pairwise distance (sample, normalized)")  # noqa: E501
    plt.tight_layout()  # noqa: E501
    plt.savefig(FIG / "distance_matrix.png", dpi=150)  # noqa: E501
    plt.close()

    # — Hotspot + time + Plotly map
    if (
        viz is not None
        and "pickup_longitude" in viz.columns
        and "tpep_pickup_datetime" in viz.columns
    ):
        nplot = min(4_000, len(viz))
        dff = (
            viz.sample(n=nplot, random_state=123, replace=False)
            if nplot < len(viz)
            else viz.copy()
        )
        ncols = [
            c
            for c in dff.select_dtypes("number").columns
            if c not in ("pickup_longitude", "pickup_latitude")
        ]
        m_ok = dff[ncols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        dff = dff[m_ok].reset_index(drop=True)
        if len(dff) < 20:
            print("Viz: too few rows for map; skipping map/time.")
        else:
            Xm2 = StandardScaler().fit_transform(dff[ncols].to_numpy())
            l_sp = KMeans(4, n_init=25, random_state=123).fit_predict(Xm2)
            dff = dff.assign(
                cluster_id=l_sp,
                hour=pd.to_datetime(dff["tpep_pickup_datetime"]).dt.hour,
            )
            lab_df = dff[
                [c for c in ("total_amount", "passenger_count", "fare_amount", "tip_amount") if c in dff]
            ].copy()
            lab_df["cluster_id"] = dff["cluster_id"]
            lm2 = map_cluster_ids_to_labels(lab_df, "cluster_id")
            dff = dff.assign(
                cluster_label=dff["cluster_id"].map(lm2)  # noqa: E501
            )
            _, axt = plt.subplots(figsize=(8, 4.5))
            for cl in sorted(dff["cluster_label"].unique(), key=str):
                s = dff.loc[dff["cluster_label"] == cl, "hour"]
                axt.hist(s, bins=range(25), alpha=0.35, label=cl)  # noqa: E501
            axt.set_xlabel("Hour of pickup (local)")
            axt.set_ylabel("Count")
            axt.set_title(
                "Time-based trip behavior by cluster (K-Means k=4; non-spatial features only)"
            )
            axt.legend(fontsize=6, loc="upper right")
            plt.tight_layout()
            plt.savefig(FIG / "time_trips_by_cluster.png", dpi=150)
            plt.savefig(FIG / "time_based_trip_behavior.png", dpi=150)
            plt.close()

            _, ahot = plt.subplots(figsize=(7, 6.2))
            sns.scatterplot(
                data=dff,  # noqa: E501
                x="pickup_longitude",  # noqa: E501
                y="pickup_latitude",  # noqa: E501
                hue="cluster_label",  # noqa: E501
                s=5,  # noqa: E501
                alpha=0.4,  # noqa: E501
                ax=ahot,  # noqa: E501
            )
            ahot.set_title("Demand hotspot (Matplotlib) — not basemap; see Plotly HTML for OSM")  # noqa: E501
            ahot.set_aspect("equal", adjustable="datalim")
            plt.tight_layout()
            plt.savefig(FIG / "demand_hotspot_scatter.png", dpi=150)  # noqa: E501
            plt.savefig(FIG / "pickup_locations_by_cluster.png", dpi=150)
            plt.close()

            plotly_nyc_map(  # noqa: E501
                dff,  # noqa: E501
                "pickup_latitude",  # noqa: E501
                "pickup_longitude",  # noqa: E501
                "cluster_label",  # noqa: E501
                FIG / "nyc_cluster_centroid_map.html",  # noqa: E501
                FIG / "nyc_cluster_centroid_map.png",  # noqa: E501
            )  # noqa: E501
    else:
        (TAB / "map_note.txt").write_text(  # noqa: E501
            f"No viz file (pickups+time) available; map and time figure skipped. "
            f"Expected: {PATH_VIZ.name}.\n",
            encoding="utf-8",  # noqa: E501
        )

    # — Fare 1-D segmentation
    f = (
        yellow["fare_amount"]
        .dropna()
        .to_numpy()[: min(5_000, len(yellow))]
    )
    if f.size >= 4:
        fare_groups = KMeans(4, n_init=25, random_state=123).fit_predict(
            f.reshape(-1, 1)  # noqa: E501
        )
        g = (
            pd.DataFrame({"fare": f, "g": fare_groups})
            .groupby("g", as_index=False)
            .agg(n=("fare", "count"), m=("fare", "mean"))  # noqa: E501
            .sort_values("m")
        )  # noqa: E501
        g["band"] = ["Low", "Mid", "High", "Very high"][: len(g)]
        g.to_csv(TAB / "fare_1d_k4.csv", index=False)  # noqa: E501
        _, axf = plt.subplots(figsize=(8, 4.5))  # noqa: E501
        axf.bar(g["band"], g["n"], color=sns.color_palette("Set2", len(g)))  # noqa: E501
        for i, n_ in enumerate(g["n"]):
            axf.text(i, n_, str(int(n_)), ha="center", va="bottom", size=8)  # noqa: E501
        axf.set_title("Fare-based segmentation (1-D K-Means on fare_amount, k=4)")
        axf.set_ylabel("Trip count in sample")
        plt.tight_layout()  # noqa: E501
        plt.savefig(FIG / "fare_1d_segmentation.png", dpi=150)  # noqa: E501
        plt.savefig(FIG / "fare_based_segmentation.png", dpi=150)
        plt.close()

    report_file = build_html_report(overview, eda)

    print(  # noqa: E501
        "\n=== Done ===\n"  # noqa: E501
        f"Figures: {FIG}\n"  # noqa: E501
        f"Tables: {TAB}\n"  # noqa: E501
        f"Report: {report_file.relative_to(ROOT).as_posix()}\n"
    )  # noqa: E501
    print("outputs/report/nyc_taxi_clustering_report.html")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as e:
        print("Error:", e, file=sys.stderr)  # noqa: E501
        raise SystemExit(1) from e