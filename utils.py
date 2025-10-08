import polars as pl
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import cdist

min_, max_ = -30, 30

def clean_lidar_points(
    df: pl.DataFrame, iqr_factor: float = 2.0, verbose: bool = True
) -> pl.DataFrame:
    df_nodup = df.unique(subset=["x", "y", "z"])

    q = df_nodup.select(
        [
            pl.col("x").quantile(0.25).alias("x_q25"),
            pl.col("x").quantile(0.75).alias("x_q75"),
            pl.col("y").quantile(0.25).alias("y_q25"),
            pl.col("y").quantile(0.75).alias("y_q75"),
            pl.col("z").quantile(0.25).alias("z_q25"),
            pl.col("z").quantile(0.75).alias("z_q75"),
        ]
    ).to_dicts()[0]

    x_iqr = q["x_q75"] - q["x_q25"]
    y_iqr = q["y_q75"] - q["y_q25"]
    z_iqr = q["z_q75"] - q["z_q25"]

    x_min, x_max = q["x_q25"] - iqr_factor * x_iqr, q["x_q75"] + iqr_factor * x_iqr
    y_min, y_max = q["y_q25"] - iqr_factor * y_iqr, q["y_q75"] + iqr_factor * y_iqr
    z_min, z_max = q["z_q25"] - iqr_factor * z_iqr, q["z_q75"] + iqr_factor * z_iqr

    clean_df = df_nodup.filter(
        (pl.col("x").is_between(x_min, x_max))
        & (pl.col("y").is_between(y_min, y_max))
        & (pl.col("z").is_between(z_min, z_max))
    )

    if verbose:
        print(f"Puntos originales: {df.height}")
        print(f"Puntos tras eliminar duplicados y outliers: {clean_df.height}")

    return clean_df


def plot_scene(
    df, color_var, dataset=None, pov=False, colorscale="hot", opacity=0.8, fig=None
):
    if fig is None:
        fig = go.Figure(
            layout=go.Layout(
                width=800,
                height=800,
                scene=dict(
                    xaxis=dict(title="-X", range=[min_, max_]),
                    yaxis=dict(title="Z", range=[min_, max_]),
                    zaxis=dict(title="-Y", range=[min_, max_]),
                ),
            ),
        )

    fig.add_trace(
        go.Scatter3d(
            x=-df["x"],
            y=df["z"],
            z=-df["y"],
            mode="markers",
            name=color_var,
            marker=dict(
                size=1,
                color=df[color_var],
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(title=color_var),
            ),
            hoverinfo="text",
            hovertext=df[color_var],
        )
    )

    if pov:
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                name="Lidar POV",
                marker=dict(size=5, color="Red"),
                hoverinfo="text",
                hovertext="Lidar POV",
            )
        )

    if dataset or fig.layout.title.text == "":
        fig.update_layout(
            showlegend=True,
            title=dict(
                text=f"Lidar Point Cloud{f': {dataset}' if dataset else ''}",
                x=0.5,
                y=0.9,
                xanchor="center",
                yanchor="top",
                font=dict(
                    family="Arial, monospace",
                    size=32,
                    color="Black",
                    variant="small-caps",
                ),
            ),
            font=dict(
                family="Arial, monospace",
                size=12,
                color="Black",
                variant="small-caps",
            ),
        )

    return fig

def create_bounding_box_lines(min_x, max_x, min_y, max_y, min_z, max_z):
    x_lines, y_lines, z_lines = [], [], []

    # Corners of the box
    corners = [
        (min_x, min_y, min_z),
        (min_x, min_y, max_z),
        (min_x, max_y, min_z),
        (min_x, max_y, max_z),
        (max_x, min_y, min_z),
        (max_x, min_y, max_z),
        (max_x, max_y, min_z),
        (max_x, max_y, max_z),
    ]

    # Lines of the box
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    for start, end in edges:
        x_lines.extend([corners[start][0], corners[end][0], None])
        y_lines.extend([corners[start][1], corners[end][1], None])
        z_lines.extend([corners[start][2], corners[end][2], None])

    return x_lines, y_lines, z_lines

def make_voxel_grid(
    df: pl.DataFrame,
    n_regions: int = 120,
    min_value: float = min_,
    max_value: float = max_,
) -> pl.DataFrame:
    voxel_size_x = (max_value - min_value) / n_regions
    voxel_size_y = (max_value - min_value) / n_regions
    voxel_size_z = (max_value - min_value) / n_regions

    static_pts = df.with_columns(
        (((pl.col("x") - min_value) / voxel_size_x).floor().clip(0, n_regions - 1))
        .cast(pl.Int32)
        .alias("vx"),
        (((pl.col("y") - min_value) / voxel_size_y).floor().clip(0, n_regions - 1))
        .cast(pl.Int32)
        .alias("vy"),
        (((pl.col("z") - min_value) / voxel_size_z).floor().clip(0, n_regions - 1))
        .cast(pl.Int32)
        .alias("vz"),
    )
    count = static_pts.group_by(["vx", "vy", "vz"]).len().rename({"len": "count"})
    # voxel = np.zeros((n_regions, n_regions, n_regions), dtype=np.uint16)
    # voxel[count["vx"], count["vy"], count["vz"]] = count["count"]
    return count


def mark_static_points(
    df_scene: pl.DataFrame,
    static_voxels: pl.DataFrame,
    n_regions: int,
    min_value: float = min_,
    max_value: float = max_,
) -> pl.DataFrame:
    voxel_size_x = (max_value - min_value) / n_regions
    voxel_size_y = (max_value - min_value) / n_regions
    voxel_size_z = (max_value - min_value) / n_regions

    df_scene = df_scene.with_columns(
        (((pl.col("x") - min_value) / voxel_size_x).floor().clip(0, n_regions - 1))
        .cast(pl.Int32)
        .alias("vx"),
        (((pl.col("y") - min_value) / voxel_size_y).floor().clip(0, n_regions - 1))
        .cast(pl.Int32)
        .alias("vy"),
        (((pl.col("z") - min_value) / voxel_size_z).floor().clip(0, n_regions - 1))
        .cast(pl.Int32)
        .alias("vz"),
    )

    static_flag = static_voxels.with_columns(pl.lit(-2).alias("cluster"))

    df_scene = df_scene.join(
        static_flag.select(["vx", "vy", "vz", "cluster"]),
        on=["vx", "vy", "vz"],
        how="left",
    ).fill_null(pl.lit(-1))

    return df_scene

def clusters_with_dbscan(
    df: pl.DataFrame,
    feature_cols: list[str],
    eps: float = 0.7,
    min_samples: int = 12,
) -> pl.DataFrame:
    df = df.with_row_index(name="id")
    subset = df.filter(pl.col("cluster") != -2)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
        subset.select(feature_cols).to_numpy()
    )
    subset = subset.with_columns(pl.Series("cluster", labels))
    df = (
        df.join(subset.select(["id", "cluster"]), on="id", how="left")
        .with_columns(
            pl.when(pl.col("cluster_right").is_not_null())
            .then(pl.col("cluster_right"))
            .otherwise(pl.col("cluster"))
            .alias("cluster")
        )
        .drop("cluster_right")
    )
    return df

def compute_centroids(points: np.ndarray, labels: np.ndarray) -> dict:
    centroids = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        cluster_points = points[labels == cluster_id]
        min_x, min_y = cluster_points[:, 0].min(), cluster_points[:, 1].min()
        max_x, max_y = cluster_points[:, 0].max(), cluster_points[:, 1].max()
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        centroids[cluster_id] = np.array([cx, cy])
    return centroids


def match_clusters(
    prev_centroids: dict, new_centroids: dict, threshold: float = 1.0
) -> dict:
    matches = {}
    if not prev_centroids or not new_centroids:
        return matches

    prev_ids = list(prev_centroids.keys())
    new_ids = list(new_centroids.keys())
    prev_points = np.array([prev_centroids[i] for i in prev_ids])
    new_points = np.array([new_centroids[i] for i in new_ids])

    dist_matrix = cdist(new_points, prev_points)
    for i, new_id in enumerate(new_ids):
        min_idx = np.argmin(dist_matrix[i])
        if dist_matrix[i, min_idx] <= threshold:
            matches[new_id] = prev_ids[min_idx]
        else:
            matches[new_id] = None
    return matches


def compute_velocities(
    prev_centroids: dict, new_centroids: dict, matches: dict, dt: float = 1.0
) -> dict:
    velocities = {}
    for new_id, prev_id in matches.items():
        if prev_id is None:
            continue
        prev_pos = prev_centroids[prev_id]
        new_pos = new_centroids[new_id]
        vx, vy = (new_pos - prev_pos) / dt
        velocities[new_id] = np.array([vx, vy])
    return velocities


def predict_positions(
    current_centroids: dict, velocities: dict, dt: float = 1.0
) -> dict:
    preds = {}
    for cid, pos in current_centroids.items():
        if cid in velocities:
            preds[cid] = pos + velocities[cid] * dt
        else:
            preds[cid] = pos
    return preds


def update_tracking_state(
    state_df: pl.DataFrame,
    frame_id: int,
    centroids: dict,
    velocities: dict,
    preds: dict,
):
    rows = []
    for cid in centroids:
        cx, cy = centroids[cid]
        vx, vy = velocities.get(cid, (np.nan, np.nan))
        px, py = preds.get(cid, (np.nan, np.nan))
        rows.append((frame_id, cid, cx, cy, vx, vy, px, py))
    new_df = pl.DataFrame(
        rows,
        schema=["frame", "cluster", "cx", "cy", "vx", "vy", "px", "py"],
        orient="row",
    )

    return pl.concat([state_df, new_df])