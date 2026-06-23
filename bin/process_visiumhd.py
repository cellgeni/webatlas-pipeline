#!/usr/bin/env python3
"""
process_visiumhd.py
====================================
Processes Visium HD Space Ranger output
"""

from __future__ import annotations
import json
import os
import fire
import typing as T
import numpy as np
import scanpy as sc
import pandas as pd
import tifffile as tf
from pathlib import Path
from skimage.io import imread
from process_h5ad import h5ad_to_zarr, reindex_anndata_obs, subset_anndata


VISIUMHD_BIN_SIZES = {
    "002": "square_002um",
    "002um": "square_002um",
    "2": "square_002um",
    "2um": "square_002um",
    "square_002um": "square_002um",
    "008": "square_008um",
    "008um": "square_008um",
    "8": "square_008um",
    "8um": "square_008um",
    "square_008um": "square_008um",
    "016": "square_016um",
    "016um": "square_016um",
    "16": "square_016um",
    "16um": "square_016um",
    "square_016um": "square_016um",
}


def _normalise_bin_size(bin_size: str | int = "008um") -> str:
    key = str(bin_size).lower().replace("_", "").replace("-", "")
    if str(bin_size).lower().startswith("square_"):
        key = str(bin_size).lower()
    if key not in VISIUMHD_BIN_SIZES:
        raise ValueError(
            "Unsupported Visium HD bin_size '{}'. Use one of: 002um, 008um, 016um.".format(
                bin_size
            )
        )
    return VISIUMHD_BIN_SIZES[key]


def _resolve_outs_dir(path: str | Path) -> Path:
    p = Path(path)
    if (p / "binned_outputs").is_dir() or (p / "segmented_outputs").is_dir():
        return p
    if (p / "outs" / "binned_outputs").is_dir():
        return p / "outs"
    return p


def _resolve_bin_dir(path: str | Path, bin_size: str | int = "008um") -> Path:
    p = Path(path)
    if p.name.startswith("square_") and (p / "spatial").is_dir():
        return p

    outs_dir = _resolve_outs_dir(p)
    bin_dir = outs_dir / "binned_outputs" / _normalise_bin_size(bin_size)
    if not bin_dir.is_dir():
        raise FileNotFoundError(
            "Could not find Visium HD binned output directory: {}".format(bin_dir)
        )
    return bin_dir


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    parquet_file = spatial_dir / "tissue_positions.parquet"
    csv_file = spatial_dir / "tissue_positions.csv"
    legacy_csv_file = spatial_dir / "tissue_positions_list.csv"

    if parquet_file.is_file():
        positions = pd.read_parquet(parquet_file)
    elif csv_file.is_file():
        positions = pd.read_csv(csv_file)
    elif legacy_csv_file.is_file():
        positions = pd.read_csv(
            legacy_csv_file,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )
    else:
        raise FileNotFoundError(
            "Could not find tissue_positions.parquet or tissue_positions.csv in {}".format(
                spatial_dir
            )
        )

    if "barcode" not in positions.columns:
        positions = positions.reset_index().rename(columns={"index": "barcode"})
    positions["barcode"] = positions["barcode"].astype(str)
    return positions.set_index("barcode")


def _read_spatial_metadata(spatial_dir: Path) -> dict[str, T.Any]:
    metadata: dict[str, T.Any] = {"images": {}, "scalefactors": {}}

    scalefactors_file = spatial_dir / "scalefactors_json.json"
    if scalefactors_file.is_file():
        with open(scalefactors_file, "rt") as f:
            metadata["scalefactors"] = json.load(f)

    for image_name in ["hires", "lowres"]:
        image_file = spatial_dir / f"tissue_{image_name}_image.png"
        if image_file.is_file():
            metadata["images"][image_name] = imread(image_file)

    return metadata


def _add_secondary_analysis(
    adata: sc.AnnData,
    bin_dir: Path,
    load_clusters: bool,
    load_embeddings: bool,
) -> sc.AnnData:
    analysis_dir = bin_dir / "analysis"
    if not analysis_dir.is_dir():
        return adata

    if load_clusters and (analysis_dir / "clustering").is_dir():
        for cluster in [
            d for d in (analysis_dir / "clustering").iterdir() if d.is_dir()
        ]:
            cluster_name = cluster.name.replace("gene_expression_", "")
            cluster_file = cluster / "clusters.csv"
            if not cluster_file.is_file():
                continue
            cluster_df = pd.read_csv(cluster_file, index_col="Barcode")
            clusters = cluster_df.reindex(adata.obs.index)
            adata.obs[cluster_name] = pd.Categorical(clusters["Cluster"])

    if load_embeddings:
        for embedding, components in [
            ("umap", "2_components"),
            ("tsne", "2_components"),
            ("pca", "10_components"),
        ]:
            embedding_dir = analysis_dir / embedding
            if not embedding_dir.is_dir():
                continue
            components_name = (
                components
                if (embedding_dir / components).exists()
                else f"gene_expression_{components}"
            )
            projection_file = embedding_dir / components_name / "projection.csv"
            if not projection_file.is_file():
                continue
            embedding_df = pd.read_csv(projection_file, index_col="Barcode")
            emb = embedding_df.reindex(adata.obs.index)
            adata.obsm[f"X_{embedding}"] = emb.values

    return adata


def _add_annotations(
    adata: sc.AnnData,
    annotations: str = None,
    annotations_column_index: str = None,
) -> sc.AnnData:
    if not annotations:
        return adata

    annot_df = pd.read_csv(annotations)
    if annotations_column_index in annot_df.columns:
        annot_df.set_index(annotations_column_index, inplace=True)
    else:
        annot_df.set_index(annot_df.columns[0], inplace=True)
    annot_df.index = annot_df.index.astype(str)
    adata.obs = pd.merge(
        adata.obs, annot_df, left_index=True, right_index=True, how="left"
    )
    return adata


def visiumhd_to_anndata(
    path: str,
    bin_size: str | int = "008um",
    load_clusters: bool = True,
    load_embeddings: bool = True,
    load_raw: bool = False,
    annotations: str = None,
    annotations_column_index: str = None,
) -> sc.AnnData:
    """Create an AnnData object from a Visium HD Space Ranger output directory.

    Args:
        path (str): Path to a Visium HD Space Ranger outs directory, pipestance
            directory, or a specific binned output directory.
        bin_size (str | int, optional): Bin size to load. Accepted values include
            ``002um``, ``008um`` and ``016um``. Defaults to ``008um``.
        load_clusters (bool, optional): If clustering files should be included in
            the AnnData object when present. Defaults to True.
        load_embeddings (bool, optional): If embedding coordinate files should be
            included in the AnnData object when present. Defaults to True.
        load_raw (bool, optional): If the raw matrix count file should be loaded
            instead of the filtered matrix. Defaults to False.

    Returns:
        AnnData: AnnData object created from Visium HD binned output data.
    """

    bin_dir = _resolve_bin_dir(path, bin_size)
    matrix_file = bin_dir / (
        "raw_feature_bc_matrix.h5" if load_raw else "filtered_feature_bc_matrix.h5"
    )
    if not matrix_file.is_file():
        raise FileNotFoundError("Could not find matrix file: {}".format(matrix_file))

    adata = sc.read_10x_h5(matrix_file)
    adata.var_names_make_unique()

    spatial_dir = bin_dir / "spatial"
    positions = _read_tissue_positions(spatial_dir)
    positions = positions.reindex(adata.obs.index)
    adata.obs = adata.obs.join(positions)
    adata.obsm["spatial"] = positions[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy()

    sample_id = _resolve_outs_dir(path).name
    if sample_id == "outs":
        sample_id = _resolve_outs_dir(path).parent.name
    adata.uns["spatial"] = {sample_id: _read_spatial_metadata(spatial_dir)}
    adata.uns["spatial"][sample_id]["metadata"] = {
        "chemistry_description": "Visium HD",
        "bin_size": _normalise_bin_size(bin_size),
    }

    adata = _add_secondary_analysis(
        adata,
        bin_dir=bin_dir,
        load_clusters=load_clusters,
        load_embeddings=load_embeddings,
    )
    adata = _add_annotations(adata, annotations, annotations_column_index)

    adata.obs.index.names = ["label_id"]
    adata.obs = adata.obs.reset_index()
    adata.obs.index = (pd.Categorical(adata.obs["label_id"]).codes + 1).astype(str)

    return adata


def visiumhd_to_zarr(
    path: str,
    stem: str,
    bin_size: str | int = "008um",
    load_clusters: bool = True,
    load_embeddings: bool = True,
    load_raw: bool = False,
    save_h5ad: bool = False,
    annotations: str = None,
    annotations_column_index: str = None,
    **kwargs,
) -> str:
    """Write a Zarr AnnData object created from Visium HD Space Ranger output.

    Args:
        path (str): Path to a Visium HD Space Ranger output directory.
        stem (str): Prefix for the output Zarr filename.
        bin_size (str | int, optional): Bin size to load. Defaults to ``008um``.
        load_clusters (bool, optional): If clustering files should be included.
            Defaults to True.
        load_embeddings (bool, optional): If embedding files should be included.
            Defaults to True.
        load_raw (bool, optional): If the raw count file should be loaded instead
            of the filtered count file. Defaults to False.
        save_h5ad (bool, optional): If the AnnData object should also be written
            to an h5ad file. Defaults to False.

    Returns:
        str: Output Zarr filename.
    """

    adata = visiumhd_to_anndata(
        path,
        bin_size=bin_size,
        load_clusters=load_clusters,
        load_embeddings=load_embeddings,
        load_raw=load_raw,
        annotations=annotations,
        annotations_column_index=annotations_column_index,
    )
    if save_h5ad:
        adata.write_h5ad(f"tmp-{stem}.h5ad")
    zarr_file = h5ad_to_zarr(adata=adata, stem=stem, **kwargs)

    return zarr_file


def visiumhd_label(
    stem: str,
    file_path: str,
    shape: tuple[int, int] = None,
    obs_subset: tuple[int, T.Any] = None,
    sample_id: str = None,
    relative_size: str = None,
    bin_size: str | int = "008um",
    annotations: str = None,
    annotations_column_index: str = None,
) -> None:
    """Write a label image for Visium HD square bins.

    Args:
        stem (str): Prefix for the output image filename.
        file_path (str): Path to an h5ad file, Visium HD output directory, or a
            specific binned output directory.
        shape (tuple[int, int], optional): Output image shape. Defaults to None.
        obs_subset (tuple(str, T.Any), optional): Tuple containing an obs column
            name and one or more values to subset the AnnData object.
        sample_id (str, optional): Sample ID string within ``uns["spatial"]``.
        relative_size (str, optional): Optional numerical obs column name that
            holds a multiplier for the square side length.
        bin_size (str | int, optional): Bin size to load. Defaults to ``008um``.
    """

    if os.path.isdir(file_path):
        adata = visiumhd_to_anndata(
            file_path,
            bin_size=bin_size,
            annotations=annotations,
            annotations_column_index=annotations_column_index,
        )
    else:
        adata = sc.read(file_path)
        adata = _add_annotations(adata, annotations, annotations_column_index)

    sample_id = sample_id or list(adata.uns["spatial"].keys())[0]

    if not shape:
        hires = adata.uns["spatial"][sample_id].get("images", {}).get("hires")
        scalef = adata.uns["spatial"][sample_id]["scalefactors"].get(
            "tissue_hires_scalef"
        )
        if hires is not None and scalef:
            shape = [int(hires.shape[0] / scalef), int(hires.shape[1] / scalef)]
        else:
            coords = adata.obsm["spatial"]
            side = adata.uns["spatial"][sample_id]["scalefactors"].get(
                "spot_diameter_fullres", 1
            )
            shape = [
                int(np.nanmax(coords[:, 1]) + side),
                int(np.nanmax(coords[:, 0]) + side),
            ]

    adata = reindex_anndata_obs(adata)
    if obs_subset:
        adata = subset_anndata(adata, obs_subset=obs_subset)

    for k in adata.obsm_keys():
        adata.obsm[k] = np.array(adata.obsm[k])

    side_length = adata.uns["spatial"][sample_id]["scalefactors"][
        "spot_diameter_fullres"
    ]
    spot_coords = adata.obsm["spatial"]
    assert adata.obs.shape[0] == spot_coords.shape[0]

    label_img = np.zeros(
        (shape[0], shape[1]),
        dtype=np.min_scalar_type(adata.obs.index.astype(int).max()),
    )

    for sp_id, (x, y) in zip(adata.obs.index, spot_coords):
        multiplier = (
            adata.obs.loc[sp_id, relative_size]
            if relative_size and relative_size in adata.obs
            else 1
        )
        half_side = (side_length * multiplier) / 2
        row_start = max(0, int(np.floor(y - half_side)))
        row_end = min(label_img.shape[0], int(np.ceil(y + half_side)))
        col_start = max(0, int(np.floor(x - half_side)))
        col_end = min(label_img.shape[1], int(np.ceil(x + half_side)))
        label_img[row_start:row_end, col_start:col_end] = int(sp_id)

    tf.imwrite(f"{stem}-label.tif", label_img)

    return


if __name__ == "__main__":
    fire.Fire(visiumhd_to_zarr)
