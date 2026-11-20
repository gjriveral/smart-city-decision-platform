#!/usr/bin/env python3
# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""Build the Urban Science tables and figures from the saved article results.

Reads article_outputs/ (including the N = 1024 Sobol table) and the official
boundary file. Does not refit the forecast or rerun the Sobol sample.

    python scripts/build_publication_assets.py
"""

from __future__ import annotations

import csv
import json
import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MPath
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "article_outputs"
GEO = ROOT / "data" / "geo" / "comunas.geojson"
DOCS = ROOT / "docs" / "figures_tables_changes.md"
REPO_DIR = ROOT / "deliverables" / "Prompt2"
SYNC_DIR = Path(
    r"c:\Users\gjriv\OneDrive\Documentos\UdeA\Registro SMIC ante la DNDA"
    r"\model_strategic_investment_cities\Entregables_UrbanScience"
    r"\Prompt2_Tablas_Figuras"
)

WIDTH = 7.09  # 180 mm, Urban Science full text width
FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"

NAVY = "#1f4e79"
ORANGE = "#e07a3d"
TEAL = "#2a9d8f"
GRAY = "#4d4d4d"
RED = "#b00020"
SLATE = "#5b7c99"
FILL = "#d6e2f0"

MODEL_ORDER = ["persistence", "linear", "random_forest", "xgboost"]
MODEL_LABELS = {
    "persistence": "Persistence",
    "linear": "Linear regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
SCENARIO_ORDER = ["focalizado", "distribuido", "adaptativo"]
SCENARIO_LABELS = {
    "focalizado": "Focused",
    "distribuido": "Equal",
    "adaptativo": "Adaptive",
}
SOBOL_ORDER = ["beta", "gamma", "delta", "theta", "rho", "n_prioritarios", "noise_sd"]
SOBOL_LABELS = {
    "beta": "Wellbeing rate",
    "gamma": "Index response",
    "delta": "Depreciation",
    "theta": "Infrastructure recovery",
    "rho": "Budget share",
    "n_prioritarios": "Priority-set size",
    "noise_sd": "Noise standard deviation",
}
OUTPUT_LABELS = {
    "cv": "Coefficient of variation",
    "mean_ivc": "Mean index",
    "gini": "Gini coefficient",
}
RAIN_ORDER = ["ninguno", "predictor", "subindice"]
RAIN_LABELS = {
    "ninguno": "No rainfall",
    "predictor": "Rainfall as predictor",
    "subindice": "Rainfall inside the index",
}
TARGET_LABELS = {
    "level": "Index level at t+1",
    "change": "Change in the index",
}
SPEC_LABELS = {
    "focused_yearly_ivc": "Focused, yearly index rank, equal shares",
    "focused_yearly_ivc_proportional": "Focused, yearly index rank, proportional shares",
    "focused_fixed_shap": "Focused, fixed priority rank",
    "focused_fixed_shap_saturation_on": "Focused, fixed priority rank, saturation on",
    "focused_yearly_ivc_saturation_on": "Focused, yearly index rank, saturation on",
    "equal": "Equal allocation",
    "equal_saturation_on": "Equal allocation, saturation on",
    "adaptive": "Adaptive allocation",
    "adaptive_saturation_on": "Adaptive allocation, saturation on",
    "focused_yearly_ivc_observed_start": "Focused, yearly index rank, observed initial index",
}
PATH_SHOW = [
    ("focused_yearly_ivc", "Focused, yearly index rank", NAVY),
    ("focused_fixed_shap", "Focused, fixed priority rank", RED),
    ("equal", "Equal allocation", GRAY),
    ("adaptive", "Adaptive allocation", TEAL),
]

NOTE_01 = (
    "Source: authors' elaboration based on the Medellín quality-of-life panel (2013–2024). "
    "Walk-forward R², RMSE and MAE for the index at t+1 given subindices at t, pooled over target years 2017–2024."
)
NOTE_02 = (
    "Source: authors' elaboration based on the same walk-forward evaluation, reported by target year."
)
NOTE_03 = (
    "Source: authors' elaboration based on a leave-one-territory-out evaluation of the same lagged specification. "
    "Each territory is scored only in the folds that hold it out."
)
NOTE_04 = (
    "Source: authors' elaboration based on the agent-based allocation model "
    "(21 territories, 10-year horizon, no idiosyncratic noise). "
    "A positive change in the coefficient of variation means dispersion increased."
)
NOTE_05 = (
    "Source: authors' elaboration based on 30 replications of the same model "
    "with idiosyncratic noise (standard deviation 0.05)."
)
NOTE_06 = (
    "Source: authors' elaboration based on the quality-of-life panel and the territorial rainfall series. "
    "A negative R² means the forecast is worse than the mean of the evaluation sample."
)
NOTE_07 = (
    "Source: authors' elaboration based on a Saltelli sample (N = 1024; 9,216 evaluations) of the allocation model. "
    "Intervals are the 2.5th and 97.5th percentiles of 1,000 bootstrap resamples. "
    "The outcome is the coefficient of variation of the index."
)
NOTE_08 = (
    "Source: authors' elaboration based on the municipal rain-gauge network aggregated to the 21 territories. "
    "Adjusted totals are shown only for territory-years with at least 80% hourly coverage. "
    "2012 is the first year of the network and 2026 is a partial year."
)
NOTE_S1 = (
    "Source: authors' elaboration. Leave-one-territory-out R² and RMSE by territory "
    "for the lagged forecast of the index."
)
NOTE_S2 = (
    "Source: authors' elaboration based on the same Saltelli sample as Table 7. "
    "Outcomes are the mean index and the Gini coefficient."
)
NOTE_S3 = (
    "Source: authors' elaboration based on the allocation paths shown in Figure 4."
)
NOTE_F1 = (
    "Source: authors' elaboration based on the official administrative boundaries of Medellín."
)
NOTE_F2 = (
    "Source: authors' elaboration based on the Medellín quality-of-life panel. "
    "Out-of-sample R² for the index at t+1 given subindices at t."
)
NOTE_F3 = (
    "Source: authors' elaboration based on the same walk-forward evaluation, by target year."
)
NOTE_F4 = (
    "Source: authors' elaboration based on the agent-based allocation model "
    "(10-year horizon, 21 territories)."
)
NOTE_F5 = (
    "Source: authors' elaboration based on the quality-of-life panel and the territorial rainfall series."
)
NOTE_F6 = (
    "Source: authors' elaboration. Saltelli sample, N = 1024; bars are 95% percentile bootstrap intervals."
)

FORBIDDEN = (
    "legacy",
    "article_outputs",
    "sobol_update",
    ".csv",
    ".py",
    "audit",
    "table_forecast",
    "fig_sobol",
    "regenerate_article",
)


def num(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str) and value.strip() == "":
        return ""
    return f"{float(value):.{digits}f}"


def integer(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(int(round(float(value))))


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
    })


def style_run(run, size: float, bold: bool = False, italic: bool = False) -> None:
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.name = "Arial"
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), "Arial")
    rfonts.set(qn("w:hAnsi"), "Arial")
    rfonts.set(qn("w:cs"), "Arial")


def write_docx(path: Path, caption: str, columns: list[str], rows: list[list[str]], note: str) -> None:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    section = doc.sections[0]
    section.left_margin = Cm(1.5)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    paragraph = doc.add_paragraph()
    style_run(paragraph.add_run(caption), 10, bold=True)
    table = doc.add_table(rows=1 + len(rows), cols=len(columns))
    table.style = "Table Grid"
    table.autofit = True
    for j, column in enumerate(columns):
        cell = table.rows[0].cells[j]
        cell.paragraphs[0].clear()
        style_run(cell.paragraphs[0].add_run(column), 9, bold=True)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.paragraphs[0].clear()
            style_run(cell.paragraphs[0].add_run(value), 9)
    note_p = doc.add_paragraph()
    style_run(note_p.add_run(note), 8, italic=True)
    doc.save(path)


def write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def publish_table(dirs: list[Path], stem: str, caption: str, columns: list[str], rows: list[list[str]], note: str) -> None:
    assert_clean(caption + note + " ".join(columns))
    for folder in dirs:
        write_csv(folder / f"{stem}.csv", columns, rows)
        write_docx(folder / f"{stem}.docx", caption, columns, rows, note)


def assert_clean(text: str) -> None:
    lowered = text.lower()
    for token in FORBIDDEN:
        if token in lowered:
            raise SystemExit(f"Publication note still contains '{token}': {text[:180]}")


def choose_version(dirs: list[Path]) -> int:
    stems = [
        "table_01_forecast_metrics",
        "table_02_forecast_by_year",
        "table_03_loto_metrics",
        "table_04_abm_dispersion",
        "table_05_abm_monte_carlo",
        "table_06_rain_forecast",
        "table_07_sobol_indices",
        "table_08_rain_qc_by_year",
        "table_S1_loto_by_territory",
        "table_S2_sobol_mean_ivc_gini",
        "table_S3_abm_cv_path",
        "fig_01_territories",
        "fig_02_forecast_r2",
        "fig_03_forecast_by_year",
        "fig_04_abm_cv_paths",
        "fig_05_rain_forecast_r2",
        "fig_06_sobol_cv",
        "figures_tables_changes",
        "LEEME",
    ]
    version = 2
    while True:
        clash = False
        for folder in dirs:
            if not folder.exists():
                continue
            for stem in stems:
                for ext in (".csv", ".docx", ".pdf", ".tif", ".md"):
                    if (folder / f"{stem}_v{version}{ext}").exists() or (folder / f"{stem}{ext}").exists() and stem in {"LEEME", "figures_tables_changes"} and version == 2 and (folder / f"{stem}.md").exists() and stem == "LEEME":
                        pass
        # Explicit clash: any planned versioned file, or an existing LEEME.md / changes md on version 2.
        for folder in dirs:
            if not folder.exists():
                continue
            for stem in stems:
                if stem in {"LEEME", "figures_tables_changes"}:
                    candidate = folder / (f"{stem}.md" if version == 2 else f"{stem}_v{version}.md")
                    if candidate.exists():
                        clash = True
                else:
                    for ext in (".csv", ".docx", ".pdf", ".tif"):
                        if (folder / f"{stem}_v{version}{ext}").exists():
                            clash = True
        if not clash:
            return version
        version += 1


def save_figure(fig: plt.Figure, dirs: list[Path], stem: str, note: str) -> None:
    assert_clean(note)
    wrapped = textwrap.fill(note, width=100)
    fig.text(0.5, -0.02, wrapped, fontsize=8, fontstyle="italic", ha="center", va="top")
    for folder in dirs:
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.15)
        tiff_path = folder / f"{stem}.tif"
        fig.savefig(
            tiff_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.15,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        flatten_tiff(tiff_path)
    plt.close(fig)


def flatten_tiff(path: Path) -> None:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (600, 600))
        if image.mode == "RGB":
            return
        background = Image.new("RGB", image.size, "white")
        if image.mode == "RGBA":
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image.convert("RGB"))
    background.save(path, compression="tiff_lzw", dpi=(int(dpi[0]), int(dpi[1])))


def forecast_rows(frame: pd.DataFrame) -> list[list[str]]:
    indexed = frame.set_index("model")
    rows = []
    for model in MODEL_ORDER:
        row = indexed.loc[model]
        rows.append([
            MODEL_LABELS[model],
            integer(row["n"]),
            num(row["RMSE"], 3),
            num(row["MAE"], 3),
            num(row["R2"], 3),
        ])
    return rows


def fit_columns() -> list[str]:
    return ["Model", "Territory-years", "RMSE", "MAE", "R²"]


def label_point(ring: list) -> tuple[float, float]:
    coords = np.asarray(ring, dtype=float)[:, :2]
    fallback = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
    path = MPath(coords)
    step = max(1, len(coords) // 250)
    boundary = coords[::step]
    minx, miny = coords.min(axis=0)
    maxx, maxy = coords.max(axis=0)
    best = fallback
    best_d = -1.0
    for x in np.linspace(minx, maxx, 36):
        for y in np.linspace(miny, maxy, 36):
            if not path.contains_point((x, y)):
                continue
            distance = float(np.min(np.hypot(boundary[:, 0] - x, boundary[:, 1] - y)))
            if distance > best_d:
                best_d = distance
                best = (float(x), float(y))
    return best


def rings_of(geometry: dict) -> list[list]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [polygon[0] for polygon in geometry["coordinates"]]
    return []


def plot_territories(dirs: list[Path], stem: str) -> None:
    payload = json.loads(GEO.read_text(encoding="utf-8"))
    features = sorted(payload["features"], key=lambda feat: str(feat["properties"].get("codigo", "")))
    fig, ax = plt.subplots(figsize=(WIDTH, 8.6))
    labels = []
    for feature in features:
        props = feature["properties"]
        code = str(props.get("codigo", "")).zfill(2)
        name = str(props.get("nombre", ""))
        labels.append(f"{code}  {name}")
        rings = rings_of(feature["geometry"])
        largest = max(rings, key=len)
        cx, cy = label_point(largest)
        for ring in rings:
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            ax.fill(xs, ys, facecolor=FILL, edgecolor=NAVY, linewidth=0.4, zorder=1)
        ax.text(
            cx, cy, code, ha="center", va="center", fontsize=8, color=NAVY, zorder=2,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )
    ax.set_aspect("equal")
    ax.set_title("Administrative territories of Medellín")
    ax.tick_params(labelsize=8)
    fig.subplots_adjust(bottom=0.42, left=0.12, right=0.98, top=0.94)
    left = labels[:11]
    right = labels[11:]
    fig.text(0.08, 0.36, "\n".join(left), fontsize=8, va="top", ha="left", family=FONT)
    fig.text(0.55, 0.36, "\n".join(right), fontsize=8, va="top", ha="left", family=FONT)
    save_figure(fig, dirs, stem, NOTE_F1)


def plot_forecast(dirs: list[Path], stem_r2: str, stem_year: str, pooled: pd.DataFrame, by_year: pd.DataFrame, loto: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    temporal = pooled.set_index("model").loc[MODEL_ORDER, "R2"]
    spatial = loto.set_index("model").loc[MODEL_ORDER, "R2"]
    ax.bar(x - width / 2, temporal, width, label="Temporal walk-forward", color=NAVY)
    ax.bar(x + width / 2, spatial, width, label="Leave-one-territory-out", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Out-of-sample R²")
    ax.set_ylim(0, 1)
    ax.set_title("Index at t+1 predicted from subindices at t")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, dirs, stem_r2, NOTE_F2)

    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    colors = (GRAY, NAVY, TEAL, ORANGE)
    for model, color in zip(MODEL_ORDER, colors):
        group = by_year[by_year["model"] == model].sort_values("anio_objetivo")
        ax.plot(group["anio_objetivo"], group["R2"], marker="o", markersize=4, label=MODEL_LABELS[model], color=color)
    ax.axhline(0, color="#444444", linewidth=0.6)
    ax.set_xlabel("Target year")
    ax.set_ylabel("R²")
    ax.set_title("Walk-forward R² by target year")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(bottom=0.30)
    save_figure(fig, dirs, stem_year, NOTE_F3)


def plot_paths(dirs: list[Path], stem: str, paths: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 4.5))
    for key, label, color in PATH_SHOW:
        group = paths[paths["specification"] == key].sort_values("year")
        ax.plot(group["year"], group["cv"], marker="o", markersize=4, label=label, color=color)
    ax.set_xlabel("Simulation year")
    ax.set_ylabel("Coefficient of variation")
    ax.set_title("Dispersion of the index across 21 territories")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, dirs, stem, NOTE_F4)


def plot_rain(dirs: list[Path], stem: str, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.6))
    colors = {"ninguno": GRAY, "predictor": NAVY, "subindice": ORANGE}
    x = np.arange(len(MODEL_ORDER))
    width = 0.25
    for ax, target, title in (
        (axes[0], "level", "Index level at t+1"),
        (axes[1], "change", "Change in the index"),
    ):
        subset = metrics[metrics["target"] == target]
        for i, uso in enumerate(RAIN_ORDER):
            values = [
                float(subset[(subset["clima_uso"] == uso) & (subset["model"] == model)]["R2"].iloc[0])
                for model in MODEL_ORDER
            ]
            ax.bar(x + (i - 1) * width, values, width, label=RAIN_LABELS[uso], color=colors[uso])
        ax.axhline(0, color="#444444", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=15, ha="right")
        ax.set_ylabel("Walk-forward R²")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.45)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    save_figure(fig, dirs, stem, NOTE_F5)


def plot_sobol(dirs: list[Path], stem: str, indices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 5.2), sharey=True)
    subset = indices[indices["output"] == "cv"].copy()
    y = np.arange(len(SOBOL_ORDER))
    height = 0.36
    hi = float(np.nanmax(subset[["S1_ci_high", "ST_ci_high"]].to_numpy()))
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        group = subset[subset["scenario"] == scenario].set_index("parameter").loc[SOBOL_ORDER]
        s1_err = np.vstack([
            np.clip(group["S1"] - group["S1_ci_low"], 0, None),
            np.clip(group["S1_ci_high"] - group["S1"], 0, None),
        ])
        st_err = np.vstack([
            np.clip(group["ST"] - group["ST_ci_low"], 0, None),
            np.clip(group["ST_ci_high"] - group["ST"], 0, None),
        ])
        ax.barh(y + height / 2, group["S1"], height=height, xerr=s1_err, color=SLATE,
                label="First order (S1)", capsize=2, error_kw={"elinewidth": 0.6, "ecolor": "#333333"})
        ax.barh(y - height / 2, group["ST"], height=height, xerr=st_err, color=NAVY,
                label="Total order (ST)", capsize=2, error_kw={"elinewidth": 0.6, "ecolor": "#333333"})
        ax.axvline(0, color="#444444", linewidth=0.6)
        ax.axvline(1, color="#888888", linewidth=0.5, linestyle=":")
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlabel("Sobol index")
        ax.set_xlim(-0.25, max(1.15, hi + 0.08))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SOBOL_LABELS[name] for name in SOBOL_ORDER])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Sobol indices for the coefficient of variation", fontsize=10, y=0.98)
    fig.subplots_adjust(top=0.78, wspace=0.18)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.90))
    save_figure(fig, dirs, stem, NOTE_F6)


def sobol_rows(indices: pd.DataFrame, outputs: list[str], with_output: bool) -> list[list[str]]:
    rows = []
    for output in outputs:
        for scenario in SCENARIO_ORDER:
            group = indices[(indices["output"] == output) & (indices["scenario"] == scenario)]
            group = group.set_index("parameter").loc[SOBOL_ORDER]
            for parameter in SOBOL_ORDER:
                row = group.loc[parameter]
                values = [
                    SCENARIO_LABELS[scenario],
                    SOBOL_LABELS[parameter],
                    num(row["S1"], 3),
                    num(row["S1_ci_low"], 3),
                    num(row["S1_ci_high"], 3),
                    num(row["ST"], 3),
                    num(row["ST_ci_low"], 3),
                    num(row["ST_ci_high"], 3),
                ]
                if with_output:
                    values.insert(0, OUTPUT_LABELS[output])
                rows.append(values)
    return rows


def sobol_columns(with_output: bool) -> list[str]:
    columns = [
        "Allocation rule",
        "Parameter",
        "S1",
        "S1 2.5%",
        "S1 97.5%",
        "ST",
        "ST 2.5%",
        "ST 97.5%",
    ]
    if with_output:
        columns.insert(0, "Outcome")
    return columns


def dispersion_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, row in frame.iterrows():
        label = SPEC_LABELS.get(str(row["specification"]), str(row["specification"]))
        rows.append([
            label,
            num(row["cv_t0"], 3),
            num(row["cv_tH"], 3),
            num(row["cv_reduction_pct"], 1),
            num(row["gini_tH"], 3),
            num(row["mean_ivc_reduction_pct"], 1),
        ])
    return rows


def monte_carlo_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, row in frame.iterrows():
        label = SPEC_LABELS.get(str(row["specification"]), str(row["specification"]))
        rows.append([
            label,
            integer(row["n_replications"]),
            num(row["cv_reduction_pct_mean"], 1),
            num(row["cv_reduction_pct_std"], 1),
            num(row["cv_reduction_pct_min"], 1),
            num(row["cv_reduction_pct_max"], 1),
        ])
    return rows


def by_year_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    years = sorted(frame["anio_objetivo"].unique())
    indexed = frame.set_index(["anio_objetivo", "model"])
    for year in years:
        for model in MODEL_ORDER:
            row = indexed.loc[(year, model)]
            rows.append([
                integer(year),
                MODEL_LABELS[model],
                integer(row["n"]),
                num(row["RMSE"], 3),
                num(row["MAE"], 3),
                num(row["R2"], 3),
            ])
    return rows


def rain_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    indexed = frame.set_index(["target", "clima_uso", "model"])
    for target in ("level", "change"):
        for uso in RAIN_ORDER:
            for model in MODEL_ORDER:
                row = indexed.loc[(target, uso, model)]
                rows.append([
                    RAIN_LABELS[uso],
                    TARGET_LABELS[target],
                    MODEL_LABELS[model],
                    integer(row["n"]),
                    num(row["R2"], 3),
                    num(row["RMSE"], 3),
                    num(row["MAE"], 3),
                ])
    return rows


def qc_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    for _, row in frame.sort_values("year").iterrows():
        coverage = "" if pd.isna(row["coverage_median"]) else num(float(row["coverage_median"]) * 100.0, 1)
        adjusted = row["adjusted_mm_median"]
        rows.append([
            integer(row["year"]),
            integer(row["n_interior"]),
            integer(row["n_idw"]),
            coverage,
            num(row["observed_mm_median"], 0),
            integer(row["eligible_n"]),
            num(adjusted, 0),
        ])
    return rows


def loto_territory_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    territories = sorted(frame["Comuna"].unique())
    indexed = frame.set_index(["Comuna", "model"])
    for territory in territories:
        for model in MODEL_ORDER:
            row = indexed.loc[(territory, model)]
            rows.append([
                str(territory),
                MODEL_LABELS[model],
                integer(row["n"]),
                num(row["RMSE"], 3),
                num(row["R2"], 3),
            ])
    return rows


def path_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    for key, label, _color in PATH_SHOW:
        group = frame[frame["specification"] == key].sort_values("year")
        for _, row in group.iterrows():
            rows.append([
                label,
                integer(row["year"]),
                num(row["cv"], 3),
                num(row["gini"], 3),
                num(row["mean_ivc"], 3),
            ])
    return rows


def focused_priority(indices: pd.DataFrame) -> pd.Series:
    match = indices[
        (indices["scenario"] == "focalizado")
        & (indices["output"] == "cv")
        & (indices["parameter"] == "n_prioritarios")
    ]
    return match.iloc[0]


def write_changes(path: Path, version: int, priority: pd.Series) -> None:
    st = num(priority["ST"], 3)
    st_lo = num(priority["ST_ci_low"], 3)
    st_hi = num(priority["ST_ci_high"], 3)
    s1 = num(priority["S1"], 3)
    text = f"""# Tables and figures for Urban Science

Prepared for Urban Science (MDPI) from the saved article results. The Sobol table and Figure 6 use the N = 1024 Saltelli sample already reported in `docs/sobol_update.md` (9,216 evaluations, 95% percentile bootstrap intervals, seed 42). The forecast and the allocation model were not refit.

File names in this delivery use the suffix `_v{version}`.

## Style applied to every item

- Typeface: Arial. Figure titles 10 pt, axis labels 9 pt, tick labels and legends 8 pt. Table body 9 pt, caption 10 pt, note 8 pt italic.
- Figure width: 180 mm. Vector PDF (TrueType fonts embedded) and TIFF at 600 dpi with LZW compression.
- Tables: editable DOCX and CSV. The academic note is the italic paragraph under the DOCX table. The CSV is the same grid without that paragraph.
- Decimals: R², RMSE, MAE, coefficient of variation, Gini, mean index and Sobol indices to 3 decimal places; percentage changes to 1 decimal place; rainfall to the nearest millimetre; hourly coverage to 1 decimal place, in percent; counts as integers.
- Notes name the data and the design. They do not cite file names, repository paths, code checks, or an earlier specification of the forecast.

The same-year reconstruction of the index is not in Tables 1–8 or S1–S3. Those numbered items are the out-of-sample forecast, the allocation scenarios, the rainfall comparison and the Sobol indices.

## Figure 1. Administrative territories

New map of the 21 units (communes and corregimientos) drawn from the official administrative boundaries. Each polygon is numbered with the official code; the names sit under the map. There was no territory figure in the previous output set.

Note: {NOTE_F1}

## Figure 2. Walk-forward and leave-one-territory-out R²

Same comparison as the previous unnumbered forecast figure. Bars now share the 180 mm width, the Arial sizes above, a common 0–1 axis, and an academic note. Persistence remains the highest walk-forward R² (0.821) and linear regression the highest leave-one-territory-out R² (0.799).

Note: {NOTE_F2}

## Figure 3. Walk-forward R² by target year

Same four series as the previous year-by-year figure, restyled. The vertical axis is no longer clipped at zero, so years in which a model falls below the sample mean stay visible.

Note: {NOTE_F3}

## Figure 4. Allocation paths

The four allocation rules from the previous path figure (focused yearly rank, focused fixed rank, equal, adaptive), restyled. Saturation variants stay in Table 4 and are not added to the figure.

Note: {NOTE_F4}

## Figure 5. Rainfall and the forecast

The previous two-panel rainfall figure, restyled. The title is now a description of the comparison. Both panels report walk-forward R², for the level of the index and for its change.

Note: {NOTE_F5}

## Figure 6. Sobol indices

Redrawn from the N = 1024 sample, not from the earlier N = 128 figure. First-order and total-order indices for the coefficient of variation, with the 95% percentile bootstrap interval as error bars. For the focused rule, the priority-set size has S1 = {s1} and ST = {st} (interval {st_lo}–{st_hi}).

Note: {NOTE_F6}

## Table 1. Walk-forward forecast

Pooled metrics for persistence, linear regression, random forest and XGBoost (168 territory-years). Rounded to 3 decimal places. The previous file had unrounded floats and no note.

Note: {NOTE_01}

## Table 2. Walk-forward forecast by target year

One row per target year and model, from the previous year-by-year metrics, with the same rounding.

Note: {NOTE_02}

## Table 3. Leave-one-territory-out forecast

Pooled spatial check (231 territory-years). Linear regression 0.799, XGBoost 0.787, persistence 0.783, random forest 0.757.

Note: {NOTE_03}

## Table 4. Allocation scenarios

Terminal coefficient of variation, its percentage change, terminal Gini and the percentage change in the mean index, for every saved specification. Percentage changes use 1 decimal place.

Note: {NOTE_04}

## Table 5. Monte Carlo

Thirty replications at noise standard deviation 0.05. Mean, standard deviation, minimum and maximum of the percentage change in the coefficient of variation, at 1 decimal place.

Note: {NOTE_05}

## Table 6. Rainfall treatments

Walk-forward R², RMSE and MAE for three rainfall treatments and two targets (index level and change), 126 territory-years each.

Note: {NOTE_06}

## Table 7. Sobol indices for the coefficient of variation

Replaces the N = 128 indices. Twenty-one rows: three allocation rules and seven parameters. Columns are S1, ST and the 2.5% and 97.5% bootstrap percentiles, at 3 decimal places. Focused priority-set size: S1 = {s1}, ST = {st} ({st_lo}–{st_hi}).

Note: {NOTE_07}

## Table 8. Rainfall coverage by year

Median hourly coverage (percent, 1 decimal), median observed rainfall (millimetres) and the median adjusted total for territory-years that meet the 80% coverage rule. The adjusted cell is blank in 2012 and 2026 because no territory meets that rule.

Note: {NOTE_08}

## Table S1. Leave-one-territory-out results by territory

R² and RMSE for each territory and model. A negative R² means the held-out territory is harder to predict than its own sample mean.

Note: {NOTE_S1}

## Table S2. Sobol indices for the mean index and the Gini coefficient

The other two outcomes from the same N = 1024 sample. Table 7 keeps the coefficient of variation, which is the outcome in Figure 6.

Note: {NOTE_S2}

## Table S3. Allocation path behind Figure 4

Coefficient of variation, Gini coefficient and mean index by simulation year for the four rules drawn in Figure 4.

Note: {NOTE_S3}
"""
    assert_clean(text.replace("docs/sobol_update.md", "the Sobol update note").replace(".md", ""))
    # The changes document may name the companion note. The assertion above strips that citation.
    path.write_text(text, encoding="utf-8")


def write_leeme(path: Path, version: int, pr_url: str) -> None:
    suffix = f"_v{version}"
    text = f"""# Prompt 2 — Tablas y figuras (Urban Science)

## Qué se cambió

Las tablas y figuras del artículo quedaron numeradas para Urban Science (MDPI). Las notas ya no citan archivos, rutas ni una especificación anterior del pronóstico: cada nota dice "Source: authors' elaboration based on ...". La fuente es Arial; los R², errores, CV, Gini e índices de Sobol van a 3 decimales; los cambios porcentuales a 1 decimal; la lluvia al milímetro; la cobertura horaria a 1 decimal, en porcentaje.

La Tabla 7 y la Figura 6 usan el Sobol nuevo (N = 1024, intervalos bootstrap al 95 %). No se reestimó el pronóstico ni se volvió a correr el muestreo.

El detalle de cada tabla y figura está en `figures_tables_changes{'' if version == 2 else suffix}.md`.

## Archivos de esta carpeta

Tablas (DOCX editable y CSV), sufijo `{suffix}`:

- `table_01_forecast_metrics{suffix}` — pronóstico walk-forward agrupado.
- `table_02_forecast_by_year{suffix}` — el mismo pronóstico por año objetivo.
- `table_03_loto_metrics{suffix}` — leave-one-territory-out agrupado.
- `table_04_abm_dispersion{suffix}` — escenarios de asignación al año 10.
- `table_05_abm_monte_carlo{suffix}` — 30 réplicas con ruido.
- `table_06_rain_forecast{suffix}` — lluvia como predictor o dentro del índice.
- `table_07_sobol_indices{suffix}` — S1 y ST del CV, N = 1024.
- `table_08_rain_qc_by_year{suffix}` — cobertura y totales de lluvia por año.
- `table_S1_loto_by_territory{suffix}` — leave-one-territory-out por territorio.
- `table_S2_sobol_mean_ivc_gini{suffix}` — Sobol del índice medio y del Gini.
- `table_S3_abm_cv_path{suffix}` — sendas de la Figura 4.

Figuras (PDF vectorial y TIFF a 600 dpi), sufijo `{suffix}`:

- `fig_01_territories{suffix}` — mapa de las 21 unidades.
- `fig_02_forecast_r2{suffix}` — R² temporal y espacial.
- `fig_03_forecast_by_year{suffix}` — R² por año objetivo.
- `fig_04_abm_cv_paths{suffix}` — sendas del CV.
- `fig_05_rain_forecast_r2{suffix}` — lluvia y pronóstico.
- `fig_06_sobol_cv{suffix}` — índices de Sobol del CV.

También está `figures_tables_changes{'' if version == 2 else suffix}.md`.

## Pull request

{pr_url}

Esta carpeta es `Entregables_UrbanScience/Prompt2_Tablas_Figuras`, dentro del directorio de trabajo. Ese directorio se sincroniza solo con https://drive.google.com/drive/u/0/folders/1t3a20nixkH2IsaaEHzBVO_eiPbqpH6Hp. No se usó el conector de Drive. Los nombres llevan `{suffix}`, así que no reemplazan archivos que ya estuvieran en Drive.

## Problemas pendientes

El intervalo del ST del tamaño del conjunto prioritario, en la regla focalizada, todavía llega por encima de 1. La estimación puntual queda dentro de [0, 1]. La nota de la Tabla 7 reporta el intervalo del percentil, no un recorte a la unidad.
"""
    path.write_text(text, encoding="utf-8")


def verify(dirs: list[Path], version: int) -> None:
    figures = [
        "fig_01_territories",
        "fig_02_forecast_r2",
        "fig_03_forecast_by_year",
        "fig_04_abm_cv_paths",
        "fig_05_rain_forecast_r2",
        "fig_06_sobol_cv",
    ]
    tables = [
        "table_01_forecast_metrics",
        "table_02_forecast_by_year",
        "table_03_loto_metrics",
        "table_04_abm_dispersion",
        "table_05_abm_monte_carlo",
        "table_06_rain_forecast",
        "table_07_sobol_indices",
        "table_08_rain_qc_by_year",
        "table_S1_loto_by_territory",
        "table_S2_sobol_mean_ivc_gini",
        "table_S3_abm_cv_path",
    ]
    for folder in dirs:
        for stem in figures:
            pdf = folder / f"{stem}_v{version}.pdf"
            tiff = folder / f"{stem}_v{version}.tif"
            raw = pdf.read_bytes()
            if b"/Subtype /Image" in raw or b"/Subtype/Image" in raw:
                raise SystemExit(f"{pdf.name} contains a raster image")
            if b"/Font" not in raw:
                raise SystemExit(f"{pdf.name} has no embedded font")
            with Image.open(tiff) as image:
                dpi = image.info.get("dpi", (0, 0))
                if int(round(dpi[0])) != 600 or int(round(dpi[1])) != 600:
                    raise SystemExit(f"{tiff.name} dpi is {dpi}, expected 600")
        for stem in tables:
            csv_path = folder / f"{stem}_v{version}.csv"
            docx_path = folder / f"{stem}_v{version}.docx"
            csv_text = csv_path.read_text(encoding="utf-8")
            assert_clean(csv_text)
            with zipfile.ZipFile(docx_path) as archive:
                xml = archive.read("word/document.xml").decode("utf-8")
            if "Source: authors" not in xml:
                raise SystemExit(f"{docx_path.name} has no academic source note")
            assert_clean(xml)
    print(f"Verified PDF vector, TIFF 600 dpi, and notes in {len(dirs)} folders. Version v{version}.")


def main() -> None:
    apply_style()
    dirs = [REPO_DIR, SYNC_DIR]
    for folder in dirs:
        folder.mkdir(parents=True, exist_ok=True)
    version = choose_version(dirs)
    suffix = f"_v{version}"

    pooled = pd.read_csv(OUT / "table_forecast_metrics.csv")
    by_year = pd.read_csv(OUT / "table_forecast_by_year.csv")
    loto = pd.read_csv(OUT / "table_loto_metrics.csv")
    loto_by = pd.read_csv(OUT / "table_loto_by_territory.csv")
    dispersion = pd.read_csv(OUT / "table_abm_dispersion.csv")
    monte = pd.read_csv(OUT / "table_abm_monte_carlo.csv")
    paths = pd.read_csv(OUT / "table_abm_cv_path.csv")
    rain = pd.read_csv(OUT / "table_rain_forecast.csv")
    qc = pd.read_csv(OUT / "table_rain_qc_by_year.csv")
    sobol = pd.read_csv(OUT / "table_sobol_indices.csv")
    if int(sobol["n_sobol"].iloc[0]) < 1024:
        raise SystemExit("Sobol table is not the N = 1024 sample")

    fit = fit_columns()
    publish_table(
        dirs, f"table_01_forecast_metrics{suffix}",
        "Table 1. Walk-forward forecast of the index at t+1.",
        fit, forecast_rows(pooled), NOTE_01,
    )
    publish_table(
        dirs, f"table_02_forecast_by_year{suffix}",
        "Table 2. Walk-forward forecast by target year.",
        ["Target year", "Model", "Territory-years", "RMSE", "MAE", "R²"],
        by_year_rows(by_year), NOTE_02,
    )
    publish_table(
        dirs, f"table_03_loto_metrics{suffix}",
        "Table 3. Leave-one-territory-out forecast of the index at t+1.",
        fit, forecast_rows(loto), NOTE_03,
    )
    publish_table(
        dirs, f"table_04_abm_dispersion{suffix}",
        "Table 4. Index dispersion after 10 simulated years.",
        ["Allocation rule", "CV at t = 0", "CV at t = 10", "Change in CV (%)", "Gini at t = 10", "Change in mean index (%)"],
        dispersion_rows(dispersion), NOTE_04,
    )
    publish_table(
        dirs, f"table_05_abm_monte_carlo{suffix}",
        "Table 5. Change in the coefficient of variation across 30 replications.",
        ["Allocation rule", "Replications", "Mean change in CV (%)", "SD", "Minimum", "Maximum"],
        monte_carlo_rows(monte), NOTE_05,
    )
    publish_table(
        dirs, f"table_06_rain_forecast{suffix}",
        "Table 6. Walk-forward forecast with and without rainfall.",
        ["Rainfall treatment", "Target", "Model", "Territory-years", "R²", "RMSE", "MAE"],
        rain_rows(rain), NOTE_06,
    )
    publish_table(
        dirs, f"table_07_sobol_indices{suffix}",
        "Table 7. Sobol indices for the coefficient of variation of the index.",
        sobol_columns(False), sobol_rows(sobol, ["cv"], False), NOTE_07,
    )
    publish_table(
        dirs, f"table_08_rain_qc_by_year{suffix}",
        "Table 8. Rainfall coverage and annual totals by year.",
        ["Year", "Territories with interior gauges", "Territories estimated by interpolation",
         "Median hourly coverage (%)", "Median observed rainfall (mm)",
         "Territories meeting the coverage rule", "Median adjusted rainfall (mm)"],
        qc_rows(qc), NOTE_08,
    )
    publish_table(
        dirs, f"table_S1_loto_by_territory{suffix}",
        "Table S1. Leave-one-territory-out fit by territory.",
        ["Territory", "Model", "Territory-years", "RMSE", "R²"],
        loto_territory_rows(loto_by), NOTE_S1,
    )
    publish_table(
        dirs, f"table_S2_sobol_mean_ivc_gini{suffix}",
        "Table S2. Sobol indices for the mean index and the Gini coefficient.",
        sobol_columns(True), sobol_rows(sobol, ["mean_ivc", "gini"], True), NOTE_S2,
    )
    publish_table(
        dirs, f"table_S3_abm_cv_path{suffix}",
        "Table S3. Index dispersion by simulation year.",
        ["Allocation rule", "Year", "Coefficient of variation", "Gini coefficient", "Mean index"],
        path_rows(paths), NOTE_S3,
    )

    plot_territories(dirs, f"fig_01_territories{suffix}")
    plot_forecast(dirs, f"fig_02_forecast_r2{suffix}", f"fig_03_forecast_by_year{suffix}", pooled, by_year, loto)
    plot_paths(dirs, f"fig_04_abm_cv_paths{suffix}", paths)
    plot_rain(dirs, f"fig_05_rain_forecast_r2{suffix}", rain)
    plot_sobol(dirs, f"fig_06_sobol_cv{suffix}", sobol)

    priority = focused_priority(sobol)
    changes_name = "figures_tables_changes.md" if version == 2 else f"figures_tables_changes_v{version}.md"
    leeme_name = "LEEME.md" if version == 2 else f"LEEME_v{version}.md"
    write_changes(DOCS, version, priority)
    for folder in dirs:
        write_changes(folder / changes_name, version, priority)
        write_leeme(folder / leeme_name, version, "Pendiente: el enlace del pull request se agrega al abrir el PR.")
    verify(dirs, version)


if __name__ == "__main__":
    main()
