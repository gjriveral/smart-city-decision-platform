# © 2026 Gabriel Jaime Rivera León. Todos los derechos reservados.

"""Elsevier figure and table export for Socio-Economic Planning Sciences.

Column widths follow the Elsevier artwork guide: 90 mm (one column) and
190 mm (two columns). Text is DejaVu Sans at no less than 7 pt. Vector
files embed TrueType fonts (pdf.fonttype = 42, ps.fonttype = 42). SVG
glyphs are stored as paths. PNG and TIFF are 600 dpi; TIFF uses LZW.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor
from PIL import Image

MM = 1.0 / 25.4
SINGLE_MM = 90.0
DOUBLE_MM = 190.0
MIN_PT = 7.0

# Okabe–Ito colours. They stay separable for the common colour-vision
# deficiencies and are paired with hatches or markers so a grayscale
# photocopy still distinguishes the series.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
BLACK = "#000000"
SKY = "#56B4E9"


def inches(mm: float) -> float:
    return mm * MM


def apply_elsevier_style() -> None:
    import logging

    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": MIN_PT,
        "ytick.labelsize": MIN_PT,
        "legend.fontsize": MIN_PT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.pad_inches": 0.02,
    })
    if plt.rcParams["xtick.labelsize"] < MIN_PT or plt.rcParams["font.size"] < MIN_PT:
        raise RuntimeError("Elsevier figures must use at least 7 pt type.")


def new_figure(width_mm: float, height_mm: float) -> plt.Figure:
    if width_mm not in (SINGLE_MM, DOUBLE_MM):
        raise ValueError(f"Width must be {SINGLE_MM:.0f} mm or {DOUBLE_MM:.0f} mm, not {width_mm}.")
    return plt.figure(
        figsize=(inches(width_mm), inches(height_mm)),
        layout="constrained",
    )


def save_figure(fig: plt.Figure, stem: str, folders: list[Path]) -> None:
    """Write PDF, EPS, SVG, 600 dpi PNG and 600 dpi LZW TIFF."""
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        base = folder / stem
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".eps"))
        fig.savefig(base.with_suffix(".svg"))
        fig.savefig(base.with_suffix(".png"), dpi=600)
        fig.savefig(
            base.with_suffix(".tif"),
            dpi=600,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)


def _pdf_media_box_mm(path: Path) -> tuple[float, float]:
    raw = path.read_bytes()
    start = raw.find(b"/MediaBox")
    if start < 0:
        raise ValueError(f"No MediaBox in {path}")
    bracket = raw[start:start + 80]
    nums = bracket.split(b"[", 1)[1].split(b"]", 1)[0].split()
    width_pt = float(nums[2]) - float(nums[0])
    height_pt = float(nums[3]) - float(nums[1])
    return width_pt * 25.4 / 72.0, height_pt * 25.4 / 72.0


def verify_figure(folder: Path, stem: str, width_mm: float) -> dict:
    png = Image.open(folder / f"{stem}.png")
    tif = Image.open(folder / f"{stem}.tif")
    png_dpi = png.info.get("dpi", (0, 0))
    tif_dpi = tif.info.get("dpi", (0, 0))
    pdf_mm = _pdf_media_box_mm(folder / f"{stem}.pdf")
    tif_comp = tif.info.get("compression", "")
    return {
        "stem": stem,
        "target_width_mm": width_mm,
        "pdf_width_mm": round(pdf_mm[0], 2),
        "pdf_height_mm": round(pdf_mm[1], 2),
        "png_px": f"{png.size[0]}x{png.size[1]}",
        "png_dpi": f"{float(png_dpi[0]):.0f}",
        "tif_px": f"{tif.size[0]}x{tif.size[1]}",
        "tif_dpi": f"{float(tif_dpi[0]):.0f}",
        "tif_compression": str(tif_comp),
        "eps": (folder / f"{stem}.eps").is_file(),
        "svg": (folder / f"{stem}.svg").is_file(),
    }


def _set_border(element, edge: str, sz: str | None) -> None:
    borders = element.find(qn("w:tcBorders")) if element.tag == qn("w:tcPr") else None
    # Caller passes tcPr or tblPr and the child name differs. Handle both below.


def _border_element(tag: str, sz: str | None) -> OxmlElement:
    el = OxmlElement(tag)
    if sz is None:
        el.set(qn("w:val"), "nil")
    else:
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
    return el


def _three_line_table(table, header_cells) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    existing = tbl_pr.find(qn("w:tblBorders"))
    if existing is not None:
        tbl_pr.remove(existing)
    borders = OxmlElement("w:tblBorders")
    borders.append(_border_element("w:top", "12"))
    borders.append(_border_element("w:left", None))
    borders.append(_border_element("w:bottom", "12"))
    borders.append(_border_element("w:right", None))
    borders.append(_border_element("w:insideH", None))
    borders.append(_border_element("w:insideV", None))
    tbl_pr.append(borders)
    for cell in header_cells:
        tc = cell._tc
        tc_pr = tc.get_or_add_tcPr()
        tc_borders = tc_pr.find(qn("w:tcBorders"))
        if tc_borders is not None:
            tc_pr.remove(tc_borders)
        tc_borders = OxmlElement("w:tcBorders")
        tc_borders.append(_border_element("w:bottom", "6"))
        tc_pr.append(tc_borders)


def _round_value(column: str, value):
    if pd.isna(value):
        return ""
    name = column.lower()
    if isinstance(value, str):
        return value
    if "r2" in name:
        return f"{float(value):.3f}"
    if "rmse" in name or name.endswith("mae") or name == "mae":
        return f"{float(value):.3f}"
    if "pct" in name or "percent" in name:
        return f"{float(value):.2f}"
    if name in {"year", "anio", "anio_objetivo", "fold", "n", "n_interior", "n_idw", "eligible_n", "n_pairs"}:
        return str(int(value)) if float(value) == int(value) else f"{float(value):.3f}"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        if name.endswith("_mm") or "mm" in name:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def rounded_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = [_round_value(col, v) for v in out[col]]
    return out


def write_docx_table(df: pd.DataFrame, path: Path, title: str) -> None:
    document = Document()
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = Mm(15)
    section.right_margin = Mm(15)
    heading = document.add_paragraph()
    run = heading.add_run(title)
    run.bold = True
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0, 0, 0)
    display = rounded_frame(df)
    table = document.add_table(rows=1, cols=len(display.columns))
    table.autofit = True
    header = table.rows[0].cells
    for cell, name in zip(header, display.columns):
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(str(name))
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(9)
    for _, row in display.iterrows():
        cells = table.add_row().cells
        for cell, value in zip(cells, row.tolist()):
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(str(value))
            run.font.name = "Calibri"
            run.font.size = Pt(9)
    _three_line_table(table, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)


def export_table(source: Path, dest_csv: Path, dest_docx: Path, title: str) -> pd.DataFrame:
    frame = pd.read_csv(source)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest_csv)
    check = pd.read_csv(dest_csv)
    if not frame.equals(check):
        raise RuntimeError(f"Copied table does not match the code output: {source}")
    write_docx_table(frame, dest_docx, title)
    return frame


def panel_letter(ax, letter: str) -> None:
    ax.text(
        0.0, 1.02, letter, transform=ax.transAxes,
        fontsize=8, fontweight="bold", va="bottom", ha="left",
    )
