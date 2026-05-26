"""
config_loader.py
================
Lee config.yaml y expone la configuración de ciudad como un objeto
tipado (dataclasses). Cualquier script del proyecto puede importar:

    from config_loader import get_config
    cfg = get_config()
    print(cfg.ciudad.unidad_territorial)   # "comuna"
    print(cfg.rutas.datos_climaticos)      # "pluviometrica"
    print(cfg.territorios[0].nombre)       # "Popular"

Para cambiar de ciudad basta con editar config.yaml; el código no cambia.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# pyyaml es la única dependencia externa de este módulo.
try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML no está instalado. Ejecuta:  py -m pip install pyyaml"
    ) from exc

# ── Rutas de búsqueda del archivo de configuración ────────────────────────────
_DEFAULT_CONFIG_NAME = "config.yaml"
_SEARCH_DIRS = [
    Path(__file__).resolve().parent,          # mismo directorio que config_loader.py
    Path(__file__).resolve().parent.parent,   # directorio padre (por si está en bigdata/)
    Path.cwd(),                               # directorio de trabajo actual
]


# ── Dataclasses de configuración ──────────────────────────────────────────────

@dataclass
class CentroConfig:
    lat: float
    lon: float


@dataclass
class CiudadConfig:
    nombre: str
    departamento: str
    codigo_dane: str
    unidad_territorial: str   # "comuna", "localidad", "corregimiento", …
    crs: str
    centro: CentroConfig


@dataclass
class RutasConfig:
    datos_climaticos: str
    datos_sociales: str
    procesados: str
    modelos: str
    logs: str
    geo: str

    def resolve(self, root: Path) -> "RutasConfig":
        """Devuelve una nueva instancia con todas las rutas convertidas a absolutas."""
        return RutasConfig(
            datos_climaticos=str(root / self.datos_climaticos),
            datos_sociales=str(root / self.datos_sociales),
            procesados=str(root / self.procesados),
            modelos=str(root / self.modelos),
            logs=str(root / self.logs),
            geo=str(root / self.geo),
        )


@dataclass
class TerritorioConfig:
    id: int
    nombre: str
    lat: float
    lon: float


@dataclass
class CityConfig:
    """Raíz del árbol de configuración. Punto de entrada para el resto del sistema."""
    ciudad: CiudadConfig
    rutas: RutasConfig
    territorios: List[TerritorioConfig]
    catalogo_estaciones: Dict[int, Dict[str, float]]
    # Ruta absoluta al archivo config.yaml cargado
    config_path: Path = field(default_factory=Path)

    # ── Accesos de conveniencia ────────────────────────────────────────────────

    @property
    def unidad_territorial(self) -> str:
        """Shortcut: nombre de la unidad territorial (ej. 'comuna')."""
        return self.ciudad.unidad_territorial

    @property
    def crs(self) -> str:
        """Shortcut: código EPSG del CRS (ej. 'EPSG:4326')."""
        return self.ciudad.crs

    def rutas_absolutas(self, root: Optional[Path] = None) -> RutasConfig:
        """
        Devuelve RutasConfig con paths absolutos anclados a *root*.
        Si root es None, se usa el directorio padre de config.yaml.
        """
        root = root or self.config_path.parent
        return self.rutas.resolve(root)

    def territorio_por_nombre(self, nombre: str) -> Optional[TerritorioConfig]:
        """Busca un territorio por nombre exacto (case-sensitive)."""
        for t in self.territorios:
            if t.nombre == nombre:
                return t
        return None

    def __repr__(self) -> str:
        return (
            f"CityConfig(ciudad={self.ciudad.nombre!r}, "
            f"unidad_territorial={self.unidad_territorial!r}, "
            f"territorios={len(self.territorios)}, "
            f"estaciones={len(self.catalogo_estaciones)})"
        )


# ── Función pública de carga ───────────────────────────────────────────────────

def get_config(config_path: Optional[str | Path] = None) -> CityConfig:
    """
    Carga y valida config.yaml, devuelve un CityConfig.

    Parámetros
    ----------
    config_path : str | Path | None
        Ruta explícita al archivo YAML. Si es None, busca config.yaml en
        los directorios definidos en _SEARCH_DIRS.

    Lanza
    -----
    FileNotFoundError  si no se encuentra el archivo.
    KeyError / ValueError  si faltan claves obligatorias.
    """
    path = _resolve_path(config_path)

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return _parse(raw, path)


def _resolve_path(config_path: Optional[str | Path]) -> Path:
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"config.yaml no encontrado en: {p}")
        return p.resolve()

    for directory in _SEARCH_DIRS:
        candidate = directory / _DEFAULT_CONFIG_NAME
        if candidate.exists():
            return candidate.resolve()

    searched = "\n  ".join(str(d / _DEFAULT_CONFIG_NAME) for d in _SEARCH_DIRS)
    raise FileNotFoundError(
        f"No se encontró '{_DEFAULT_CONFIG_NAME}'. Rutas buscadas:\n  {searched}"
    )


def _parse(raw: dict, path: Path) -> CityConfig:
    """Construye el árbol de dataclasses a partir del dict YAML."""
    _require_keys(raw, ["ciudad", "rutas", "territorios"], context="raíz")

    # Ciudad
    c = raw["ciudad"]
    _require_keys(c, ["nombre", "departamento", "codigo_dane",
                       "unidad_territorial", "crs", "centro"], context="ciudad")
    centro_raw = c["centro"]
    _require_keys(centro_raw, ["lat", "lon"], context="ciudad.centro")

    ciudad = CiudadConfig(
        nombre=str(c["nombre"]),
        departamento=str(c["departamento"]),
        codigo_dane=str(c["codigo_dane"]),
        unidad_territorial=str(c["unidad_territorial"]),
        crs=str(c["crs"]),
        centro=CentroConfig(lat=float(centro_raw["lat"]), lon=float(centro_raw["lon"])),
    )

    # Rutas
    r = raw["rutas"]
    _require_keys(r, ["datos_climaticos", "datos_sociales", "procesados",
                       "modelos", "logs", "geo"], context="rutas")
    rutas = RutasConfig(
        datos_climaticos=str(r["datos_climaticos"]),
        datos_sociales=str(r["datos_sociales"]),
        procesados=str(r["procesados"]),
        modelos=str(r["modelos"]),
        logs=str(r["logs"]),
        geo=str(r["geo"]),
    )

    # Territorios
    territorios: List[TerritorioConfig] = []
    for i, t in enumerate(raw["territorios"]):
        _require_keys(t, ["id", "nombre", "lat", "lon"],
                      context=f"territorios[{i}]")
        territorios.append(TerritorioConfig(
            id=int(t["id"]),
            nombre=str(t["nombre"]),
            lat=float(t["lat"]),
            lon=float(t["lon"]),
        ))

    if not territorios:
        raise ValueError("La sección 'territorios' no puede estar vacía.")

    # Catálogo de estaciones (opcional — dict int → {lat, lon})
    raw_catalogo = raw.get("catalogo_estaciones", {}) or {}
    catalogo: Dict[int, Dict[str, float]] = {
        int(k): {"lat": float(v["lat"]), "lon": float(v["lon"])}
        for k, v in raw_catalogo.items()
    }

    return CityConfig(
        ciudad=ciudad,
        rutas=rutas,
        territorios=territorios,
        catalogo_estaciones=catalogo,
        config_path=path,
    )


def _require_keys(d: dict, keys: list, context: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(
            f"Claves obligatorias faltantes en '{context}': {missing}\n"
            f"Revisa config.yaml."
        )


# ── CLI de verificación rápida ─────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  Ciudad              : {cfg.ciudad.nombre}")
    print(f"  Departamento        : {cfg.ciudad.departamento}")
    print(f"  Codigo DANE         : {cfg.ciudad.codigo_dane}")
    print(f"  Unidad territorial  : {cfg.unidad_territorial}")
    print(f"  CRS                 : {cfg.crs}")
    print(f"  Centro              : {cfg.ciudad.centro.lat}, {cfg.ciudad.centro.lon}")
    print(f"  Territorios cargados: {len(cfg.territorios)}")
    print(f"  Estaciones catalogo : {len(cfg.catalogo_estaciones)}")
    print(f"  Config desde        : {cfg.config_path}")
    print(sep)
    print("\nRutas relativas:")
    for k, v in vars(cfg.rutas).items():
        print(f"  {k:<20}: {v}")
    print("\nRutas absolutas:")
    abs_rutas = cfg.rutas_absolutas()
    for k, v in vars(abs_rutas).items():
        print(f"  {k:<20}: {v}")
    print("\nPrimeros 3 territorios:")
    for t in cfg.territorios[:3]:
        print(f"  {t.id:>2}. {t.nombre:<20} ({t.lat}, {t.lon})")
    print()
