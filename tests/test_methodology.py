"""Guardrails for the revised IVC forecast and the allocation model."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abm.dispersion import coefficient_of_variation, gini
from abm.historical_allocation import load_historical_allocation
from bigdata.siata_territory_annual import (
    EXCLUDED_MODEL_YEARS,
    MIN_COVERAGE,
    aggregate_territories,
    annotate_observed_table,
    hours_in_year,
    load_catalog,
    load_catalog_records,
    load_territory_polygons,
    main as siata_main,
    resolve_station_territory,
)
from config_loader import get_config
from ml.clima_hook import (
    assert_climate_use,
    composite_ivc,
    fit_minmax_train_only,
    predictor_columns,
)
from ml.lagged_forecast import build_lagged_frame, walk_forward

PANEL = ROOT / "ModeloDatos/ModeloDatos/data/processed/indices_comunas.csv"
WEIGHTS = {
    "idx_desempleo": 1.0,
    "idx_habitat": 1.0,
    "idx_educacion": 1.0,
    "idx_riesgo_clima": 0.5,
    "idx_pobreza": 1.0,
    "idx_tejido_social": 1.0,
}
SUBS = list(WEIGHTS)


def _urban():
    spec = importlib.util.spec_from_file_location("urban_model_test", ROOT / "abm" / "04_urban_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.logger.setLevel(40)
    return mod


class CircularityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(PANEL, encoding="utf-8-sig")

    def test_ivc_is_the_coded_weighted_mean_not_a_uniform_mean(self):
        d = self.df.dropna(subset=["ivc", *SUBS])
        recon = composite_ivc(d, WEIGHTS)
        self.assertLess(float(np.max(np.abs(recon - d["ivc"]))), 1e-12)
        uniform = {k: 1.0 for k in WEIGHTS}
        recon_u = composite_ivc(d, uniform)
        self.assertGreater(float(np.max(np.abs(recon_u - d["ivc"]))), 0.01)

    def test_climate_subindex_is_constant(self):
        self.assertEqual(self.df["idx_riesgo_clima"].nunique(dropna=False), 1)
        self.assertEqual(float(self.df["idx_riesgo_clima"].iloc[0]), 0.5)

    def test_lagged_row_uses_features_from_t_and_ivc_from_t_plus_one(self):
        frame = build_lagged_frame(self.df)
        row = frame.iloc[10]
        cur = self.df[(self.df["Comuna"] == row["Comuna"]) & (self.df["Año"] == row["anio"])].iloc[0]
        nxt = self.df[(self.df["Comuna"] == row["Comuna"]) & (self.df["Año"] == row["anio"] + 1)].iloc[0]
        self.assertAlmostEqual(float(row["ivc_t"]), float(cur["ivc"]))
        self.assertAlmostEqual(float(row["ivc_next"]), float(nxt["ivc"]))
        self.assertAlmostEqual(float(row["idx_pobreza"]), float(cur["idx_pobreza"]))
        self.assertEqual(int(row["anio_objetivo"]), int(row["anio"]) + 1)

    def test_walk_forward_does_not_train_on_the_test_year(self):
        frame = build_lagged_frame(self.df)
        pred, specs = walk_forward(frame, ["idx_pobreza", "idx_habitat"], "Comuna", "linear", min_train_years=3)
        self.assertTrue(specs)
        for spec in specs:
            self.assertNotIn(spec.test_year, spec.train_years)
            self.assertGreaterEqual(len(spec.train_years), 3)
        # Persistence on a test row is that row's own IVC_t.
        pers, _ = walk_forward(frame, [], "Comuna", "persistence", min_train_years=3)
        self.assertTrue(np.allclose(pers["y_pred"], pers["ivc_t"]))


class ClimateHookTests(unittest.TestCase):
    def test_rainfall_cannot_be_inside_the_index_and_a_predictor(self):
        with self.assertRaises(ValueError):
            predictor_columns("subindice", ["precip_anual_mm"])
        with self.assertRaises(ValueError):
            predictor_columns("ninguno", ["precip_anual_mm"])
        with self.assertRaises(ValueError):
            assert_climate_use("both")
        cols = predictor_columns("predictor", ["precip_anual_mm"])
        self.assertIn("precip_anual_mm", cols)
        self.assertNotIn("idx_riesgo_clima", cols)
        self.assertNotIn("precip_anual_mm", predictor_columns("subindice"))

    def test_scaler_does_not_see_the_test_extreme(self):
        train = np.array([[0.0], [1.0]])
        test = np.array([[100.0]])
        _, scaled_test, meta = fit_minmax_train_only(train, test)
        self.assertEqual(meta["max"], [1.0])
        self.assertGreater(float(scaled_test[0, 0]), 1.0)

    def test_fixture_rain_is_refused_as_a_model_input(self):
        with self.assertRaises(RuntimeError):
            build_lagged_frame(
                self_panel(),
                rain=pd.DataFrame({
                    "unidad_territorial": ["Popular"],
                    "anio": [2018],
                    "precip_anual_mm": [10.0],
                    "fuente": ["fixture"],
                }),
                rain_feature_cols=["precip_anual_mm"],
            )


def self_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL, encoding="utf-8-sig")


class AllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _urban()

    def _run(self, ranking, saturation, years=3):
        model = self.mod.UrbanModel(
            strategy="focalizado",
            ivc_data={"A": 0.80, "B": 0.55, "C": 0.20},
            n_years=years,
            annual_budget=300.0,
            seed=42,
            rho=0.90,
            shap_priority={"A": 1.0, "B": 0.0, "C": -1.0},
            gamma=0.08,
            ranking_rule=ranking,
            n_priority=1,
            reparto_top="igual",
            saturation_active=saturation,
            noise_sd=0.0,
        )
        return model.run_simulation()

    def test_dynamic_rank_changes_and_fixed_shap_does_not(self):
        dynamic = self._run("dynamic_ivc", False)
        fixed = self._run("fixed_shap", False)
        ut = "comuna"
        dyn_y3 = dynamic[dynamic["year"] == 3].set_index(ut)["investment_allocated"]
        fix_y3 = fixed[fixed["year"] == 3].set_index(ut)["investment_allocated"]
        self.assertEqual(dyn_y3.idxmax(), "B")
        self.assertEqual(fix_y3.idxmax(), "A")
        self.assertAlmostEqual(float(dyn_y3.sum()), 300.0, places=6)

    def test_saturation_flag_changes_the_low_vulnerability_path(self):
        ivc = {"Low": 0.15, "High": 0.85}
        paths = {}
        for flag in (False, True):
            model = self.mod.UrbanModel(
                strategy="distribuido",
                ivc_data=ivc,
                n_years=5,
                annual_budget=200.0,
                seed=1,
                gamma=0.05,
                saturation_active=flag,
                noise_sd=0.0,
            )
            df = model.run_simulation()
            last = df[df["year"] == 5].set_index("comuna")["ivc_current"]
            paths[flag] = float(last["Low"])
        self.assertGreater(paths[True], paths[False])

    def test_historical_file_missing_is_a_skip_not_a_crash(self):
        mapping, status = load_historical_allocation(ROOT / "data/external/does_not_exist.csv")
        self.assertEqual(mapping, {})
        self.assertEqual(status["state"], "missing")

    def test_historical_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inv.csv"
            pd.DataFrame({
                "unidad_territorial": ["Popular", "Popular"],
                "anio": [2013, 2014],
                "inversion": [10.0, 25.0],
            }).to_csv(path, index=False)
            mapping, status = load_historical_allocation(path)
        self.assertEqual(status["state"], "loaded")
        self.assertEqual(mapping[2014]["Popular"], 25.0)

    def test_published_cv_uses_population_std(self):
        oos = pd.read_csv(ROOT / "models/ivc_predictions_oos.csv", encoding="utf-8-sig")
        latest = oos.sort_values("Año").groupby("Comuna").last()["ivc_predicho"]
        self.assertAlmostEqual(coefficient_of_variation(latest), 0.18753226555900998, places=8)
        self.assertGreater(gini(latest), 0.0)


class RainfallProcessorTests(unittest.TestCase):
    def test_fixture_splits_interior_stations_from_idw(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "territorio.csv"
            siata_main([
                "--mode", "fixture",
                "--out", str(out),
            ])
            df = pd.read_csv(out)
        self.assertEqual(list(df.columns[:6]), [
            "unidad_territorial", "anio", "precip_anual_mm", "dias_ge_50mm",
            "max_24h_mm", "max_horario_mm",
        ])
        y2019 = df[df["anio"] == 2019].set_index("unidad_territorial")
        cristobal = y2019.loc["Corregimiento de San Cristóbal"]
        america = y2019.loc["La América"]
        popular = y2019.loc["Popular"]
        self.assertEqual(cristobal["metodo_agregacion"], "estaciones_interior")
        self.assertAlmostEqual(float(cristobal["precip_anual_mm"]), 65.0, places=6)
        # Two observed hours do not make a complete day. The missing hours
        # are not counted as dry, so the day is not a day ≥ 50 mm and there
        # is no valid 24-hour window.
        self.assertEqual(float(cristobal["dias_ge_50mm"]), 0.0)
        self.assertTrue(pd.isna(cristobal["max_24h_mm"]))
        self.assertAlmostEqual(float(cristobal["max_horario_mm"]), 60.0, places=6)
        self.assertEqual(america["metodo_agregacion"], "estaciones_interior")
        self.assertAlmostEqual(float(america["precip_anual_mm"]), 6.0, places=6)
        self.assertEqual(float(america["dias_ge_50mm"]), 0.0)
        self.assertEqual(popular["metodo_agregacion"], "idw")
        y2020 = df[df["anio"] == 2020].set_index("unidad_territorial")
        self.assertEqual(y2020.loc["La América", "metodo_agregacion"], "idw")
        self.assertEqual(y2020.loc["Corregimiento de San Cristóbal", "metodo_agregacion"], "estaciones_interior")
        self.assertTrue((df["fuente"] == "fixture").all())
        # A few hours of fixture data are not an annual total.
        self.assertFalse(bool(df["elegible"].any()))
        self.assertTrue(df["precip_anual_ajustada_mm"].isna().all())


class RainQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.read_csv(ROOT / "data/processed/siata_territorio_anual.csv")
        cls.raw = raw
        cls.adj = raw if "elegible" in raw.columns else annotate_observed_table(raw)

    def test_leap_year_hours(self):
        self.assertEqual(hours_in_year(2024), 8784)
        self.assertEqual(hours_in_year(2023), 8760)
        self.assertEqual(hours_in_year(1900), 8760)

    def test_popular_2012_is_idw_because_its_gauge_is_not_yet_in_the_file(self):
        row = self.raw[(self.raw["unidad_territorial"] == "Popular") & (self.raw["anio"] == 2012)].iloc[0]
        self.assertEqual(row["metodo_agregacion"], "idw")
        self.assertNotIn("274", str(row["estaciones_ids"]).split(","))
        later = self.raw[(self.raw["unidad_territorial"] == "Popular") & (self.raw["anio"] == 2019)].iloc[0]
        self.assertEqual(later["metodo_agregacion"], "estaciones_interior")
        self.assertEqual(str(later["estaciones_ids"]), "274")

    def test_high_coverage_totals_match_medellin_climatology(self):
        interior = self.raw[
            (self.raw["metodo_agregacion"] == "estaciones_interior")
            & (self.raw["cobertura_horaria_media"] >= 0.8)
        ]
        median = float(interior["precip_anual_mm"].median())
        self.assertGreater(median, 1400)
        self.assertLess(median, 2200)
        y2012 = self.raw[self.raw["anio"] == 2012]
        y2026 = self.raw[self.raw["anio"] == 2026]
        self.assertLess(float(y2012["cobertura_horaria_media"].median()), 0.5)
        self.assertLess(float(y2026["cobertura_horaria_media"].median()), 0.25)

    def test_coverage_adjustment_is_exact_for_a_single_station(self):
        adj = annotate_observed_table(self.raw.drop(
            columns=[c for c in ("precip_anual_ajustada_mm", "dias_ge_50mm_ajustada", "elegible", "ajuste", "cobertura_minima") if c in self.raw.columns]
        ))
        row = adj[(adj["unidad_territorial"] == "Popular") & (adj["anio"] == 2019)].iloc[0]
        self.assertTrue(bool(row["elegible"]))
        self.assertAlmostEqual(
            float(row["precip_anual_ajustada_mm"]),
            float(row["precip_anual_mm"]) / min(1.0, float(row["cobertura_horaria_media"])),
            places=6,
        )
        partial = adj[(adj["unidad_territorial"] == "Popular") & (adj["anio"] == 2012)].iloc[0]
        self.assertFalse(bool(partial["elegible"]))
        self.assertTrue(pd.isna(partial["precip_anual_ajustada_mm"]))
        self.assertEqual(MIN_COVERAGE, 0.80)
        self.assertEqual(EXCLUDED_MODEL_YEARS, frozenset({2012, 2026}))

    def test_partial_years_are_flagged_even_when_coverage_is_high(self):
        frame = pd.DataFrame({
            "unidad_territorial": ["Popular", "Popular"],
            "anio": [2012, 2026],
            "precip_anual_mm": [1800.0, 400.0],
            "dias_ge_50mm": [2.0, 0.0],
            "cobertura_horaria_media": [0.95, 0.95],
        })
        adj = annotate_observed_table(frame)
        self.assertFalse(bool(adj["elegible"].any()))
        self.assertTrue(adj["precip_anual_ajustada_mm"].isna().all())
        self.assertEqual(set(adj["motivo_exclusion"]), {"anio_parcial"})

    def test_station_352_enters_popular_by_polygon_not_by_the_official_list(self):
        official = json.loads((ROOT / "data/external/asignacion_estacion_UT_oficial.json").read_text())
        official = {int(k): v for k, v in official.items()}
        self.assertNotIn(352, official)
        catalog = load_catalog_records(ROOT / "data/external/pluvio_siata_catalogo.json")
        polygons = load_territory_polygons(ROOT / "data/geo/comunas.geojson")
        names = [p["nombre"] for p in polygons]
        territory, source = resolve_station_territory(352, official, catalog, polygons, names)
        self.assertEqual(territory, "Popular")
        self.assertEqual(source, "punto_en_poligono")
        # A listed station keeps the official territory even if a polygon would differ.
        forced = dict(official)
        forced[352] = "Santa Cruz"
        territory, source = resolve_station_territory(352, forced, catalog, polygons, names)
        self.assertEqual((territory, source), ("Santa Cruz", "oficial"))

    def test_aggregate_groups_station_352_into_popular(self):
        """Polygon placement survives aggregation; the official JSON must not overwrite it."""
        cfg = get_config()
        territories = [{"nombre": t.nombre, "lat": t.lat, "lon": t.lon} for t in cfg.territorios]
        catalog = load_catalog(ROOT / "data/external/pluvio_siata_catalogo.json")
        official = {
            int(k): v
            for k, v in json.loads(
                (ROOT / "data/external/asignacion_estacion_UT_oficial.json").read_text()
            ).items()
        }
        self.assertNotIn(352, official)
        station = pd.DataFrame({
            "estacion_id": [274, 352],
            "anio": [2019, 2019],
            "precip_anual_mm": [1556.512, 1800.0],
            "dias_ge_50mm": [1.0, 2.0],
            "max_24h_mm": [50.292, 60.0],
            "max_horario_mm": [49.53, 30.0],
            "cobertura_horaria": [0.970434, 0.95],
            "unidad_territorial_oficial": ["Popular", "Popular"],
            "fuente_asignacion": ["oficial", "punto_en_poligono"],
        })
        out = aggregate_territories(station, official, catalog, territories, 2.0, "full")
        row = out[(out["unidad_territorial"] == "Popular") & (out["anio"] == 2019)].iloc[0]
        ids = set(str(row["estaciones_ids"]).split(","))
        self.assertEqual(ids, {"274", "352"})
        self.assertEqual(row["metodo_agregacion"], "estaciones_interior")
        self.assertEqual(int(row["n_estaciones_elegibles"]), 2)
        self.assertTrue(bool(row["elegible"]))
        expected = ((1556.512 / 0.970434) + (1800.0 / 0.95)) / 2
        self.assertAlmostEqual(float(row["precip_anual_ajustada_mm"]), expected, places=3)


if __name__ == "__main__":
    unittest.main()
