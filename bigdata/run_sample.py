"""
run_sample.py — Lightweight runner for SIATA pipeline verification.
Loads 01_siata_processor.py via importlib (its filename starts with a digit,
preventing direct import) and overrides SAMPLE_MODE / SAMPLE_N before
calling main(), so only the first SAMPLE_N station files are processed.
"""
import importlib.util
import os

_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_siata_processor.py")
_spec = importlib.util.spec_from_file_location("siata_processor", _script)
siata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(siata)

siata.SAMPLE_MODE = True
siata.SAMPLE_N = 10
siata.main()
