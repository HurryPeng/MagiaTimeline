# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT


paddleocr_datas, paddleocr_binaries, paddleocr_hiddenimports = collect_all("paddleocr")
cython_datas, cython_binaries, cython_hiddenimports = collect_all("Cython")
customtkinter_datas, customtkinter_binaries, customtkinter_hiddenimports = collect_all("customtkinter")

a1 = Analysis(
    ["MagiaTimeline.py"],
    pathex=[],
    binaries=[
        ("venv/Lib/site-packages/paddle/libs/", "paddle/libs/"),
    ] + paddleocr_binaries + cython_binaries,
    datas=[
        ("README.md", "move_to_root"),
        ("README-zh_CN.md", "move_to_root"),
        ("ConfigSchema.json", "move_to_root"),
        ("config.yml", "move_to_root"),
        ("template.asst", "move_to_root"),
        ("PaddleOCRModels/", "move_to_root/PaddleOCRModels/"),
        ("logo/", "move_to_root/logo/"),
    ] + paddleocr_datas + cython_datas,
    hiddenimports=[] + paddleocr_hiddenimports + cython_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pkg_resources"],
    optimize=0,
)
pyz1 = PYZ(a1.pure)

exe1 = EXE(
    pyz1,
    a1.scripts,
    [],
    exclude_binaries=True,
    name="MagiaTimeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="logo/MagiaTimeline-Logo-Transparent.png",
)

# --- MagiaTimeline-GUI ---
a2 = Analysis(
    ["MagiaTimeline-GUI.py"],
    pathex=[],
        binaries=[
        ("venv/Lib/site-packages/paddle/libs/", "paddle/libs/"),
    ] + paddleocr_binaries + cython_binaries + customtkinter_binaries,
    datas=paddleocr_datas + cython_datas + customtkinter_datas,
    hiddenimports=[] + paddleocr_hiddenimports + cython_hiddenimports + customtkinter_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pkg_resources"],
    optimize=0,
)
pyz2 = PYZ(a2.pure, a2.zipped_data)

exe2 = EXE(
    pyz2,
    a2.scripts,
    [],
    exclude_binaries=True,
    name="MagiaTimeline-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="logo/MagiaTimeline-Logo-Transparent.png",
)

coll = COLLECT(
    exe1, exe2,
    a1.binaries + a2.binaries,
    a1.datas    + a2.datas,
    strip=False, upx=False, name="MagiaTimeline"
)
