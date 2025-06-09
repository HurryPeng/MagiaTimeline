# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

import os
os.environ["PADDLE_PDX_CACHE_HOME"] = "./PaddleOCRModels"

def collect_all_list(package_names):
    extra_datas = []
    extra_binaries = []
    extra_hiddenimports = []
    for package_name in package_names:
        datas, binaries, hiddenimports = collect_all(package_name)
        extra_datas.extend(datas)
        extra_binaries.extend(binaries)
        extra_hiddenimports.extend(hiddenimports)
    return extra_datas, extra_binaries, extra_hiddenimports

extra_datas, extra_binaries, extra_hiddenimports = collect_all_list([
    "paddleocr",
    "paddlex",
    "pypdfium2",
    "customtkinter",
    # paxxled[ocr] hooks
    "ftfy",
    "imagesize",
    "lxml",
    "cv2",
    "openpyxl",
    "premailer",
    "pyclipper",
    "pypdfium2",
    "sklearn",
    "shapely",
    "tokenizers"
])

a1 = Analysis(
    ["MagiaTimeline.py"],
    pathex=[],
    binaries=[
        ("venv/Lib/site-packages/paddle/libs/", "paddle/libs/"),
    ] + extra_binaries,
    datas=[
        ("README.md", "move_to_root"),
        ("README-zh_CN.md", "move_to_root"),
        ("ConfigSchema.json", "move_to_root"),
        ("config.yml", "move_to_root"),
        ("template.asst", "move_to_root"),
        ("PaddleOCRModels/", "move_to_root/PaddleOCRModels/"),
        ("logo/", "move_to_root/logo/"),
    ] + extra_datas,
    hiddenimports=[] + extra_hiddenimports,
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
    ] + extra_binaries,
    datas=extra_datas,
    hiddenimports=[] + extra_hiddenimports,
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
