# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

paddleocr_datas, paddleocr_binaries, paddleocr_hiddenimports = collect_all('paddleocr')

a = Analysis(
    ['MagiaTimeline.py'],
    pathex=[],
    binaries=[
        ('venv/Lib/site-packages/paddle/libs/', 'paddle/libs/'),
    ] + paddleocr_binaries,
    datas=[
        ('README.md', 'move_to_root'),
        ('src.mp4', 'move_to_root'),
        ('ConfigSchema.json', 'move_to_root'),
        ('config.yml', 'move_to_root'),
        ('template.asst', 'move_to_root'),
        ('PaddleOCRModels/', 'move_to_root/PaddleOCRModels/'),
    ] + paddleocr_datas,
    hiddenimports=[] + paddleocr_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MagiaTimeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo/MagiaTimeline-Logo-Transparent.png',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MagiaTimeline',
)
