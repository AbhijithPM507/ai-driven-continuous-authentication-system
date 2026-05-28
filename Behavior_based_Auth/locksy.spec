# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('web/**', 'web'),
        ('database/**', 'database'),
        ('utils/**', 'utils'),
        ('models/**', 'models'),
        ('config.py', '.'),
    ],
    hiddenimports=[
        'eel',
        'pynput.keyboard',
        'pynput.mouse',
        'tensorflow',
        'sklearn.svm',
        'sklearn.ensemble',
        'sklearn.neighbors',
        'sklearn.linear_model',
        'sklearn.preprocessing',
        'sklearn.metrics',
        'joblib',
        'numpy',
        'scipy.stats',
        'pysqlcipher3',
        'ctypes',
        'sqlite3',
        'json',
        'threading',
        'collections',
        'logging',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'flask',
        'flask_socketio',
        'flask_jwt_extended',
        'flask_cors',
        'flask_bcrypt',
        'bcrypt',
        'jwt',
        'werkzeug',
        'jinja2',
        'markupsafe',
        'itsdangerous',
        'templates',
        'static',
        'test_training',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Locksy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
)

app = BUNDLE(
    exe,
    name='Locksy.app' if sys.platform == 'darwin' else None,
    icon='icon.ico' if Path('icon.ico').exists() and sys.platform == 'darwin' else None,
    bundle_identifier=None,
)
