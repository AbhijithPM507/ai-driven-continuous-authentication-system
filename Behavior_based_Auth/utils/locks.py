import sys
import subprocess
import logging

logger = logging.getLogger(__name__)

def lock_workstation():
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.LockWorkStation()
        logger.info("Workstation locked via Windows API")
    elif sys.platform == 'darwin':
        subprocess.run([
            '/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession',
            '-suspend'
        ])
        logger.info("Workstation locked via macOS CGSession")
    elif sys.platform.startswith('linux'):
        subprocess.run(['xdg-screensaver', 'lock'])
        logger.info("Workstation locked via xdg-screensaver")
    else:
        logger.warning(f"No lock mechanism for platform: {sys.platform}")
