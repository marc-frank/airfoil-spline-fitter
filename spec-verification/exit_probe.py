"""Exit behaviour of spline_fitter.__main__.main() (spec.md 1.1).

Usage: python exit_probe.py idle|fit|code3 [native] [token]
  idle   the window closes 300 ms after it is shown
  fit    loads NACA 2412 and fits first, then closes
  code3  ends the event loop with QApplication.exit(3) instead
The window writes `token` to QSettings key "verify_token" just before it closes;
`python exit_probe.py read` prints that key (and removes it) from a new process.
"""
import os, sys
if "native" not in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "packages", "bspline-varpro", "src")]

if sys.argv[1] == "read":
    from PyQt6.QtCore import QSettings
    s = QSettings("MSFra", "AirfoilSplineFitter")
    print("token", s.value("verify_token", ""))
    s.remove("verify_token")
    sys.exit(0)

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import spline_fitter.__main__ as m
from spline_fitter.gui.ui import SplineFitterWindow

mode = sys.argv[1]
token = next((a for a in sys.argv[2:] if a != "native"), "")
_show = SplineFitterWindow.showMaximized


def show(self):
    _show(self)

    def go():
        if mode == "fit":
            self.new_naca_airfoil("2412", 41, True); self._fitting_controller.on_fit()
            self._project_manager._project_modified = False
        self._settings.setValue("verify_token", token)
        if mode == "code3":
            QApplication.exit(3)
        else:
            self.close()
    QTimer.singleShot(300, go)


SplineFitterWindow.showMaximized = show
m.main()
print("main returned", flush=True)   # never reached: main() ends the process
