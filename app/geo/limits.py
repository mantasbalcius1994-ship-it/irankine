"""Serverio ribos (nustatomos /etc/toolbox.env). Numatytosios - 2 branduolių / 4 GB serveriui.
Matavimai: TIN ~1 GB / 1 mln. taškų (su užklausomis), taškų debesis ~24 B / taškas."""
import os

MAX_TIN_POINTS = int(float(os.environ.get("TOOLBOX_MAX_TIN_POINTS", 2_500_000)))
MAX_CLOUD_POINTS = int(float(os.environ.get("TOOLBOX_MAX_CLOUD_POINTS", 40_000_000)))
