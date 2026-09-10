"""Create trusted application-menu and desktop launchers for this installation."""

from pathlib import Path
import os
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
icon = root / "assets/locaii.svg"
entry = f"""[Desktop Entry]
Type=Application
Name=loCAII
Comment=Local AI workbench: chat, files, images and video
Exec="{root}/launch.sh"
Icon={icon}
Terminal=false
Categories=Utility;Development;Graphics;
StartupNotify=true
StartupWMClass=loCAII
"""
locations = [
    root / "loCAII.desktop",
    Path.home() / ".local/share/applications/loCAII.desktop",
]
try:
    desktop = subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip()
    if desktop:
        locations.append(Path(desktop) / "loCAII.desktop")
except (FileNotFoundError, subprocess.SubprocessError):
    pass
for location in locations:
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(entry)
    location.chmod(0o755)
    if shutil.which("gio"):
        subprocess.run(
            ["gio", "set", str(location), "metadata::trusted", "true"],
            capture_output=True,
        )
    print(location)
