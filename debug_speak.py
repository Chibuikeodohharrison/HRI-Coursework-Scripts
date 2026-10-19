"""
Find out WHY martypy's speak fails.
Checks 1) whether ffmpeg/ffprobe are visible to this Python process, and
2) what bytes the speech service actually returned (valid mp3 or an error page?).
Run:  python debug_speak.py
"""
import io
import shutil
import subprocess

from martypy import Marty
from pydub import AudioSegment

MARTY_IP = "10.53.109.234"      # <-- CHANGE to Marty's real IP

# 1) Are the tools visible to THIS process?
print("1) Tool check")
for tool in ("ffmpeg", "ffprobe"):
    path = shutil.which(tool)
    print(f"   {tool}: {path}")
    if path:
        out = subprocess.run([path, "-version"], capture_output=True, text=True)
        print("     ", out.stdout.splitlines()[0] if out.stdout else out.stderr[:100])

# 2) Capture what the speech service returned, before pydub parses it
print("2) Capturing the service response")
_original = AudioSegment.from_file


def _spy(file, *args, **kwargs):
    data = file.getvalue() if isinstance(file, io.BytesIO) else b""
    print(f"   received {len(data)} bytes")
    print("   first 120 bytes:", data[:120])
    with open("debug_response.bin", "wb") as f:
        f.write(data)
    print("   saved to debug_response.bin")
    return _original(file, *args, **kwargs)


AudioSegment.from_file = staticmethod(_spy)

m = Marty("wifi", MARTY_IP)
try:
    m.speak("Hello there", blocking=True)
    print("   speak worked")
except Exception as e:
    print("   speak failed:", type(e).__name__, e)
m.close()