"""
Test EVERY input device for 3 seconds each and report its peak level.
Keep talking the whole time ("one two three four ...") until it finishes.

Run:  python mic_scan.py
"""
import array
import time

import pyaudio

SECONDS_PER_DEVICE = 3

pa = pyaudio.PyAudio()
results = []

print("Keep talking continuously until the scan finishes...\n")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info.get("maxInputChannels", 0) <= 0:
        continue
    name = info["name"]
    try:
        rate = int(info["defaultSampleRate"])
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                         input_device_index=i, frames_per_buffer=1024)
    except Exception as e:
        print(f"{i:>3}  {name[:50]:<50}  cannot open ({str(e)[:40]})")
        continue

    peak = 0.0
    end = time.time() + SECONDS_PER_DEVICE
    while time.time() < end:
        data = array.array("h", stream.read(1024, exception_on_overflow=False))
        rms = (sum(s * s for s in data) / len(data)) ** 0.5
        peak = max(peak, rms)
    stream.stop_stream()
    stream.close()
    results.append((i, name, peak))
    print(f"{i:>3}  {name[:50]:<50}  peak {peak:7.0f}")

pa.terminate()

print("\nBest candidates:")
for i, name, peak in sorted(results, key=lambda r: -r[2])[:3]:
    print(f"  index {i}: {name}  (peak {peak:.0f})")
print("\nA real microphone hearing speech usually shows a peak of several hundred or more.")