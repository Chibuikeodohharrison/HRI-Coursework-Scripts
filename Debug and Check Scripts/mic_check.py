"""
Diagnose a microphone that "hears nothing".
1) Lists input devices with their CURRENT index numbers.
2) Shows a live volume meter for the one you choose, so you can see whether
   it responds when you speak.
3) Prints what energy threshold the speech recogniser would calibrate to.

Run:  python mic_check.py
"""
import array
import time

import pyaudio
import speech_recognition as sr

CURRENT_INDEX = 9          # the MIC_INDEX your script uses
METER_SECONDS = 6

pa = pyaudio.PyAudio()

print("Input devices (index: name):")
inputs = []
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info.get("maxInputChannels", 0) > 0:
        inputs.append(i)
        marker = "   <-- your current MIC_INDEX" if i == CURRENT_INDEX else ""
        print(f"  {i}: {info['name']}{marker}")

if CURRENT_INDEX not in inputs:
    print(f"\n!! Index {CURRENT_INDEX} is NOT an input device any more. "
          "That is probably your problem. Pick one from the list above.")

choice = input("\nIndex to test (Enter = current): ").strip()
index = int(choice) if choice else CURRENT_INDEX

print(f"\nTesting index {index} for {METER_SECONDS}s. Speak normally now...")
try:
    info = pa.get_device_info_by_index(index)
    rate = int(info["defaultSampleRate"])
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                     input=True, input_device_index=index, frames_per_buffer=1024)
except Exception as e:
    raise SystemExit(f"Could not open this device: {e}\n"
                     "It may be in use by another app, muted, or blocked in Windows "
                     "Settings > Privacy > Microphone.")

peak = 0
end = time.time() + METER_SECONDS
while time.time() < end:
    data = array.array("h", stream.read(1024, exception_on_overflow=False))
    rms = (sum(s * s for s in data) / len(data)) ** 0.5
    peak = max(peak, rms)
    bar = "#" * int(min(rms / 100, 50))
    print(f"\r{rms:7.0f} {bar:<50}", end="")
print(f"\n\nPeak level while speaking: {peak:.0f}")
stream.stop_stream()
stream.close()
pa.terminate()

if peak < 200:
    print("Very quiet: the mic is muted, wrong device, or blocked. "
          "Check Windows Settings > Privacy > Microphone, and Sound settings > Input.")
else:
    print("The mic is picking up sound.")

# What would the recogniser calibrate to?
r = sr.Recognizer()
with sr.Microphone(device_index=index) as source:
    print("\nCalibrating for 2s - stay QUIET...")
    r.adjust_for_ambient_noise(source, duration=2)
print(f"Calibrated energy threshold: {r.energy_threshold:.0f}")
print(f"Your speech peak was {peak:.0f}. "
      + ("Speech is ABOVE the threshold: fine."
         if peak > r.energy_threshold
         else "Speech is BELOW the threshold: that is why nothing is detected. "
              "Set ENERGY_THRESHOLD lower, or re-run in a quieter room."))