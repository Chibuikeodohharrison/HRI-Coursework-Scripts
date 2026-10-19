"""
Helper for speech3.py: speak a reply without cut-off AND without long delays.

Findings from testing:
  * speech was heard fully at 18 words but cut off at 39 words;
  * speak(blocking=True) returns before playback ends;
  * every speak() call costs roughly 4 s (voice generation + upload to Marty)
    regardless of length, so FEWER calls = a faster response.

Strategy: split at full stops, then GROUP whole sentences into chunks of up to
GROUP_UP_TO_WORDS words, so most replies need only one or two speak() calls.
"""
import re
import threading
import time
import asyncio
import os
import tempfile
import edge_tts
import pygame

# Sentences are grouped into one speak() call until this many words.
# 18 words was heard fully in testing. Set to 0 for strictly one sentence per call.
GROUP_UP_TO_WORDS = 18

# A single sentence longer than this is split at commas as a last resort
# (39 words was cut off). None = never split inside a sentence.
SENTENCE_HARD_LIMIT = 25

WORDS_PER_SECOND = 2.5        # for the wait after speak() returns early; tune it
GAP_AFTER_CHUNK = 0.2         # seconds between chunks

# "during" = one short gesture starts at the same time as the speech (no delay)
# "before" = gesture first, then speak (adds ~1.2 s)   "none" = no gesture
GESTURE_MODE = "during"
SHOW_TIMING = True

# ---------------- Laptop speaker output ----------------
VOICE = "en-GB-RyanNeural"
# Alternative: "en-GB-SoniaNeural"



def speak_on_laptop(text):
    """Generate speech and play it through the laptop's default audio output."""
    temp_path = None

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        async def generate_audio():
            communicator = edge_tts.Communicate(text, VOICE)
            await communicator.save(temp_path)

        print("[TTS] Generating audio...")
        asyncio.run(generate_audio())

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError("Generated audio file is missing or empty.")

        print(f"[TTS] Audio generated: {os.path.getsize(temp_path)} bytes")
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        print("[TTS] Playback started.")

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

        print("[TTS] Playback finished.")

    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except pygame.error:
            pass

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def split_into_sentences(reply):
    """Split after . ! ? followed by a space, and at line breaks (keeps 3.5 intact)."""
    parts = re.split(r'(?<=[.!?])\s+|\n+', reply.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_long_sentence(sentence, limit):
    words = sentence.split()
    if limit is None or len(words) <= limit:
        return [sentence]
    pieces, current = [], ""
    for part in re.split(r'(?<=,)\s+', sentence):
        candidate = (current + " " + part).strip()
        if current and len(candidate.split()) > limit:
            pieces.append(current)
            current = part
        else:
            current = candidate
    if current:
        pieces.append(current)
    final = []
    for p in pieces:
        w = p.split()
        for i in range(0, len(w), limit):
            final.append(" ".join(w[i:i + limit]))
    return final


def split_for_speech(reply):
    """Sentences first, then group whole sentences into chunks up to GROUP_UP_TO_WORDS."""
    sentences = []
    for s in split_into_sentences(reply):
        sentences.extend(_split_long_sentence(s, SENTENCE_HARD_LIMIT))

    if not GROUP_UP_TO_WORDS:
        return sentences

    chunks, current = [], ""
    for s in sentences:
        candidate = (current + " " + s).strip()
        if current and len(candidate.split()) > GROUP_UP_TO_WORDS:
            chunks.append(current)
            current = s
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _one_gesture(marty):
    try:
        marty.arms(30, 0, 500)
        time.sleep(0.6)
        marty.arms(0, 0, 500)
    except Exception as e:
        print("Gesture failed:", e)


def speak_reply(marty, reply, gestures=True, log_fn=None):
    """Speak a reply in as few calls as is safe; wait for each to finish playing."""
    t_total = time.time()

    if gestures and GESTURE_MODE == "before":
        _one_gesture(marty)
        time.sleep(0.6)

    chunks = split_for_speech(reply)
    for n, chunk in enumerate(chunks):
        if gestures and GESTURE_MODE == "during" and n == 0:
            threading.Thread(target=_one_gesture, args=(marty,), daemon=True).start()

        start = time.time()
        try:
            speak_on_laptop(chunk)
        except Exception as e:
            print("Speak failed:", e)
            if log_fn:
                log_fn("speak_error", str(e))

        elapsed = time.time() - start
        words = len(chunk.split())
        if SHOW_TIMING:
            print(f"  [timing] chunk {n + 1}/{len(chunks)} ({words} words): "
                  f"speak() took {elapsed:.1f}s")

        time.sleep(GAP_AFTER_CHUNK)

    if SHOW_TIMING:
        print(f"  [timing] total speaking time: {time.time() - t_total:.1f}s")
    if log_fn:
        log_fn("speak_total_seconds", f"{time.time() - t_total:.1f}")


if __name__ == "__main__":
    demo = ("Great question. To solve 2x + 3 = 11, subtract 3 from both sides, "
            "which gives 2x = 8. Then divide both sides by 2 to get x = 4. "
            "Does that make sense?")
    for i, c in enumerate(split_for_speech(demo), 1):
        print(i, f"({len(c.split())} words):", c)