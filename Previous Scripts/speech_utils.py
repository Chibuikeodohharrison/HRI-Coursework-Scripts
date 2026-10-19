"""
Helper for speech3.py: speak a reply in short chunks so Marty's audio is not cut off.

Findings from testing: Marty's speech was heard fully at 18 words but cut off at
39 words, and speak(blocking=True) returns before playback has finished.
So we (1) split replies into short chunks and (2) wait for each to finish.
"""
import re
import time

MAX_WORDS_PER_CHUNK = 14      # 18 words worked, 39 failed, so stay well under
WORDS_PER_SECOND = 2.5        # tune with a stopwatch (see notes)


def split_for_speech(reply, max_words=MAX_WORDS_PER_CHUNK):
    """Break a reply into short chunks that Marty can speak without cut-off."""
    chunks = []
    for sentence in re.split(r'(?<=[.!?])\s+', reply.strip()):
        words = sentence.split()
        if not words:
            continue
        if len(words) <= max_words:
            chunks.append(sentence)
            continue
        # Long sentence: split at commas/conjunctions, then re-join up to the limit
        parts = re.split(r'(?<=,)\s+|\s+(?=(?:and|then|so|but|because)\b)', sentence)
        current = ""
        for part in parts:
            candidate = (current + " " + part).strip()
            if current and len(candidate.split()) > max_words:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)

    # Safety net: hard-split anything still too long
    final = []
    for c in chunks:
        w = c.split()
        for i in range(0, len(w), max_words):
            final.append(" ".join(w[i:i + max_words]))
    return final


def speak_reply(marty, reply, gestures=True, log_fn=None):
    """Speak a reply chunk by chunk, waiting for each to finish playing.

    log_fn: optional function(event, text) for logging, e.g. log_event.
    """
    if gestures:                       # one short gesture BEFORE speaking
        try:
            marty.arms(30, 0, 500)
            time.sleep(0.6)
            marty.arms(0, 0, 500)
            time.sleep(0.6)
        except Exception as e:
            print("Gesture failed:", e)

    for chunk in split_for_speech(reply):
        start = time.time()
        try:
            ok = marty.speak(chunk, blocking=True)
            if ok is False:
                print("speak() returned False for:", chunk)
        except Exception as e:
            print("Speak failed:", e)
            if log_fn:
                log_fn("speak_error", str(e))

        # speak() returns early, so wait for the rest of the playback
        elapsed = time.time() - start
        expected = len(chunk.split()) / WORDS_PER_SECOND
        if elapsed < expected:
            time.sleep(expected - elapsed)
        time.sleep(0.4)                # small gap between chunks
