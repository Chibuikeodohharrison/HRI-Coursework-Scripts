"""
Helper for speech3.py: speak a reply one SENTENCE at a time so Marty's audio
is not cut off and the pauses fall where the punctuation is.

Findings from testing: Marty's speech was heard fully at 18 words but cut off at
39 words, and speak(blocking=True) returns before playback has finished.
So we (1) speak sentence by sentence and (2) wait for each to finish.
"""
import re
import time

# A sentence longer than this is split at its commas as a last resort.
# 18 words worked and 39 failed in testing, so stay below ~30.
# Set to None to NEVER split inside a sentence (risk: long sentences cut off).
SENTENCE_HARD_LIMIT = 25

WORDS_PER_SECOND = 3        # tune with a stopwatch (see notes)
GAP_AFTER_SENTENCE = 0.1      # pause between sentences (seconds)


def split_into_sentences(reply):
    """Split at . ! ? followed by a space, and at line breaks.

    Decimals like 3.5 and 'x = 2.5' are not split, because the full stop is not
    followed by a space.
    """
    parts = re.split(r'(?<=[.!?])\s+|\n+', reply.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_long_sentence(sentence, limit):
    """Last resort: break an over-long sentence at commas, else at the word limit."""
    words = sentence.split()
    if len(words) <= limit:
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

    final = []                          # hard split anything still too long
    for p in pieces:
        w = p.split()
        for i in range(0, len(w), limit):
            final.append(" ".join(w[i:i + limit]))
    return final


def split_for_speech(reply, hard_limit=SENTENCE_HARD_LIMIT):
    """One chunk per sentence; only very long sentences are broken up."""
    chunks = []
    for sentence in split_into_sentences(reply):
        if hard_limit is None:
            chunks.append(sentence)
        else:
            chunks.extend(_split_long_sentence(sentence, hard_limit))
    return chunks


def speak_reply(marty, reply, gestures=True, log_fn=None):
    """Speak a reply sentence by sentence, waiting for each to finish playing.

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
        time.sleep(GAP_AFTER_SENTENCE)


if __name__ == "__main__":
    # Quick offline check of the splitting (no robot needed)
    demo = ("Great question. To solve 2x + 3 = 11, subtract 3 from both sides, "
            "which gives 2x = 8, and then divide both sides by 2 to get x = 4. "
            "Does that make sense?")
    for i, c in enumerate(split_for_speech(demo), 1):
        print(i, f"({len(c.split())} words):", c)