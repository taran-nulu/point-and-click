"""Deciding which hand aims and which hand clicks.

With two hands visible you cannot just take whatever MediaPipe ranked first,
and you cannot re-decide every frame either: when both hands are open they look
alike, so a resolver that picks fresh each frame would swap roles constantly and
the cursor would jump between your hands mid-gesture.

So the assignment is **sticky**. It is learned once and only revisited when the
hand holding it disappears.

How the pointing hand is found
------------------------------
- **Pinned** (``pointing_hand = "left"``/``"right"``): match MediaPipe's
  handedness. There is a subtlety here - MediaPipe documents handedness as
  assuming a mirrored selfie input, and we mirror the frame before detection, so
  the label should already be the user's real hand. That is an assumption, not
  something we can verify without a hand in frame, so the HUD prints the
  resolved assignment and pinning is always available as the escape hatch.
- **Auto**: the hand whose index finger is extended relative to its other three
  fingers. A pointing hand scores clearly positive, a fist clearly negative, so
  the margin is wide.
"""

from __future__ import annotations

from .gestures import pointer_score

# How much one hand must out-score the other before we are willing to swap roles.
# Large on purpose: a wrong swap is far more disruptive than a late swap, and the
# gesture itself is usually decisive anyway.
SWITCH_MARGIN = 0.6


class RoleResolver:
    """Maps a frame's hands onto (pointing, clicking)."""

    def __init__(self, pointing_hand: str = "auto") -> None:
        if pointing_hand not in ("auto", "left", "right"):
            raise ValueError(
                f"pointing_hand must be auto/left/right, got {pointing_hand!r}"
            )
        self.pref = pointing_hand
        self.side: str | None = None
        """Handedness of the pointing hand, once learned. None while unknown."""
        self.source: str = "none"
        """How it was decided, for the HUD: pinned/auto/none."""

    # ------------------------------------------------------------------ utils

    def _match_pref(self, hand) -> bool:
        if self.pref == "left":
            return (hand.handedness or "").lower() == "left"
        return (hand.handedness or "").lower() == "right"

    def _remember(self, hand, source: str) -> None:
        if hand is not None and hand.handedness:
            self.side = hand.handedness
        self.source = source

    def reset(self) -> None:
        self.side = None
        self.source = "none"

    # ---------------------------------------------------------------- resolve

    def resolve(self, hands):
        """Return ``(pointing, clicking)`` for this frame; either may be None."""
        if not hands:
            # Keep the learned side: hands briefly vanish all the time, and
            # forgetting it would make the roles flap on every blink.
            return (None, None)

        if len(hands) == 1:
            # Only one hand, so there is nothing to disambiguate.
            if self.pref != "auto":
                # Honour the pin: if the pinned hand is the one present, it is
                # the click hand and there is nothing to aim with. Handing the
                # aim to the click hand would fire phantom clicks.
                if self._match_pref(hands[0]):
                    return (None, hands[0])
                # The pinned aiming hand is out of frame. Do not silently aim
                # with the click hand.
                return (None, None)
            # Learned by gesture: with one hand, assume it is the aiming one,
            # which degrades to a frozen cursor rather than phantom clicks.
            return (hands[0], None)

        # Two hands. Prefer an explicit pin.
        if self.pref != "auto":
            for hand in hands:
                if self._match_pref(hand):
                    other = next(h for h in hands if h is not hand)
                    self._remember(hand, "pinned")
                    return (hand, other)
            # The pinned hand is out of frame: do not silently aim with the other.
            return (None, None)

        scored = []
        for hand in hands:
            score = pointer_score(hand.landmarks)
            if score is not None:
                scored.append((score, hand))
        if not scored:
            return (None, None)

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best = scored[0]

        if self.side is not None:
            # Stickiness: if the hand we already assigned is still here, keep it
            # unless the other is decisively more pointer-like.
            current = next((h for h in hands if h.handedness == self.side), None)
            if current is not None and current is not best:
                current_score = next(
                    (s for s, h in scored if h is current), None
                )
                if current_score is not None and best_score - current_score <= SWITCH_MARGIN:
                    other = next(h for h in hands if h is not current)
                    return (current, other)
            if current is not None:
                other = next(h for h in hands if h is not current)
                self._remember(current, "auto")
                return (current, other)

        # Unknown, or the learned hand vanished. Take the clearer pointer.
        other = next(h for h in hands if h is not best)
        self._remember(best, "auto")
        return (best, other)

    def describe(self) -> str:
        """One-line summary of the current assignment, for the HUD."""
        if self.side is None:
            return "unknown"
        return f"{self.side} ({self.source})"