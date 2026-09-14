"""Partial transcript tiers — mirrors PartialTranscriptManager.java (spec §18)."""
from .config import TOKEN_STABLE_UPDATES


class PartialTranscriptManager:
    def __init__(self):
        self._committed = []
        self.speculative = ""
        self._survival: dict = {}
        self._last_spec = ""
        self.finalized = False

    def update_speculative(self, partial) -> None:
        if self.finalized:
            return
        nxt = (partial or "").strip()
        # Transducer partials re-emit the whole hypothesis including already
        # committed words — strip that prefix or it would be committed twice.
        committed_str = " ".join(self._committed)
        if committed_str and nxt.startswith(committed_str):
            nxt = nxt[len(committed_str):].strip()
        if nxt == self._last_spec:
            for k in self._survival:
                self._survival[k] += 1
        else:
            kept = {}
            for tok in nxt.split():
                kept[tok] = self._survival.get(tok, 0) + 1
            self._survival = kept
            self.speculative = nxt
            self._last_spec = nxt
        self._promote_stable_head()

    def _promote_stable_head(self) -> None:
        if not self.speculative:
            return
        toks = self.speculative.split()
        stable = 0
        for tok in toks:
            if self._survival.get(tok, 0) >= TOKEN_STABLE_UPDATES:
                stable += 1
            else:
                break
        if stable == 0:
            return
        head = " ".join(toks[:stable])
        committed_str = " ".join(self._committed)
        if head.startswith(committed_str):
            delta = head[len(committed_str):].strip()
            if delta:
                self._committed.extend(delta.split())
                rest = self.speculative[len(head):].strip()
                self.speculative = rest
                self._last_spec = rest
                for tok in delta.split():
                    self._survival.pop(tok, None)

    def rollback_speculative(self) -> None:
        self.speculative = ""
        self._last_spec = ""
        self._survival.clear()

    def commit(self, tokens) -> None:
        if not tokens or not tokens.strip():
            return
        t = tokens.strip()
        committed_str = " ".join(self._committed)
        if committed_str and t.startswith(committed_str):
            t = t[len(committed_str):].strip()
            if not t:
                return
        self._committed.extend(t.split())
        if self.speculative.startswith(t):
            self.speculative = self.speculative[len(t):].strip()
            self._last_spec = self.speculative
        else:
            self.rollback_speculative()

    def finalize(self, final_text) -> None:
        self._committed = (final_text or "").strip().split() if final_text else []
        self.rollback_speculative()
        self.finalized = True

    def reset(self) -> None:
        self._committed = []
        self.rollback_speculative()
        self.finalized = False

    @property
    def committed(self) -> str:
        return " ".join(self._committed)

    @property
    def display(self) -> str:
        c = self.committed
        if not self.speculative:
            return c
        return self.speculative if not c else c + " " + self.speculative

    @property
    def tier(self) -> str:
        if self.finalized:
            return "FINAL"
        return "SPECULATIVE" if self.speculative else "STABLE"
