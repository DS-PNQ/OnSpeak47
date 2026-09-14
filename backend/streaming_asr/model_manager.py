"""Model manager + fake streaming engines — mirrors ZipformerModelManager.java.

RAM buckets (spec §5, MUST 7): >=8GB → vi+en+zh resident; 6GB → vi+en;
<6GB → vi only. FakeEngine feeds scripted transcripts for local tests.
"""
from dataclasses import dataclass, field


@dataclass
class StreamingAsrEngine:
    lang: str


class FakeEngine(StreamingAsrEngine):
    """Deterministic scripted engine for tests (mirrors FakeEngine in Java)."""

    def __init__(self, lang: str):
        super().__init__(lang)
        self.text = ""
        self.confidence = 0.5
        self.fed_samples = 0
        self.script: list = []  # optional per-chunk scripted partials
        self.decode_hook = None  # fn(fed_samples) -> (text, conf), applied on decode
        self.is_fake = True

    def inject_partial(self, text: str, confidence: float = 0.8) -> None:
        self.text = text or ""
        self.confidence = confidence

    def accept_audio(self, samples) -> None:
        self.fed_samples += len(samples)
        if self.script:
            text, conf = self.script.pop(0)
            self.text, self.confidence = text, conf

    def decode_available(self) -> None:
        if self.decode_hook is not None:
            try:
                text, conf = self.decode_hook(self.fed_samples)
                if text:
                    self.text, self.confidence = text, conf
            except Exception:
                pass

    def is_ready_to_decode(self, scheduler_samples: int = 2560) -> bool:
        return self.fed_samples >= scheduler_samples

    def partial(self) -> tuple:
        return self.text, self.confidence

    def final(self) -> tuple:
        return self.text, self.confidence

    def reset(self) -> None:
        self.text = ""
        self.fed_samples = 0


class ZipformerModelManager:
    def __init__(self, total_mem_gb: float = 8.0, factory=None):
        self.total_mem_gb = total_mem_gb
        self.factory = factory or (lambda lang: FakeEngine(lang))
        self._resident: dict = {}

    def preload_for_device(self) -> list:
        if self.total_mem_gb >= 8:
            langs = ["vi", "en", "zh"]
        elif self.total_mem_gb >= 6:
            langs = ["vi", "en"]
        else:
            langs = ["vi"]
        return [self.get(l) for l in langs]

    def get(self, lang: str) -> FakeEngine:
        if lang not in self._resident:
            self._resident[lang] = self.factory(lang)
        return self._resident[lang]

    def is_resident(self, lang: str) -> bool:
        return lang in self._resident

    def trim_to(self, active: str, candidate) -> None:
        for lang in [l for l in self._resident if l not in (active, candidate)]:
            del self._resident[lang]
