from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


@dataclass
class SummaryResult:
    summary_text: str
    num_inputs: int


@dataclass
class ChatResponse:
    answer: str
    used_context: bool


class ReviewSummarizer:
    """Summarize tenant reviews via Transformer pipelines."""

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6") -> None:
        self.model_name = model_name
        self._pipeline = None
        if pipeline is not None:
            try:
                self._pipeline = pipeline("summarization", model=self.model_name)
            except Exception as exc:
                print(f"[ReviewSummarizer] Failed to init HF pipeline: {exc}")
                self._pipeline = None

    def summarize(self, reviews: Iterable[str], max_chars: int = 512) -> SummaryResult:
        texts = [text.strip() for text in reviews if text and text.strip()]
        if not texts:
            raise ValueError("No review text provided for summarization")
        joined = " \n".join(texts)
        truncated = joined[:max_chars]
        if self._pipeline is not None:
            summary = self._pipeline(truncated, max_length=130, min_length=30, do_sample=False)[0][
                "summary_text"
            ]
        else:
            chunks = truncated.split(".")
            summary = ". ".join(chunk.strip() for chunk in chunks[:3] if chunk.strip())
        return SummaryResult(summary_text=summary.strip(), num_inputs=len(texts))


class PropertyChatbot:
    """Simple chatbot for property Q&A using a prompt template."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        self.model_name = model_name
        self._pipeline = None
        if pipeline is not None:
            try:
                self._pipeline = pipeline("feature-extraction", model=self.model_name)
            except Exception as exc:
                print(f"[PropertyChatbot] Failed to init HF pipeline: {exc}")
                self._pipeline = None

    def answer(self, question: str, context: str | None = None) -> ChatResponse:
        if self._pipeline is None:
            fallback = f"根據問題「{question}」，建議您聯繫房東或查看租賃合約詳細資訊。"
            return ChatResponse(answer=fallback, used_context=False)
        # Extremely simple rule-based fallback for demo
        if "租金" in question:
            ans = "租金根據房屋類型、地段與設施而定，通常在 8,000–45,000 元之間。"
        elif "交通" in question:
            ans = "多數房源靠近捷運站或公車站，通勤便利。"
        else:
            ans = "請提供更多問題細節，我會盡力協助。"
        if context:
            ans += f" 參考資訊：{context[:100]}..."
        return ChatResponse(answer=ans, used_context=bool(context))
