from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchTool(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class WebSearchTool:
    """Small wrapper around DDGS so the rest of the agent is tool-agnostic."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError(
                "DDGS is not installed. Run: uv sync"
            ) from exc

        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        title=str(item.get("title", "Untitled source")),
                        url=str(item.get("href", item.get("url", ""))),
                        snippet=str(item.get("body", item.get("snippet", ""))),
                    )
                )
        return results


class LocalQwenModel:
    """Lazy-loading local LLM adapter for Qwen2.5-0.5B-Instruct."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers is not installed. Run: uv sync"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cautious business research assistant. Use only the "
                    "evidence supplied in the prompt. If a fact is not supported, say so."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(
            self._model.device
        )
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=500,
            do_sample=False,
        )
        response_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        return self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()


SEI_CONTEXT = (
    "SEI interests: renewable energy, solar PV, energy storage, EV charging, "
    "heat pumps, energy infrastructure, net-zero projects, and UK/European/Chinese "
    "energy opportunities."
)

OXVALUE_CONTEXT = (
    "OxValue.ai interests: artificial intelligence and software, technology companies, "
    "business valuation, investment and fundraising, investors, and strategic partners."
)


class BusinessResearchAgent:
    def __init__(self, search_tool: SearchTool, model: LanguageModel) -> None:
        self.search_tool = search_tool
        self.model = model

    @staticmethod
    def identify_target(request: str) -> str:
        cleaned = request.strip().rstrip(".?")
        lower = cleaned.lower()
        prefixes = (
            "research ",
            "please research ",
            "research company ",
        )
        for prefix in prefixes:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                break

        separators = (
            " and tell me",
            " and assess",
            " and explain",
            " and determine",
            " for sei",
            " for oxvalue",
        )
        lower_cleaned = cleaned.lower()
        cut_positions = [
            lower_cleaned.find(separator)
            for separator in separators
            if lower_cleaned.find(separator) != -1
        ]
        if cut_positions:
            cleaned = cleaned[: min(cut_positions)]
        return cleaned.strip(" ,:-") or request.strip()

    @staticmethod
    def choose_business_context(request: str) -> tuple[str, str]:
        text = request.lower()
        if "oxvalue" in text and "sei" not in text:
            return "OxValue.ai", OXVALUE_CONTEXT
        if "sei" in text and "oxvalue" not in text:
            return "SEI", SEI_CONTEXT
        return "SEI and OxValue.ai", f"{SEI_CONTEXT}\n{OXVALUE_CONTEXT}"

    @staticmethod
    def _format_evidence(results: Iterable[SearchResult]) -> str:
        blocks = []
        for index, result in enumerate(results, start=1):
            blocks.append(
                f"[{index}] {result.title}\nURL: {result.url}\nSnippet: {result.snippet}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _format_sources(results: Iterable[SearchResult]) -> str:
        lines = ["## Sources"]
        for result in results:
            lines.append(f"- [{result.title}]({result.url})")
        return "\n".join(lines)

    def run(self, request: str, max_results: int = 5) -> str:
        target = self.identify_target(request)
        organisation, business_context = self.choose_business_context(request)

        query = (
            f'"{target}" company business products technology recent developments'
        )
        results = self.search_tool.search(query, max_results=max_results)
        if not results:
            return (
                "# Business Research Briefing\n\n"
                f"## Company / Topic\n\n{target}\n\n"
                "## Result\n\n"
                "No external search results were found, so no evidence-based briefing "
                "can be produced."
            )

        evidence = self._format_evidence(results)
        prompt = f"""
Research request: {request}
Research target: {target}
Business context: {organisation}
{business_context}

External evidence:
{evidence}

Produce a concise Markdown briefing using only the evidence above.

Use exactly these Markdown sections:

# Business Research Briefing

## Company / Topic

## Background

## Main Business, Products or Technology

## Recent Activity

## Relevance to {organisation}

## Limitations / Information Gaps

Keep the relevance assessment modest and evidence-based.

Only state factual claims that are supported by the supplied external evidence.
Do not infer that the company operates in a business area simply because that
area appears in the business context.

If the supplied evidence does not support a claim, explicitly state that there
is insufficient evidence.

Do not add source URLs inside the briefing because the application will append
the sources separately.
""".strip()
        briefing = self.model.generate(prompt)
        return f"{briefing}\n\n{self._format_sources(results)}"
