from __future__ import annotations

import argparse
from pathlib import Path

from agent import BusinessResearchAgent, LocalQwenModel, WebSearchTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Business research AI agent."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Markdown file containing the research request.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.md"),
        help="Markdown file for the generated briefing (default: output.md).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of web-search results to collect (default: 5).",
    )
    return parser

def read_request(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")
    if path.suffix.lower() != ".md":
        raise SystemExit("Input file must be a Markdown (.md) file.")

    request = path.read_text(encoding="utf-8").strip()
    if not request:
        raise SystemExit("The research request is empty.")
    
    return request

def write_output(path: Path, content: str) -> None:
    if path.suffix.lower() != ".md":
        raise SystemExit("Output file must be a Markdown (.md) file.")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def main() -> None:
    args = build_parser().parse_args()
    request = read_request(args.input_file)

    agent = BusinessResearchAgent(
        search_tool=WebSearchTool(),
        model=LocalQwenModel(),
    )
    print("\nResearching...\n")

    briefing = agent.run(
        request,
        max_results=args.max_results,
    )

    write_output(args.output, briefing)
    print(f"Briefing saved to: {args.output}")

if __name__ == "__main__":
    main()
