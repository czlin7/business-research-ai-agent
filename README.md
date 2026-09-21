# Business Research AI Agent

A small Python prototype that researches a company or technology using web search and produces a structured relevance briefing for **SEI** or **OxValue.ai**.

The project demonstrates a simple AI-agent workflow combining a local language model (LLM)  with external web search.

## Features

* Markdown input and output
* Web search using [DDGS](https://pypi.org/project/ddgs/)
* Local inference with [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
* SEI / OxValue.ai business-context assessment
* Multi-step research workflow
* Structured briefing with source links

## Workflow

```text
input.md
    ↓
Research Request
    ↓
Business Context
    ↓
Web Search
    ↓
Retrieved Evidence
    ↓
Local Qwen Model
    ↓
Structured Briefing
    ↓
output.md
```

The agent:

1. reads the research request;
2. identifies the company or topic;
3. selects the relevant business context;
4. searches the web for external evidence;
5. passes the evidence to the local language model;
6. generates a structured relevance briefing;
7. writes the report and source links to Markdown.

## Project Structure

```text
business-research-ai-agent/
├── .gitignore
├── .python-version
├── agent.py
├── main.py
├── input.md
├── pyproject.toml
├── uv.lock
├── README.md
└── tests/
    └── test_agent.py
```

`output.md` is generated when the agent is run.

## Requirements

* Python 3.11
* [uv](https://docs.astral.sh/uv/)
* Internet access for web search
* Internet access on the first run to download the language model

## Setup

```powershell
uv sync
```

`uv` manages the project environment and dependencies from `pyproject.toml` and `uv.lock`.

## Usage

Write a research request in `input.md`, for example:

```markdown
Research Oxford PV and tell me whether it may be relevant to SEI.
```

Run the agent:

```powershell
uv run python main.py input.md
```

The briefing is written to:

```text
output.md
```

A different output path can be specified:

```powershell
uv run python main.py input.md --output reports/oxford-pv.md
```

## Model and Tools

### Qwen2.5-0.5B-Instruct

The prototype uses **Qwen2.5-0.5B-Instruct**, a small instruction-tuned language model with approximately **0.5 billion parameters**.

The model was selected because it is small enough to run locally on a standard laptop while still supporting instruction-following, summarisation and structured text generation.

The model is downloaded from Hugging Face Hub on the first run and cached locally. **Hugging Face Transformers** is used to load the tokenizer and model weights and perform inference on the local machine.

### DDGS

**DDGS** provides the external web-search capability.

Current company information is retrieved from the web before being passed to the local model as evidence.

```text
DDGS Web Search → Internet
Qwen Inference   → Local machine
```

## Tests

Run the automated tests with:

```powershell
uv run python -m unittest discover -s tests -v
```

## Limitations

This is a learning prototype rather than a production research system.

Search results may be incomplete or unreliable, and the language model may misinterpret evidence or generate unsupported claims. Important commercial or technical claims should therefore be checked against the original sources.
