import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import BusinessResearchAgent, LocalQwenModel, SearchResult


class FakeSearchTool:
    def __init__(self):
        self.last_query = None

    def search(self, query, max_results=5):
        self.last_query = query
        return [
            SearchResult(
                title="Example source",
                url="https://example.com/company",
                snippet="Example Company develops solar technology and announced a new project.",
            )
        ]


class FakeModel:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt
        return "1. Company / Topic\nExample Company\n\n5. Relevance to SEI\nPotential solar relevance."


class BusinessResearchAgentTests(unittest.TestCase):
    def test_identify_target(self):
        request = "Research Oxford PV and tell me whether it may be relevant to SEI."
        self.assertEqual(BusinessResearchAgent.identify_target(request), "Oxford PV")

    def test_choose_sei_context(self):
        name, context = BusinessResearchAgent.choose_business_context(
            "Research Oxford PV and assess relevance to SEI."
        )
        self.assertEqual(name, "SEI")
        self.assertIn("solar PV", context)

    def test_choose_oxvalue_context(self):
        name, context = BusinessResearchAgent.choose_business_context(
            "Research Example AI and assess relevance to OxValue.ai."
        )
        self.assertEqual(name, "OxValue.ai")
        self.assertIn("business valuation", context)

    def test_complete_pipeline_with_test_doubles(self):
        search = FakeSearchTool()
        model = FakeModel()
        agent = BusinessResearchAgent(search, model)

        output = agent.run(
            "Research Example Company and tell me whether it may be relevant to SEI."
        )

        self.assertIn("Example Company", search.last_query)
        self.assertIn("External evidence", model.last_prompt)
        self.assertIn("solar technology", model.last_prompt)
        self.assertIn("## Sources", output)
        self.assertIn("https://example.com/company", output)


class LocalQwenModelTests(unittest.TestCase):
    @patch("agent.urlopen")
    def test_uses_qwen35_q4_k_m_through_local_ollama(self, mock_urlopen):
        response = io.BytesIO(
            json.dumps({"message": {"content": "Generated briefing"}}).encode()
        )
        mock_urlopen.return_value.__enter__.return_value = response

        model = LocalQwenModel()

        self.assertEqual(model.generate("Research request"), "Generated briefing")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3.5:4b-q4_K_M")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])


if __name__ == "__main__":
    unittest.main()
