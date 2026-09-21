import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import BusinessResearchAgent, SearchResult


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
        self.assertIn("Sources:", output)
        self.assertIn("https://example.com/company", output)


if __name__ == "__main__":
    unittest.main()
