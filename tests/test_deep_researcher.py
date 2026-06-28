"""Tests for deep_researcher.py — graph structure and basic module sanity."""
import inspect

from open_deep_research.deep_researcher import (
    clarify_with_user,
    compile_report,
    deep_researcher,
    export_report,
    final_report_generation,
    final_review,
    generate_report_outline,
    generate_research_plan,
    optional_plan_review,
    parse_document,
    researcher,
    researcher_tools,
    rewrite_sections,
    supervisor,
    supervisor_tools,
    verify_citations,
    write_research_brief,
    write_sections_parallel,
)

EXPECTED_NODES = {
    "__start__",
    "clarify_with_user",
    "parse_document",
    "write_research_brief",
    "classify_research_request",
    "generate_research_plan",
    "optional_plan_review",
    "research_supervisor",
    "verify_citations",
    "generate_report_outline",
    "write_sections_parallel",
    "compile_report",
    "export_report",
    "final_review",
    "rewrite_sections",
    "final_report_generation",
    "__end__",
}

TESTABLE_IMPORTS = [
    clarify_with_user, compile_report, export_report,
    final_report_generation, final_review, generate_report_outline,
    generate_research_plan, optional_plan_review, parse_document,
    researcher, researcher_tools, rewrite_sections, supervisor,
    supervisor_tools, verify_citations, write_research_brief,
    write_sections_parallel,
]


class TestDeepResearcherGraph:
    def test_graph_is_compiled(self):
        assert hasattr(deep_researcher, "get_graph")

    def test_graph_has_expected_nodes(self):
        actual = set(deep_researcher.get_graph().nodes.keys())
        assert actual == EXPECTED_NODES

    def test_graph_start_to_clarify_edge(self):
        edges = deep_researcher.get_graph().edges
        assert any(e.source == "__start__" and e.target == "clarify_with_user" for e in edges)

    def test_graph_export_to_end_edge(self):
        edges = deep_researcher.get_graph().edges
        assert any(e.source == "export_report" and e.target == "__end__" for e in edges)

    def test_export_report_reaches_end(self):
        edges = deep_researcher.get_graph().edges
        assert any(e.source == "export_report" and e.target == "__end__" for e in edges)

    def test_final_review_branches_to_rewrite_or_export(self):
        edges = deep_researcher.get_graph().edges
        final_review_edges = [(e.source, e.target) for e in edges if e.source == "final_review"]
        targets = {t for _, t in final_review_edges}
        assert "rewrite_sections" in targets or "export_report" in targets


class TestDeepResearcherFunctions:
    def test_all_functions_are_async(self):
        for fn in TESTABLE_IMPORTS:
            assert inspect.iscoroutinefunction(fn), f"{fn.__name__} is not async"
