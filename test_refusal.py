"""
Tests for DocuBot's refusal guardrail. No API key needed.

Run with:
    python test_refusal.py
"""

from docubot import DocuBot

REFUSAL = "I do not know based on these docs."

# Questions the docs have nothing to say about. Retrieval should find no
# evidence, so DocuBot should refuse.
UNRELATED_QUERIES = [
    "What is the capital of France?",
    "How do I bake chocolate chip cookies?",
    "What is the weather today?",
    "Is there any mention of payment processing in these docs?",
    "How do I configure Kubernetes autoscaling?",
]

# Questions the docs do answer. The guardrail must not refuse these.
RELATED_QUERIES = [
    "Where is the auth token generated?",
    "How do I connect to the database?",
    "Which endpoint lists all users?",
    "How does a client refresh an access token?",
    "Which fields are stored in the users table?",
    "What does the /api/projects/<project_id> route return?",
]


def test_refuses_unrelated_questions():
    bot = DocuBot()
    for query in UNRELATED_QUERIES:
        assert bot.retrieve(query) == [], f"retrieved evidence for: {query}"
        assert bot.answer_retrieval_only(query) == REFUSAL, f"did not refuse: {query}"


def test_answers_related_questions():
    bot = DocuBot()
    for query in RELATED_QUERIES:
        assert bot.retrieve(query) != [], f"wrongly refused: {query}"


if __name__ == "__main__":
    test_refuses_unrelated_questions()
    test_answers_related_questions()
    print("All refusal tests passed.")
