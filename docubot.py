"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

# Common words that appear in almost every document. They are ignored when
# scoring, so they cannot make an unrelated question look relevant.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "be", "in", "of", "to", "for",
    "and", "or", "on", "with", "by", "as", "at", "from", "it", "this",
    "these", "there", "any", "do", "does", "can", "i", "my", "me", "you",
    "how", "what", "where", "which", "who",
    # Words that refer to the docs themselves, not to a topic in them
    "docs", "documentation", "mention",
}

# Word endings removed by stem(), checked in this order. This lets
# "connect" match "connection" and "generated" match "Generation".
SUFFIXES = ["ing", "ion", "ed", "es", "e", "s"]

# Refusal guardrail: a section must score at least this many points to count
# as evidence. Each matching word is worth 1 point, plus 1 more if it is also
# in the section's heading, so 2 points means "two matching words" or "one
# word that the section is about". If no section reaches it, DocuBot answers
# "I do not know based on these docs."
MIN_SCORE = 2

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Split documents into heading-based sections
        self.chunks = self.split_into_chunks(self.documents)  # List of (filename, section_text)

        # Build a retrieval index over the sections (implemented in Phase 1)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    def split_into_chunks(self, documents):
        """
        Splits each document into sections. A new section starts at every
        markdown heading line (a line beginning with "#"), and the heading
        stays at the top of its section so the section still makes sense
        on its own. A "###" subsection also gets its parent "##" heading,
        so "### users" keeps the context "## Tables".
        Returns a list of tuples: (filename, section_text)
        """
        chunks = []
        for filename, text in documents:
            parent_heading = ""
            current_lines = []
            for line in text.split("\n"):
                if line.startswith("#") and current_lines:
                    chunks.append((filename, "\n".join(current_lines).strip()))
                    current_lines = []
                if line.startswith("## "):
                    parent_heading = line
                if line.startswith("### ") and parent_heading:
                    current_lines.append(parent_heading)
                current_lines.append(line)
            if current_lines:
                chunks.append((filename, "\n".join(current_lines).strip()))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def stem(self, word):
        """
        Removes one common ending so different forms of a word match:
        "connection" -> "connect", "generated" -> "generat",
        "users" -> "user". At least 3 letters are always kept.
        """
        if word.endswith("ss"):
            return word
        for suffix in SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def tokenize(self, text):
        """
        Splits text into lowercase words with surrounding punctuation removed,
        drops stop words, and stems what is left.
        Shared by build_index, score_document, and retrieve so that every
        method cleans words the same way.
        """
        words = []
        for raw_word in text.split():
            word = raw_word.lower().strip(".,:;!?()[]{}<>\"'`*#-")
            if word and word not in STOP_WORDS:
                words.append(self.stem(word))
        return words

    def build_index(self, chunks):
        """
        Phase 1:
        Build a tiny inverted index mapping lowercase words to the chunks
        they appear in. Each chunk is identified by its position in the
        chunks list, since one file now produces many chunks.

        Example structure:
        {
            "token": [3, 4, 12],
            "database": [0, 1, 25]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for chunk_id in range(len(chunks)):
            filename, text = chunks[chunk_id]
            for word in self.tokenize(text):
                if word not in index:
                    index[word] = []
                if chunk_id not in index[word]:
                    index[word].append(chunk_id)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Phase 1:
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score

        tokenize already drops stop words, so only meaningful words count.
        A word that also appears in the section's heading earns 1 bonus
        point, because headings say what a section is about.
        """
        query_words = set(self.tokenize(query))
        doc_words = set(self.tokenize(text))

        heading_lines = []
        for line in text.split("\n"):
            if line.startswith("#"):
                heading_lines.append(line)
        heading_words = set(self.tokenize(" ".join(heading_lines)))

        score = 0
        for word in query_words:
            if word in doc_words:
                score += 1
            if word in heading_words:
                score += 1
        return score

    def retrieve(self, query, top_k=3):
        """
        Phase 1:
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        candidates = []
        for word in self.tokenize(query):
            if word in self.index:
                for chunk_id in self.index[word]:
                    if chunk_id not in candidates:
                        candidates.append(chunk_id)

        scored = []
        for chunk_id in candidates:
            filename, text = self.chunks[chunk_id]
            score = self.score_document(query, text)
            if score >= MIN_SCORE:
                scored.append((score, filename, text))

        scored.sort(reverse=True)

        results = []
        for score, filename, text in scored:
            results.append((filename, text))
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
