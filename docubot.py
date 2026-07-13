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
import math  # needed for the logarithm in the BM25 IDF calculation

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

        # Split each document into sections; these chunks are the unit BM25 scores against
        self.chunks = self.build_chunks(self.documents)  # List of (filename, section_text)

        # Build a retrieval index over chunks (implemented in Phase 1)
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

    # -----------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------

    def heading_level(self, line):
        """Return the number of leading '#' characters in a heading line, or None if not a heading."""
        stripped = line.strip()
        if not stripped.startswith("#"):
            return None
        return len(stripped) - len(stripped.lstrip("#"))

    def split_into_sections(self, text, max_split_level=2):
        """
        Split markdown text into sections, starting a new section at each
        heading whose level is <= max_split_level (default: '#' and '##').
        Deeper headings (e.g. '###') stay attached to their parent section.
        """
        lines = text.split("\n")
        sections = []
        current_lines = []

        for line in lines:
            level = self.heading_level(line)
            # start a new section only at top-level headings, and only if we've
            # already collected content for a previous section
            if level is not None and level <= max_split_level and current_lines:
                sections.append("\n".join(current_lines).strip())
                current_lines = [line]  # begin the new section with its heading line
            else:
                current_lines.append(line)

        if current_lines:
            sections.append("\n".join(current_lines).strip())  # append the final section

        return [s for s in sections if s]  # drop any empty sections

    def build_chunks(self, documents):
        """
        Split every document into section-level chunks.
        Returns a list of (filename, section_text) tuples, the unit BM25 scores against.
        """
        chunks = []
        for filename, text in documents:
            for section_text in self.split_into_sections(text):
                chunks.append((filename, section_text))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        (Phase 1):
        Build a tiny inverted index mapping lowercase words to the chunks
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}  # word -> list with one entry per chunk containing it (len() = BM25 document frequency)
        total_length = 0  # running total of word counts, used to compute the corpus average length

        for filename, text in documents:
            words = text.lower().split()  # lowercase the doc and split on whitespace into raw tokens
            cleaned_words = []  # will hold the punctuation-stripped tokens for this document

            for word in set(words):
                word = word.strip(".,!?;:()[]\"'")  # strip surrounding punctuation from the token
                if not word:  # skip tokens that were pure punctuation (now empty)
                    continue
                cleaned_words.append(word)  # keep the cleaned token for length counting

                index.setdefault(word, []).append(filename)

            total_length += len(cleaned_words)  # add this document's word count to the running total

        # average chunk length across the corpus, used by BM25's length-normalization term
        self.avg_doc_length = total_length / len(documents) if documents else 0

        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text, k1=1.5, b=0.75):
        """
        Return a BM25 relevance score for how well the text matches the query.

        k1 controls how quickly extra occurrences of a term stop adding value.
        b controls how much document length is penalized (0 = no penalty, 1 = full).
        """
        # lowercase and strip punctuation from the query the same way build_index cleaned documents
        query_words = [w.strip(".,!?;:()[]\"'") for w in query.lower().split()]
        query_words = [w for w in query_words if w]  # drop any tokens that became empty

        # do the same cleanup to the candidate document's text
        words = [w.strip(".,!?;:()[]\"'") for w in text.lower().split()]
        words = [w for w in words if w]

        doc_length = len(words) or 1  # this chunk's length in words; avoid dividing by zero below
        N = len(self.chunks)  # total number of chunks in the corpus
        avgdl = self.avg_doc_length or 1  # corpus average chunk length; avoid dividing by zero

        score = 0.0
        for term in query_words:
            term_freq = words.count(term)  # how many times this query term appears in the document
            if term_freq == 0:
                continue  # term doesn't appear here, so it contributes nothing to the score

            df = len(self.index.get(term, []))  # number of documents that contain this term at all
            # IDF: rarer terms across the corpus count for more than common ones
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            numerator = term_freq * (k1 + 1)  # raw term frequency, boosted by (k1 + 1)
            # denominator saturates term frequency and normalizes for document length vs. the average
            denominator = term_freq + k1 * (1 - b + b * (doc_length / avgdl))

            score += idf * (numerator / denominator)  # add this term's contribution to the total score

        return score

    def retrieve(self, query, top_k=3, min_score=2.3):
        """
        Use the index and scoring function to select top_k relevant section chunks.

        min_score: chunks scoring at or below this are treated as noise (e.g. a single
        stray word match) rather than genuine relevance, and are excluded. Tune this
        against your own docs/queries — raw BM25 scores aren't bounded or comparable
        across corpora.

        Return a list of (filename, text) sorted by score descending.
        """
        # score every chunk in the corpus against the query
        scored = [(self.score_document(query, text), filename, text) for filename, text in self.chunks]

        scored.sort(key=lambda item: item[0], reverse=True)  # highest score first
        # print("max score:", scored[0][0] if scored else 0)  # debug: print the highest score for this query
        
        # drop chunks that don't clear the relevance bar and strip the score from the tuple
        results = [(filename, text) for score, filename, text in scored if score > min_score]
        
        return results[:top_k]  # return only the top_k most relevant documents

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
