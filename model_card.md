# DocuBot Model Card
<!-- 
This model card is a short reflection on your DocuBot system. Fill it out after you have implemented retrieval and experimented with all three modes:

1. Naive LLM over full docs  
2. Retrieval only  
3. RAG (retrieval plus LLM)

Use clear, honest descriptions. It is fine if your system is imperfect. -->

---

## 1. System Overview

**What is DocuBot trying to do?**  
Describe the overall goal in 2 to 3 sentences.

> DocuBot answers developer questions about a small project (auth, API, database, setup) by reading the markdown files in `docs/`. It exists to compare three ways of answering the same question — dumping the whole corpus into an LLM, doing keyword retrieval with no LLM, and combining retrieval with an LLM (RAG) — so we can see where each approach is strong, weak, or actively misleading.

**What inputs does DocuBot take?**  
For example: user question, docs in folder, environment variables.

> A developer's natural-language question, the `.md`/`.txt` files in `docs_folder` (default `docs/`), and, when LLM modes are used, a `GEMINI_API_KEY` environment variable (loaded via `.env`) used to construct a `GeminiClient`.

**What outputs does DocuBot produce?**

> Depending on mode: (1) naive mode returns free-form prose generated from the entire corpus with no citations; (2) retrieval-only mode returns the raw top-`k` matching text chunks labeled with their source filename, with no synthesis; (3) RAG mode returns a synthesized natural-language answer that cites which file(s) it relied on, or the literal string `"I do not know based on the docs I have."` when it judges the retrieved snippets insufficient.

---

## 2. Retrieval Design

**How does your retrieval system work?**  
Describe your choices for indexing and scoring.

- How do you turn documents into an index?
- How do you score relevance for a query?
- How do you choose top snippets?

> Each markdown file is first split into section-level chunks at every `#`/`##` heading (`split_into_sections`, `max_split_level=2`); deeper `###` headings stay attached to their parent section. Every chunk is then indexed into a simple inverted index (`build_index`) mapping lowercase, punctuation-stripped tokens to the chunks they appear in, alongside the corpus's average chunk length for BM25's length normalization. `score_document` computes a standard BM25 score per chunk (`k1=1.5`, `b=0.75`), and `retrieve` sorts all chunks by score, drops anything scoring at or below `min_score=2.3` (a guardrail against single stray-word matches on off-topic queries), and returns the top `k` (default 3) as `(filename, text)` pairs.

**What tradeoffs did you make?**  
For example: speed vs precision, simplicity vs accuracy.

> Section-level chunking is simple and cheap (no embeddings, no vector store, pure Python + `math.log`) and keeps each chunk topically coherent, but it means chunk sizes are uneven — a single "## Tables" section in `DATABASE.md` bundles three separate table schemas into one long chunk. BM25's length-normalization term (`b=0.75`) then systematically penalizes that long, information-dense chunk relative to short chunks that happen to share a query word, which caused real misses in testing (see Section 5). We also traded recall for precision with `min_score=2.3`: it successfully suppresses zero/near-zero matches, but as shown below it still lets through chunks that share a common word with the query (e.g. "database", "environment") without being topically relevant, so it is a partial guardrail, not a semantic relevance filter.

---

## 3. Use of the LLM (Gemini)

**When does DocuBot call the LLM and when does it not?**  
Briefly describe how each mode behaves.

- Naive LLM mode:
- Retrieval only mode:
- RAG mode:

> **Naive LLM mode** always calls Gemini, passing the *entire* concatenated corpus (`full_corpus_text()`) plus the raw question, with no retrieval step and no refusal instruction beyond the model's own judgment. **Retrieval-only mode** never calls the LLM — it runs BM25 retrieval and returns the raw matched chunks (or `"I do not know based on these docs."` if nothing clears `min_score`). **RAG mode** always retrieves first (same BM25 call as retrieval-only, `top_k=3`); if nothing is retrieved it short-circuits to the same refusal string without ever calling the LLM, and if snippets are found, it calls Gemini with only those snippets as context.

**What instructions do you give the LLM to keep it grounded?**  
Summarize the rules from your prompt. For example: only use snippets, say "I do not know" when needed, cite files.

> RAG mode's prompt (`answer_from_snippets` in `llm_client.py`) instructs the model to: (1) answer using only the information in the provided snippets; (2) refuse by replying exactly `"I do not know based on the docs I have."` if the snippets are not enough to answer confidently; (3) not invent functions, endpoints, or config values not present in the snippets; (4) briefly mention which file(s) it relied on when it does answer. Naive mode's prompt has none of these rules — it is just "you are a documentation assistant, here are the docs, answer the question," which is why its groundedness depends entirely on the model's own honesty rather than an enforced contract.

---

## 4. Experiments and Comparisons

Run the **same set of queries** in all three modes. Fill in the table with short notes.

You can reuse or adapt the queries from `dataset.py`.

| Query | Naive LLM: helpful or harmful? | Retrieval only: helpful or harmful? | RAG: helpful or harmful? | Notes |
|------|---------------------------------|--------------------------------------|---------------------------|-------|
| Where is the auth token generated? | Helpful — correct, concise | Helpful — accurate but 3 raw chunks (some redundant) | Helpful — concise, cites AUTH.md | All three converge; easy query with an exact keyword match. |
| What environment variables are required for authentication? | Looks helpful, but papers over a contradiction — see Section 5, Failure Case 1 | Partially helpful — retrieved AUTH.md's *intro* paragraph (which happens to contain the phrase "environment variables") instead of AUTH.md's actual "Environment Variables" section | Helpful but incomplete — answers from SETUP.md only, never surfaces AUTH.md's own "Environment Variables" section even though it was retrieved | BM25 top-3 missed the single most relevant chunk in the whole corpus. |
| How do I connect to the database? | Helpful — correct, matches DATABASE.md content | Harmful/misleading — all 3 chunks are from SETUP.md; DATABASE.md was never retrieved (evaluation.py logs this as a miss) | Accidentally helpful — correct answer, but only because SETUP.md duplicates the `DATABASE_URL` info; cites "SETUP.md" as if that were the authoritative source | See Section 5, Failure Case 2 — a real retrieval bug masked by content overlap between files. |
| Which endpoint lists all users? | Helpful — correct (`GET /api/users`, admin only) | Technically "hit" by filename (API_REFERENCE.md retrieved) but the actual endpoint chunk was not — the 3 chunks shown (API_REFERENCE.md intro, DATABASE.md overview, AUTH.md overview) don't contain the answer | Appropriately cautious — replies "I do not know based on the docs I have," correctly refusing rather than guessing, but this is a retrieval-caused false negative | Best example of RAG's guardrail correctly refusing on top of a retrieval miss, rather than hallucinating an endpoint. |
| What does the /api/projects/<project_id> route return? | Helpful — correct, verbose | Helpful — right chunk retrieved, but reader must scan past an unrelated SETUP.md troubleshooting chunk to find it | Most helpful — clean bulleted field list, cites API_REFERENCE.md, no noise | Clearest case of RAG's synthesis adding real value over raw retrieval. |
| Is there any mention of payment processing in these docs? | Helpful — correctly says no such mention exists | Harmful/misleading if taken at face value — 3 unrelated chunks pass `min_score=2.3` (SETUP.md, DATABASE.md, SETUP.md) with no indication they're irrelevant | Helpful — correctly replies "I do not know based on the docs I have" despite being fed irrelevant snippets | Shows the retrieval guardrail alone is leaky; the LLM's own instruction-following is what actually catches this case in RAG mode. |
| How does a client refresh an access token? | Helpful — correct, detailed | Helpful — right AUTH.md and API_REFERENCE.md chunks retrieved | Helpful — concise, cites both files correctly | Clean convergence case. |
| Which fields are stored in the users table? | Helpful — correct, full table reproduced | Harmful/misleading — DATABASE.md's "## Tables" section (the one chunk with the actual schema) was never retrieved; instead got DATABASE.md's Overview + intro and an irrelevant AUTH.md chunk | Appropriately cautious but unhelpful — replies "I do not know," a retrieval-caused false refusal identical in shape to the "users" endpoint case | Same root cause as row 2: a long, information-dense chunk lost to BM25 length normalization against shorter, off-target chunks. |

**What patterns did you notice?**  

- When does naive LLM look impressive but untrustworthy? Whenever the docs contain overlapping or slightly contradictory guidance — naive mode confidently produces one clean, uncited narrative and silently resolves the contradiction itself, so a reader has no way to tell that the underlying document actually says two different things (see "environment variables" row and Failure Case 1). More generally, naive mode's apparent reliability here is largely an artifact of a tiny 4-file corpus that fits entirely in context; it has no citation mechanism and would not scale to a larger doc set without losing exactly this property.
- When is retrieval only clearly better? Never in terms of end-user usability — even on queries where its raw chunks are 100% correct (e.g. "Where is the auth token generated?"), the burden is entirely on the reader to skim several chunks (including duplicated or irrelevant ones) and manually extract the answer. Its main value is as a *diagnostic* tool: since it shows exactly which raw chunks were selected, it is the only mode that makes retrieval bugs (missed chunks, irrelevant chunks passing the threshold) visible instead of hidden behind a fluent LLM sentence.
- When is RAG clearly better than both? When retrieval succeeds and the answer needs synthesis across a couple of well-matched chunks — the `/api/projects/<project_id>` and "refresh token" queries show RAG at its best: as accurate as naive, as evidence-backed as retrieval-only, cited, and far more concise than either. RAG is also the only mode with an enforced, explicit refusal contract, which is why it (not naive, not retrieval-only) is the only mode to correctly flag the deliberately out-of-scope "payment processing" query as unanswerable rather than just returning noise or a guess.

---

## 5. Failure Cases and Guardrails

**Describe at least two concrete failure cases you observed.**  
For each one, say:

- What was the question?  
- What did the system do?  
- What should have happened instead?

> **Failure case 1 — naive mode smooths over a real contradiction in the docs.** Question: "What environment variables are required for authentication?" `AUTH.md` itself is internally inconsistent: its "Environment Variables" section header lists `AUTH_SECRET_KEY` and `TOKEN_LIFETIME_SECONDS` and states "Both variables must be configured before starting the server," yet in the very same paragraph says `TOKEN_LIFETIME_SECONDS` "Defaults to 3600 seconds if not set." Naive mode's answer picked one side of the contradiction ("`TOKEN_LIFETIME_SECONDS` (Optional)... defaults to 3600 seconds") and presented it as a single, confident, uncited fact. What should have happened: the system should have surfaced the discrepancy ("AUTH.md says both are required, but also says one has a default — please confirm with the team") or at minimum cited the exact source line so a developer could notice the tension themselves. Confident, uncited synthesis is the actual risk of naive mode, not just fabrication of facts that aren't in the docs at all.

> **Failure case 2 — a retrieval miss causes RAG to substitute the wrong authoritative source, undetected.** Question: "How do I connect to the database?" BM25 retrieved 3 chunks, all from `SETUP.md`, and none from `DATABASE.md` — the file that actually owns this topic (connection strings, `db.py`, table schemas, DB-specific failure modes). `evaluation.py` correctly flags this as a miss (`Hit: False`). Because `SETUP.md` happens to repeat the `DATABASE_URL` variable, RAG mode still produced a correct-looking answer and cited "SETUP.md" as if it were the right source. What should have happened instead: `DATABASE.md`'s "## Connection Configuration" section should have ranked in the top 3 (it is a near-verbatim match for "connect to the database"), and if it hadn't, RAG should have had a lower-confidence signal available (e.g. a similarity floor tied to the specific expected file) rather than silently answering from an adjacent file that only partially overlaps — for a slightly different question (e.g. "what happens on a SQLite locking error?") the same retrieval bug would have produced an unwarranted "I do not know," since that content lives only in `DATABASE.md`.

**When should DocuBot say “I do not know based on the docs I have”?**  
Give at least two specific situations.

> (1) When no chunk clears the BM25 relevance threshold at all — e.g. a query about a topic genuinely absent from the docs (payment processing, rate limiting, OAuth) should hit the `min_score` floor and return the refusal directly from `retrieve()`/`answer_rag()` without ever calling the LLM. (2) When chunks *are* retrieved but don't actually contain the specific fact asked for — e.g. "Which endpoint lists all users?" and "Which fields are stored in the users table?" both retrieved plausible-looking but off-target chunks; RAG's prompt-level instruction ("if the snippets do not provide enough evidence, refuse to guess") is what correctly triggers refusal here, since the code path did not short-circuit (something *was* retrieved) but no single retrieved chunk answers the question.

**What guardrails did you implement?**  
Examples: refusal rules, thresholds, limits on snippets, safe defaults.

> Two independent layers, and testing showed both are necessary because each has gaps the other covers: (1) A retrieval-side `min_score=2.3` threshold in `DocuBot.retrieve()` that drops near-zero BM25 matches and returns a fixed refusal string when nothing clears it — but this threshold is calibrated for *any* shared keyword, not topical relevance, so it still let three off-topic chunks through for the "payment processing" query. (2) An LLM-side prompt contract in `answer_from_snippets()` requiring Gemini to use only the provided snippets, invent nothing, and reply with the exact string `"I do not know based on the docs I have."` when the snippets are insufficient — this is what actually caught the payment-processing case and the two retrieval-miss cases above, i.e. it is currently doing more of the safety work than the numeric threshold.

---

## 6. Limitations and Future Improvements

**Current limitations**  
List at least three limitations of your DocuBot system.

1. BM25 length normalization systematically disadvantages long, information-dense chunks (e.g. `DATABASE.md`'s combined "## Tables" section) against short chunks that share a query word by coincidence, causing the exact chunk with the answer to fall out of the top-3 in at least two of the eight sample queries.
2. The evaluation harness in `evaluation.py` only checks file-level hits (`fname in retrieved_files`), not whether the *specific relevant section* was retrieved — it reported a 0.75 hit rate, but at least two of those six "hits" retrieved the right file with the wrong chunk, so true chunk-level retrieval accuracy is meaningfully lower than the headline number suggests.
3. The `min_score=2.3` guardrail filters near-zero BM25 scores but not "shares one common word, wrong topic" chunks, so retrieval-only mode (which has no LLM to catch this) can present irrelevant snippets for out-of-scope questions as if they were relevant.
4. Naive mode's "groundedness" only holds because the demo corpus (4 short files) fits entirely in the LLM's context window; nothing in that mode's design (no chunking, no citations, no retrieval) would survive being pointed at a real, multi-hundred-file documentation set.

**Future improvements**  
List two or three changes that would most improve reliability or usefulness.

1. Weight or exempt heading-matched terms (BM25-F style, boosting matches in the chunk's own heading) so a section literally titled "Environment Variables" or "Tables" outranks an unrelated chunk that happens to mention the same words in passing.
2. Extend `evaluation.py` to check for expected *substrings* within the retrieved chunk text (not just filenames), so retrieval regressions like the two chunk-level misses above would show up as failures instead of being masked as hits.
3. Add a second, stricter threshold (or an LLM-based relevance check) between "shares a keyword" and "is topically relevant," so retrieval-only mode gets the same off-topic protection that RAG mode currently gets almost entirely from Gemini's own prompt-following rather than from the retrieval layer itself.

---

## 7. Responsible Use

**Where could this system cause real world harm if used carelessly?**  
Think about wrong answers, missing information, or over trusting the LLM.

> The clearest real-world risk demonstrated here is silent source substitution: RAG mode answered "How do I connect to the database?" confidently and correctly, but cited `SETUP.md` when the authoritative source was actually `DATABASE.md`, which was never retrieved at all. In a real codebase, the overlap that saved this particular answer (SETUP.md happening to repeat `DATABASE_URL`) will not always exist — a developer trusting the citation as "the place to look for more detail" would be sent to the wrong file, or would get a correct-sounding but incomplete answer on a question where the two files diverge (e.g. DB-specific failure modes that only live in `DATABASE.md`). Naive mode's risk is different: it can quietly resolve a genuine contradiction in the source docs (Failure Case 1) into one confident, uncited sentence, hiding the fact that the documentation itself needs to be fixed. Both risks compound with scale — on a 4-file demo corpus the failure rate is visible in testing; on a real multi-hundred-file doc set, naive mode stops being able to see the whole corpus at all, and RAG's retrieval-miss rate would need to be measured, not assumed away.

**What instructions would you give real developers who want to use DocuBot safely?**  
Write 2 to 4 short bullet points.

- Treat RAG's cited filename as "what was retrieved," not "the ground truth" — when the answer matters, open the cited file yourself and confirm the specific section actually supports the claim, especially for questions that span more than one doc.
- Don't rely on naive mode for anything beyond quick exploration of a small corpus; it has no citations and no refusal contract, so its confidence level tells you nothing about whether the underlying docs actually agree with each other.
- Watch retrieval-only mode's raw output for chunks that don't obviously relate to the question — under the current `min_score` threshold, a shared keyword is enough to pass the bar, so an irrelevant chunk showing up is not itself a bug, but treating it as relevant would be.
- If DocuBot (in any mode) returns `"I do not know based on the docs I have,"` treat that as "the retrieval step likely missed the right section," not "the answer doesn't exist in the docs" — as shown above, this refusal fired on two questions the docs actually do answer.

---
