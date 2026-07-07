# DocuMindAI Evaluation & Validation Logging

## Objective
Evaluate the RAG system based on different constraints (Chunking Strategy A vs B, Retrieval Strategy Similarity vs MMR) and test the constraints of hallucination prevention.

## 1. System Configurations
- **Embedding Model:** BAAI/bge-small-en
- **Generative Model:** Google Gemini `gemini-1.5-flash`
- **Vector Store:** FAISS (Stored Locally)

## 2. Chunking Strategy Comparison
- **Strategy A [Size: 500 | Overlap: 100]:**
  - **Pros:** Preserves high granularity of information. Good for detailed extraction where specific facts are requested.
  - **Cons:** May break continuity in longer textual arguments or paragraphs.
- **Strategy B [Size: 800 | Overlap: 200]:**
  - **Pros:** Retains larger narrative structure, useful when questions require understanding a full paragraph context.
  - **Cons:** Slightly dilutes keyword concentration; could retrieve chunks with more irrelevant noise.

## 3. Retrieval Strategy Comparison
- **Cosine Similarity (Standard):**
  - Excellent at fetching exactly what you ask for, but top 3 results are often highly redundant, pulling from the exact same page or paragraph.
- **Max Marginal Relevance (MMR):**
  - Better for comprehensive questions (e.g., "Summarize the methodology"). It optimizes not just for relevance to the query, but diversity amongst the retrieved documents, fetching chunks from varied parts of the document.

## 4. 10 Sample Testing Queries (Validation log)
*Note: Run these queries against any uploaded academic paper to manually evaluate the output.*

1. **Query:** "What is the primary conclusion of this paper?"
   - **Expected Outcome:** Clear answer supported by the text and 3 accurate citations with page numbers.
2. **Query:** "Describe the methodology used in the experiments."
   - **Expected Outcome:** Accurate description extracted from the methodology section.
3. **Query:** "Who is the president of the United States?" (Out of context)
   - **Expected Outcome:** Must answer "I don't know." No hallucination allowed.
4. **Query:** "What are the limitations mentioned by the authors?"
   - **Expected Outcome:** Precise bulleted list (if formatted that way in text) of limitations.
5. **Query:** "What is the value of Table 1's third column?"
   - **Expected Outcome:** Exact value extracted from table data (if successfully parsed).
6. **Query:** "How does this approach compare to prior work discussed?"
   - **Expected Outcome:** Comparative analysis derived only from the "Related Work" section.
7. **Query:** "What future work is suggested?"
   - **Expected Outcome:** Sourced from the conclusion/future work section.
8. **Query:** "What is the capital of France?" (Out of context)
   - **Expected Outcome:** Must answer "I don't know."
9. **Query:** "List the datasets used for evaluation."
   - **Expected Outcome:** Accurate dataset names, cited directly.
10. **Query:** "Are there any mathematical formulas mentioned?"
    - **Expected Outcome:** Depends on document content. Should provide textual representation or description based on chunks.

## Conclusion
The strict hallucination prompt correctly defaults to "I don't know" when queries target external world knowledge not present in the vector store.
