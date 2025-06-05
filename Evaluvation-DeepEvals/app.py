from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import hallucination
from deepeval.metrics import AnswerRelevancyMetric


#when it is wrong - what need to focus(What need to fix)
#ContextualRelevancyMetric


# 📄 Score Calculation Logic
# This method calculates the ratio of "yes" verdicts to the total number of verdicts. If strict_mode=True and the score is below a given threshold, the final score is set to 0.

#Unlike Contextual Precision or Recall (which look at relevance of specific chunks or matching expected output
# Contextual Relevancy assesses the overall quality of the retrieved context — so even a few weak or unrelated chunks can significantly reduce the score.


#"""
 #   | Possible Cause                                     | Explanation                                                                                        |
#| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
#| ❌ **Irrelevant context chunks**                    | Even 1–2 unrelated pieces in `retrieval_context` can pull down the LLM score.                      |
#| 🤖 **Retriever fetches noisy data**                | If the retriever brings in loosely related or generic documents, LLM deems the whole context weak. |
#| 🎯 **No strong alignment between input & context** | LLM might not find the context directly answering or supporting the user query.                    |
#| 📉 **Too much context**                            | Large or unfocused context leads to dilution in relevance, affecting score.                        |
#| 📊 **Mismatch between query and domain**           | If the `input` is specific but the context is generic or domain-shifted, relevance score drops.    |
#"""


# """
# ✅ How to Improve Contextual Relevancy Score:
# Tighten your retriever results
# → Use fewer, more focused documents (e.g., top 3 chunks instead of top 10).

# Apply re-ranking before scoring
# → Tools like Cohere Rerank, or use LLM-based re-ranking to prioritize the most relevant chunks.

# Use semantic filters or metadata filtering
# → Filter documents by tags, domain, or topic before retrieval.

# Debug with DeepEval's explanation output
# → Look at the reason returned with the score — it helps understand why the score is low.

# Compare with Contextual Precision or Recall
# → If those are high and relevancy is low, your ranking may be okay but the context is still too broad or noisy.
# """

# I have a couple of doubt's
# how this retriver is working wheather it has any generated based on the ranking manner or not!


# 🧠 #HallucinationMetric
# 📄 Score Calculation Logic
# The HallucinationMetric checks whether the model’s actual_output is factually grounded in the given context. If the response contains information not present or contradicted by the context, it's flagged as a hallucination.


# | ❌ **Possible Cause**                     | 📌 **Explanation**                                                                                      |
# | ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
# | ❗ **Incorrect or invented facts**        | The model states something **not mentioned** or **contradicted** by the context.                        |
# | 📖 **Overgeneralization or assumptions** | Model draws conclusions not explicitly supported by the context.                                        |
# | ⚠️ **Outdated or mismatched context**    | The context is incomplete or doesn't align with the latest facts, confusing the LLM.                    |
# | 📉 **Ambiguous or vague context**        | If the context is unclear or open to interpretation, the model might generate loosely grounded answers. |
# | 🔎 **Too little context**                | Insufficient detail in context causes the model to "fill in the blanks" with fabricated facts.          |


# Great question! Let's walk through this block of code with a **simple example** to make it clear. This logic calculates **weighted precision at K** — a way to score how relevant your retrieved contexts are, especially rewarding **earlier correct items**.

# ---

# ### ✅ Code Block Recap

# ```python
# for k, is_relevant in enumerate(node_verdicts, start=1):
#     if is_relevant:
#         relevant_nodes_count += 1
#         precision_at_k = relevant_nodes_count / k
#         sum_weighted_precision_at_k += precision_at_k
# ```

# ---

# ## 🧠 Concept

# This loop rewards **early correct answers** more than later ones.

# ---

# ## 📘 Example 1

# Imagine this is the verdict list:

# ```python
# node_verdicts = [1, 0, 1, 1]  # Means: [yes, no, yes, yes]
# ```

# So, we are iterating through:

# | k (position) | is\_relevant (1 or 0) | relevant\_nodes\_count | Precision\@k   | Sum of weighted |
# | ------------ | --------------------- | ---------------------- | -------------- | --------------- |
# | 1            | ✅ 1                   | 1                      | 1 / 1 = 1.00   | +1.00           |
# | 2            | ❌ 0                   | 1                      | - (skipped)    |                 |
# | 3            | ✅ 1                   | 2                      | 2 / 3 ≈ 0.6667 | +0.6667         |
# | 4            | ✅ 1                   | 3                      | 3 / 4 = 0.75   | +0.75           |

# Now total:

# ```python
# sum_weighted_precision_at_k = 1.00 + 0.6667 + 0.75 = 2.4167
# relevant_nodes_count = 3
# final_score = 2.4167 / 3 ≈ 0.8056
# ```

# ---

# ## 📘 Example 2 (all relevant, perfectly ranked)

# ```python
# node_verdicts = [1, 1, 1, 1]  # all yes
# ```

# | k | is\_relevant | relevant\_nodes\_count | Precision\@k | Sum   |
# | - | ------------ | ---------------------- | ------------ | ----- |
# | 1 | ✅ 1          | 1                      | 1.00         | +1.00 |
# | 2 | ✅ 1          | 2                      | 2 / 2 = 1.0  | +1.00 |
# | 3 | ✅ 1          | 3                      | 3 / 3 = 1.0  | +1.00 |
# | 4 | ✅ 1          | 4                      | 4 / 4 = 1.0  | +1.00 |

# Total score = 4.0 / 4 = **1.0 (Perfect)**

# ---

# ## 📘 Example 3 (relevant but late)

# ```python
# node_verdicts = [0, 0, 1, 1]
# ```

# | k | is\_relevant | relevant\_nodes\_count | Precision\@k | Sum   |
# | - | ------------ | ---------------------- | ------------ | ----- |
# | 1 | ❌ 0          | 0                      | -            |       |
# | 2 | ❌ 0          | 0                      | -            |       |
# | 3 | ✅ 1          | 1                      | 1 / 3 ≈ 0.33 | +0.33 |
# | 4 | ✅ 1          | 2                      | 2 / 4 = 0.5  | +0.5  |

# Final score = (0.33 + 0.5) / 2 = **0.4167**

# Notice how **late relevant answers hurt the score**.

# ---

# ## 🧠 Summary

# | Scenario                      | Final Score (approx) |
# | ----------------------------- | -------------------- |
# | Relevant early (Example 1)    | 0.81                 |
# | All relevant, ranked early    | 1.00 (Perfect)       |
# | Relevant but late (Example 3) | 0.42 (Lower)         |

# ---

# If you'd like, I can also visualize this with a small chart or write a function to simulate this scoring. Want that?





# | Metric           | What It Tells You                                                   | Focus Area in RAG Pipeline                                                           | Why It Matters                                                        | Common Fixes                                                                                                                            |
# | ---------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
# | 🎯 **Precision** | Are retrieved chunks actually useful for generating the answer?     | 🔍 **Retriever Ranking & Filtering**<br>📄 **Context Chunk Quality**                 | High precision means most retrieved info is relevant and useful.      | - Improve chunking strategy (split meaningfully)<br>- Tune retriever (BM25, hybrid, dense)<br>- Rerank retrieved docs<br>- Filter noise |
# | 📈 **Recall**    | Is the retrieved context covering all parts of the expected answer? | 🧠 **Retriever Coverage**<br>🔁 **Query Reformulation**<br>📚 **Index Completeness** | High recall means context contains all needed pieces to answer fully. | - Increase `top_k` retrieval<br>- Reformulate input queries<br>- Improve document indexing<br>- Add missing knowledge                   |



# 🧠 Example Scenario
# Suppose:

# You retrieved 5 documents (nodes)

# Only 3 were useful for the final answer → Precision = 3 / 5 = 0.6

# Your expected output has 4 sentences

# Only 2 sentences are supported by context → Recall = 2 / 4 = 0.5



# | Metric    | Numerator                                 | Denominator                           | Score | Interpretation                                    |
# | --------- | ----------------------------------------- | ------------------------------------- | ----- | ------------------------------------------------- |
# | Precision | Relevant nodes: 3                         | Retrieved nodes: 5                    | 0.6   | 60% of retrieved nodes were actually useful       |
# | Recall    | Supported sentences in expected output: 2 | Total sentences in expected output: 4 | 0.5   | Only half the answer was supported by the context |
