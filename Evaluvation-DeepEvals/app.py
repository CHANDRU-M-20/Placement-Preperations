from deepeval.metrics import FaithfulnessMetric
# from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import HallucinationMetric



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
