"""
Job Matching Engine
===================
Accepts job descriptions, performs hybrid semantic + keyword search
against the resume vector DB, scores and ranks candidates 0-100.
Includes Gemini tool-calling for intelligent metadata extraction.
"""

import os
import re
import json
import time
from typing import Optional

from google import genai
from google.genai import types

from resume_rag import (
    ResumeRAGPipeline,
    EmbeddingGenerator,
    VectorStore,
    RESUME_DIR,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_CHAT_MODEL = "gemini-3-flash-preview"
JD_FILE = os.path.join(os.path.dirname(__file__), "jd", "job-description.json")


# ---------------------------------------------------------------------------
# Tool Definitions for Gemini Function Calling
# ---------------------------------------------------------------------------
SEARCH_RESUMES_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_resumes",
            description="Search the resume vector database for candidates matching a query. Use this to find relevant resumes based on semantic similarity.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="The search query describing the ideal candidate or required skills/experience",
                    ),
                    "top_k": types.Schema(
                        type="INTEGER",
                        description="Number of top results to return (default 10)",
                    ),
                    "min_experience_years": types.Schema(
                        type="INTEGER",
                        description="Minimum years of experience filter (optional)",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="keyword_search_resumes",
            description="Search resumes using keyword matching in document text. Use this to find candidates with specific critical skills or certifications.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "keywords": types.Schema(
                        type="STRING",
                        description="Comma-separated list of must-have keywords to search for in resume text",
                    ),
                    "top_k": types.Schema(
                        type="INTEGER",
                        description="Number of top results to return (default 20)",
                    ),
                },
                required=["keywords"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_candidate_details",
            description="Get full details about a specific candidate by name. Returns all chunks and metadata for that candidate.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "candidate_name": types.Schema(
                        type="STRING",
                        description="Name of the candidate to look up",
                    ),
                },
                required=["candidate_name"],
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# JD Processor
# ---------------------------------------------------------------------------
class JDProcessor:
    """Load and process job descriptions."""

    @staticmethod
    def load_jds(jd_path: str = JD_FILE) -> list[dict]:
        with open(jd_path, "r") as f:
            return json.load(f)

    @staticmethod
    def jd_to_text(jd: dict) -> str:
        """Convert a structured JD to searchable text."""
        parts = [
            f"Job Title: {jd['title']}",
            f"Level: {jd['level']}",
            f"Experience Required: {jd['experience_years']} years",
        ]
        if "industries" in jd:
            parts.append(f"Industries: {', '.join(jd['industries'])}")
        if "responsibilities" in jd:
            parts.append("\nKey Responsibilities:")
            for resp in jd["responsibilities"]:
                parts.append(f"- {resp['name']}: {resp['description']}")
        return "\n".join(parts)

    @staticmethod
    def extract_keywords(jd: dict) -> list[str]:
        """Extract critical skill keywords from a JD."""
        text = JDProcessor.jd_to_text(jd)
        # Common technical terms and tools to look for
        keywords = set()
        # Extract from responsibilities
        for resp in jd.get("responsibilities", []):
            # Look for tool/technology names (capitalized words, abbreviations)
            tools = re.findall(r"\b[A-Z][A-Za-z+#]*(?:\.[A-Za-z]+)*\b", resp["description"])
            keywords.update(tools)
            # Look for abbreviations
            abbrevs = re.findall(r"\b[A-Z]{2,}\b", resp["description"])
            keywords.update(abbrevs)
        # Remove common non-skill words
        noise = {"The", "Act", "Own", "Work", "Lead", "Use", "AND", "For"}
        keywords -= noise
        return list(keywords)


# ---------------------------------------------------------------------------
# Hybrid Searcher
# ---------------------------------------------------------------------------
class HybridSearcher:
    """
    Combines semantic vector search with keyword-based filtering.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(self, pipeline: ResumeRAGPipeline):
        self.pipeline = pipeline

    def semantic_search(self, query: str, top_k: int = 20, min_exp: int | None = None) -> list[dict]:
        """Vector similarity search."""
        return self.pipeline.search(query, top_k=top_k, min_experience=min_exp)

    def keyword_search(self, keywords: list[str], top_k: int = 20) -> list[dict]:
        """
        Search for chunks containing specific keywords.
        Uses ChromaDB's where_document filter.
        """
        all_results = []
        for keyword in keywords[:10]:  # Limit to top 10 keywords
            try:
                # We need an embedding for the query - use the keyword itself
                query_embedding = self.pipeline.embedder.embed_query(keyword)
                results = self.pipeline.vector_store.query(
                    query_embedding=query_embedding,
                    n_results=top_k,
                    where_document={"$contains": keyword},
                )
                if results and results["documents"]:
                    for i, doc in enumerate(results["documents"][0]):
                        meta = results["metadatas"][0][i]
                        distance = results["distances"][0][i]
                        all_results.append({
                            "candidate_name": meta["candidate_name"],
                            "resume_path": meta["resume_path"],
                            "section": meta["section"],
                            "text": doc,
                            "distance": distance,
                            "similarity": round(1 - distance, 4),
                            "skills": meta["skills"],
                            "experience_years": meta["experience_years"],
                            "education": meta["education"],
                            "matched_keyword": keyword,
                        })
            except Exception:
                continue
        return all_results

    def hybrid_search(
        self,
        query: str,
        keywords: list[str],
        top_k: int = 10,
        min_exp: int | None = None,
    ) -> list[dict]:
        """
        Merge semantic and keyword results using Reciprocal Rank Fusion.
        """
        semantic_results = self.semantic_search(query, top_k=20, min_exp=min_exp)
        keyword_results = self.keyword_search(keywords, top_k=20) if keywords else []

        # Build RRF scores per candidate
        candidate_scores: dict[str, dict] = {}
        k = 60  # RRF constant

        # Semantic ranking
        for rank, result in enumerate(semantic_results):
            name = result["candidate_name"]
            rrf_score = 1.0 / (k + rank + 1)
            if name not in candidate_scores:
                candidate_scores[name] = {
                    "rrf_score": 0,
                    "semantic_rank": rank + 1,
                    "keyword_rank": None,
                    "best_result": result,
                    "semantic_similarity": result["similarity"],
                    "matched_keywords": set(),
                    "relevant_excerpts": [],
                }
            candidate_scores[name]["rrf_score"] += rrf_score
            candidate_scores[name]["relevant_excerpts"].append(result["text"][:200])

        # Keyword ranking
        seen_keyword_candidates = {}
        for result in keyword_results:
            name = result["candidate_name"]
            if name not in seen_keyword_candidates:
                seen_keyword_candidates[name] = len(seen_keyword_candidates)
            rank = seen_keyword_candidates[name]
            rrf_score = 1.0 / (k + rank + 1)

            if name not in candidate_scores:
                candidate_scores[name] = {
                    "rrf_score": 0,
                    "semantic_rank": None,
                    "keyword_rank": rank + 1,
                    "best_result": result,
                    "semantic_similarity": result.get("similarity", 0),
                    "matched_keywords": set(),
                    "relevant_excerpts": [],
                }
            candidate_scores[name]["rrf_score"] += rrf_score
            if result.get("matched_keyword"):
                candidate_scores[name]["matched_keywords"].add(result["matched_keyword"])
            if keyword_results:
                candidate_scores[name]["relevant_excerpts"].append(result["text"][:200])

        # Sort by RRF score and return top-K
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        return [
            {
                "candidate_name": name,
                "resume_path": data["best_result"]["resume_path"],
                "rrf_score": data["rrf_score"],
                "semantic_similarity": data["semantic_similarity"],
                "matched_keywords": list(data["matched_keywords"]),
                "skills": data["best_result"]["skills"],
                "experience_years": data["best_result"]["experience_years"],
                "education": data["best_result"]["education"],
                "relevant_excerpts": list(set(data["relevant_excerpts"]))[:3],
            }
            for name, data in ranked[:top_k]
        ]


# ---------------------------------------------------------------------------
# Candidate Scorer
# ---------------------------------------------------------------------------
class CandidateScorer:
    """
    Score candidates 0-100 based on:
      40% semantic similarity
      30% skill overlap
      20% experience fit
      10% education relevance
    """

    @staticmethod
    def score(candidate: dict, jd: dict, jd_keywords: list[str]) -> int:
        # Semantic similarity (already 0-1 from cosine)
        semantic = candidate.get("semantic_similarity", 0)

        # Skill overlap
        candidate_skills = set(
            s.strip().lower()
            for s in candidate.get("skills", "").split(",")
            if s.strip()
        )
        jd_skill_set = set(k.lower() for k in jd_keywords)
        if jd_skill_set:
            skill_overlap = len(candidate_skills & jd_skill_set) / len(jd_skill_set)
        else:
            skill_overlap = 0

        # Experience fit
        required_exp = CandidateScorer._parse_exp_range(jd.get("experience_years", "0"))
        candidate_exp = candidate.get("experience_years", 0)
        if required_exp > 0:
            if candidate_exp >= required_exp:
                exp_score = 1.0
            elif candidate_exp >= required_exp * 0.7:
                exp_score = 0.7
            else:
                exp_score = max(0, candidate_exp / required_exp)
        else:
            exp_score = 0.5  # No requirement specified

        # Education relevance (simple heuristic)
        edu = candidate.get("education", "").lower()
        edu_score = 0.5  # Default
        if any(term in edu for term in ["master", "ms ", "m.s.", "mtech", "m.tech"]):
            edu_score = 0.8
        if any(term in edu for term in ["phd", "ph.d", "doctorate"]):
            edu_score = 1.0
        if any(term in edu for term in ["bachelor", "bs ", "b.s.", "btech", "b.tech", "b.e."]):
            edu_score = 0.6

        # Weighted combination
        score = (
            0.40 * semantic
            + 0.30 * skill_overlap
            + 0.20 * exp_score
            + 0.10 * edu_score
        ) * 100

        return min(100, max(0, round(score)))

    @staticmethod
    def _parse_exp_range(exp_str: str) -> int:
        """Parse experience strings like '3–4', '10+', '9+'."""
        exp_str = str(exp_str)
        # "10+" → 10
        match = re.search(r"(\d+)\+", exp_str)
        if match:
            return int(match.group(1))
        # "3-4" or "3–4" → 3 (minimum)
        match = re.search(r"(\d+)\s*[-–—]\s*(\d+)", exp_str)
        if match:
            return int(match.group(1))
        # Plain number
        match = re.search(r"(\d+)", exp_str)
        if match:
            return int(match.group(1))
        return 0


# ---------------------------------------------------------------------------
# Match Explainer
# ---------------------------------------------------------------------------
class MatchExplainer:
    """Generate human-readable match reasoning."""

    @staticmethod
    def explain(candidate: dict, jd: dict, match_score: int, jd_keywords: list[str]) -> str:
        parts = []

        # Score summary
        if match_score >= 80:
            parts.append(f"Strong match (score: {match_score}/100).")
        elif match_score >= 60:
            parts.append(f"Good match (score: {match_score}/100).")
        elif match_score >= 40:
            parts.append(f"Moderate match (score: {match_score}/100).")
        else:
            parts.append(f"Weak match (score: {match_score}/100).")

        # Skills
        candidate_skills = set(
            s.strip().lower()
            for s in candidate.get("skills", "").split(",")
            if s.strip()
        )
        jd_skill_set = set(k.lower() for k in jd_keywords)
        matched = candidate_skills & jd_skill_set
        missing = jd_skill_set - candidate_skills

        if matched:
            parts.append(f"Matched skills: {', '.join(sorted(matched))}.")
        if missing:
            parts.append(f"Missing skills: {', '.join(sorted(missing))}.")

        # Experience
        exp = candidate.get("experience_years", 0)
        required = CandidateScorer._parse_exp_range(jd.get("experience_years", "0"))
        if exp >= required:
            parts.append(f"Experience ({exp} yrs) meets the {required}+ year requirement.")
        elif exp > 0:
            parts.append(f"Experience ({exp} yrs) is below the {required}+ year requirement.")

        # Education
        edu = candidate.get("education", "")
        if edu:
            parts.append(f"Education: {edu[:100]}.")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Job Matcher  (main class with tool calling)
# ---------------------------------------------------------------------------
class JobMatcher:
    """
    Main matching engine. Uses Gemini tool calling to intelligently
    search and match resumes against job descriptions.
    """

    def __init__(self, api_key: str | None = None):
        self.pipeline = ResumeRAGPipeline(api_key=api_key)
        self.searcher = HybridSearcher(self.pipeline)
        self.scorer = CandidateScorer()
        self.explainer = MatchExplainer()
        self.jd_processor = JDProcessor()

        if api_key:
            self.genai_client = genai.Client(api_key=api_key)
        else:
            self.genai_client = genai.Client()

    def match_jd(self, jd: dict, top_k: int = 10) -> dict:
        """
        Match a single JD against all resumes.
        Returns structured output in the required format.
        """
        jd_text = self.jd_processor.jd_to_text(jd)
        jd_keywords = self.jd_processor.extract_keywords(jd)
        min_exp = self.scorer._parse_exp_range(jd.get("experience_years", "0"))

        print(f"\n{'━' * 60}")
        print(f"Matching: {jd['title']} ({jd['level']})")
        print(f"Keywords: {', '.join(jd_keywords[:10])}")
        print(f"Min experience: {min_exp} years")
        print(f"{'━' * 60}")

        # Hybrid search
        candidates = self.searcher.hybrid_search(
            query=jd_text,
            keywords=jd_keywords,
            top_k=top_k,
            min_exp=None,  # Don't hard-filter, let scoring handle it
        )

        # Score, explain, and format
        top_matches = []
        for candidate in candidates:
            score = self.scorer.score(candidate, jd, jd_keywords)
            reasoning = self.explainer.explain(candidate, jd, score, jd_keywords)

            # Determine matched skills
            candidate_skill_set = set(
                s.strip() for s in candidate.get("skills", "").split(",") if s.strip()
            )
            matched_skills = [
                s for s in candidate_skill_set
                if any(k.lower() in s.lower() for k in jd_keywords)
            ]

            top_matches.append({
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "match_score": score,
                "matched_skills": matched_skills + candidate.get("matched_keywords", []),
                "relevant_excerpts": candidate.get("relevant_excerpts", []),
                "reasoning": reasoning,
            })

        # Sort by match_score descending
        top_matches.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "job_description": jd_text,
            "top_matches": top_matches[:top_k],
        }

    def match_jd_with_tools(self, jd: dict, top_k: int = 10) -> dict:
        """
        Match a JD using Gemini tool calling for intelligent search orchestration.
        The LLM decides what searches to run and how to interpret results.
        """
        jd_text = self.jd_processor.jd_to_text(jd)

        print(f"\n{'━' * 60}")
        print(f"[Tool Calling] Matching: {jd['title']} ({jd['level']})")
        print(f"{'━' * 60}")

        # System prompt for the tool-calling agent
        system_prompt = """You are a recruitment AI assistant. Given a job description, 
        use the available tools to search a resume database and find the best matching candidates.
        
        Strategy:
        1. First, do a semantic search with the full job description to find broadly relevant candidates.
        2. Then, do keyword searches for critical/must-have skills mentioned in the JD.
        3. For promising candidates, get their full details to make a thorough assessment.
        4. Provide your final analysis with reasoning for each match.
        
        Be thorough but efficient. Focus on finding truly relevant candidates."""

        # Start the conversation
        user_message = f"""Find the top {top_k} candidates matching this job description:

{jd_text}

Use the search tools to find and evaluate candidates. After searching, provide your final 
ranking with scores (0-100) and reasoning for each candidate."""

        response = self.genai_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[SEARCH_RESUMES_TOOL],
                temperature=0.1,
            ),
        )

        # Process tool calls in a loop
        all_search_results = []
        max_iterations = 8
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Check if there are function calls
            has_function_calls = False
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        has_function_calls = True
                        break

            if not has_function_calls:
                break

            # Process each function call
            function_responses = []
            for part in response.candidates[0].content.parts:
                if not part.function_call:
                    continue

                fn_name = part.function_call.name
                fn_args = dict(part.function_call.args) if part.function_call.args else {}
                print(f"  🔧 Tool call: {fn_name}({json.dumps(fn_args, default=str)})")

                # Execute the tool
                result = self._execute_tool(fn_name, fn_args)
                all_search_results.extend(result if isinstance(result, list) else [result])

                function_responses.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": json.dumps(result[:5] if isinstance(result, list) else result, default=str)},
                    )
                )

            # Continue the conversation with tool results
            response = self.genai_client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=user_message)]),
                    response.candidates[0].content,
                    types.Content(role="tool", parts=function_responses),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[SEARCH_RESUMES_TOOL],
                    temperature=0.1,
                ),
            )

        # Extract the final text response from Gemini
        final_text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    final_text += part.text

        print(f"\n  📝 Gemini analysis complete ({iteration} iterations)")

        # Also do our own structured matching as fallback/supplement
        jd_keywords = self.jd_processor.extract_keywords(jd)
        structured_result = self.match_jd(jd, top_k=top_k)

        # Augment with Gemini's analysis
        structured_result["gemini_analysis"] = final_text

        return structured_result

    def _execute_tool(self, fn_name: str, fn_args: dict) -> list[dict] | dict:
        """Execute a tool call and return the result."""
        if fn_name == "search_resumes":
            query = fn_args.get("query", "")
            top_k = int(fn_args.get("top_k", 10))
            min_exp = fn_args.get("min_experience_years")
            if min_exp is not None:
                min_exp = int(min_exp)
            results = self.pipeline.search(query, top_k=top_k, min_experience=min_exp)
            # Simplify for the LLM
            return [
                {
                    "candidate_name": r["candidate_name"],
                    "resume_path": r["resume_path"],
                    "similarity": r["similarity"],
                    "section": r["section"],
                    "experience_years": r["experience_years"],
                    "skills": r["skills"][:200],
                    "excerpt": r["text"][:300],
                }
                for r in results
            ]

        elif fn_name == "keyword_search_resumes":
            keywords = [k.strip() for k in fn_args.get("keywords", "").split(",")]
            top_k = int(fn_args.get("top_k", 20))
            results = self.searcher.keyword_search(keywords, top_k=top_k)
            return [
                {
                    "candidate_name": r["candidate_name"],
                    "similarity": r["similarity"],
                    "matched_keyword": r.get("matched_keyword", ""),
                    "skills": r["skills"][:200],
                    "excerpt": r["text"][:300],
                }
                for r in results
            ]

        elif fn_name == "get_candidate_details":
            name = fn_args.get("candidate_name", "")
            # Search for this specific candidate
            try:
                results = self.pipeline.vector_store.query(
                    query_embedding=self.pipeline.embedder.embed_query(name),
                    n_results=20,
                    where={"candidate_name": {"$eq": name}},
                )
                if results and results["documents"]:
                    return {
                        "candidate_name": name,
                        "chunks": [
                            {
                                "section": results["metadatas"][0][i]["section"],
                                "text": doc[:500],
                                "skills": results["metadatas"][0][i]["skills"],
                                "experience_years": results["metadatas"][0][i]["experience_years"],
                                "education": results["metadatas"][0][i]["education"],
                            }
                            for i, doc in enumerate(results["documents"][0])
                        ],
                    }
            except Exception:
                pass
            return {"candidate_name": name, "error": "Not found"}

        return {"error": f"Unknown tool: {fn_name}"}

    def match_all_jds(self, jd_path: str = JD_FILE, top_k: int = 10, use_tools: bool = False) -> list[dict]:
        """Match all JDs from the JSON file."""
        jds = self.jd_processor.load_jds(jd_path)
        results = []
        for jd in jds:
            if use_tools:
                result = self.match_jd_with_tools(jd, top_k=top_k)
            else:
                result = self.match_jd(jd, top_k=top_k)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Job Matching Engine")
    parser.add_argument(
        "--jd-file", default=JD_FILE, help="Path to job descriptions JSON"
    )
    parser.add_argument(
        "--jd-id", type=int, default=None, help="Match a specific JD by ID (1-indexed)"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of top matches"
    )
    parser.add_argument(
        "--use-tools", action="store_true",
        help="Use Gemini tool calling for intelligent search"
    )
    parser.add_argument(
        "--output", default=None, help="Output JSON file path"
    )

    args = parser.parse_args()
    matcher = JobMatcher()

    if args.jd_id is not None:
        jds = JDProcessor.load_jds(args.jd_file)
        jd = next((j for j in jds if j["id"] == args.jd_id), None)
        if not jd:
            print(f"JD with id={args.jd_id} not found")
            exit(1)
        if args.use_tools:
            results = [matcher.match_jd_with_tools(jd, top_k=args.top_k)]
        else:
            results = [matcher.match_jd(jd, top_k=args.top_k)]
    else:
        results = matcher.match_all_jds(
            jd_path=args.jd_file,
            top_k=args.top_k,
            use_tools=args.use_tools,
        )

    # Print results
    for result in results:
        print(f"\n{'═' * 70}")
        print(f"JOB: {result['job_description'][:80]}...")
        print(f"{'═' * 70}")
        for match in result["top_matches"]:
            print(f"\n  📋 {match['candidate_name']} — Score: {match['match_score']}/100")
            print(f"     Skills: {', '.join(match['matched_skills'][:5])}")
            print(f"     {match['reasoning'][:150]}")

        if "gemini_analysis" in result:
            print(f"\n  🤖 Gemini Analysis:\n{result['gemini_analysis'][:500]}")

    # Save to file
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "match_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {output_path}")
