import json
import logging
from typing import Any, Optional

from ollama import AsyncClient

from app.config import settings

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert requirements analyst for software projects.
Extract functional requirements, features, and user stories from the project document below.
For each item, determine:
  - title (short, descriptive)
  - description (detailed explanation)
  - category (one of: functional, non-functional, feature, user-story, bug-fix, enhancement)
  - priority (one of: critical, high, medium, low)
  - related_technologies (array of technology names mentioned in context of this requirement)
  - ambiguous (boolean: true if the requirement is unclear, incomplete, or needs clarification)
  - clarification_prompt (if ambiguous, a question to ask the user; otherwise null)

Return ONLY valid JSON as an array of objects. No markdown, no code fences, no extra text.
If no requirements are found, return an empty array [].

Document content:
"""

CLARIFICATION_SYSTEM_PROMPT = """You are a requirements analyst helping to clarify ambiguous software requirements.
Given the original requirement and a user's clarification, rewrite the requirement clearly.
Return ONLY valid JSON:
{
  "title": "clarified title",
  "description": "clarified detailed description",
  "category": "functional | non-functional | feature | user-story | bug-fix | enhancement",
  "priority": "critical | high | medium | low",
  "ambiguous": false,
  "clarification_prompt": null
}
No markdown, no code fences, no extra text.
"""

GAP_ANALYSIS_SYSTEM_PROMPT = """You are an expert project risk and gap analyst. Given a project's full context including its description, scope, constraints, timeline, feature requirements, and technology stack, perform a comprehensive gap analysis.

Produce three structured reports as a JSON object with the following keys:

1. "assumptions": Things the project seems to assume without explicit evidence. Each item:
   - title (short)
   - description (detailed explanation)
   - category (one of: scope, timeline, technology, resource, process)
   - severity (high, medium, low)

2. "clarifications": Areas where information is missing, ambiguous, or incomplete. Each item:
   - title (short)
   - description (what needs clarification)
   - category (one of: ambiguity, incompleteness, missing_detail, contradiction)
   - severity (high, medium, low)
   - related_feature (title of related feature if applicable, otherwise null)

3. "risks": Potential risks and blockages that could impact project success. Each item:
   - title (short)
   - description (detailed explanation of the risk)
   - category (one of: missing_dependency, unrealistic_deadline, unclear_scope, resource_gap, tech_incompatibility, requirement_gap)
   - severity (critical, high, medium, low)
   - type ("risk" for potential issues, "blockage" for definite blockers)

Return ONLY valid JSON. No markdown, no code fences, no extra text.
If a section has no items, use an empty array.

Project Context:
"""


class OllamaService:
    def __init__(self) -> None:
        self.client = AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.model = settings.OLLAMA_MODEL
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            await self.client.list()
            self._available = True
            logger.info("Ollama service is available (model: %s)", self.model)
        except Exception as exc:
            self._available = False
            logger.warning("Ollama service is not available: %s", exc)
        return self._available

    async def extract_requirements(self, content: str) -> list[dict[str, Any]]:
        if not await self.is_available():
            logger.warning("Ollama unavailable; falling back to keyword extraction")
            return []

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                options={"temperature": 0.1, "num_predict": 4096},
            )

            raw = response["message"]["content"].strip()
            return self._parse_json_response(raw)

        except Exception as exc:
            logger.error("Ollama requirement extraction failed: %s", exc)
            return []

    async def clarify_requirement(
        self, original: dict[str, Any], clarification: str
    ) -> dict[str, Any]:
        if not await self.is_available():
            return {
                "title": original.get("title", ""),
                "description": clarification,
                "category": original.get("category", "functional"),
                "priority": original.get("priority", "medium"),
                "ambiguous": False,
                "clarification_prompt": None,
            }

        prompt = json.dumps({"original_requirement": original, "clarification": clarification})

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1, "num_predict": 2048},
            )

            raw = response["message"]["content"].strip()
            result = self._parse_json_response(raw)
            if isinstance(result, list) and result:
                return result[0]
            if isinstance(result, dict):
                return result
            return self._default_clarified(original, clarification)

        except Exception as exc:
            logger.error("Ollama clarification failed: %s", exc)
            return self._default_clarified(original, clarification)

    async def analyze_gaps(self, project_context: str) -> dict[str, Any]:
        if not await self.is_available():
            logger.warning("Ollama unavailable; falling back to keyword gap analysis")
            return {}

        max_chars = 12000
        if len(project_context) > max_chars:
            project_context = project_context[:max_chars] + "\n...[truncated]"

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": GAP_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": project_context},
                ],
                options={"temperature": 0.1, "num_predict": 8192},
            )

            raw = response["message"]["content"].strip()
            parsed = self._parse_json_response(raw)
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Ollama gap analysis returned non-dict: %s", type(parsed))
            return {}

        except Exception as exc:
            logger.error("Ollama gap analysis failed: %s", exc)
            return {}

    def _parse_json_response(self, raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(line for line in lines if not line.startswith("```"))
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to parse Ollama response as JSON: %.200s", raw)
            return []

    def _default_clarified(self, original: dict, clarification: str) -> dict:
        return {
            "title": original.get("title", ""),
            "description": clarification,
            "category": original.get("category", "functional"),
            "priority": original.get("priority", "medium"),
            "ambiguous": False,
            "clarification_prompt": None,
        }


ollama_service = OllamaService()
