import json
import logging
from typing import Any, Optional

from ollama import AsyncClient

from app.config import settings

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert requirements analyst.
Extract software functional requirements and features from the document.
Ignore metadata, email headers, and greetings.

For each feature, output a JSON object with:
  - "title": (Max 5 words, e.g. "Secure Document Upload")
  - "description": (Detailed explanation)
  - "category": ("functional" or "feature")
  - "priority": ("high", "medium", or "low")
  - "related_technologies": (Array of technologies)
  - "ambiguous": (boolean)
  - "clarification_prompt": (string or null)

Output ONLY a JSON array of these objects. Example:
[
  {
    "title": "User Authentication System",
    "description": "Users must be able to log in securely.",
    "category": "functional",
    "priority": "high",
    "related_technologies": [],
    "ambiguous": false,
    "clarification_prompt": null
  }
]
Do not write anything outside the JSON array.
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

ARCH_TRINITY_SYSTEM_PROMPT = """You are a Principal Software Architect.
Given a project context and a specific feature request, define the implementation strategy, architecture, and deployment plan.

Use the following XML output format EXACTLY:
<thinking>
Think step-by-step about the technical requirements, components, and data flow.
</thinking>
<trinity>
  <implementation>
    (Core engineering stack, dependencies, edge cases for this feature)
  </implementation>
  <architecture>
    (Logical components, data models, logic flow)
  </architecture>
  <deployment>
    (Server needs, DB migration, deployment steps)
  </deployment>
</trinity>
"""

WBS_TASK_SYSTEM_PROMPT = """You are an Expert Technical Project Manager.
Given an Architectural Trinity for a feature, break it down into granular milestones and tasks.

Return ONLY a valid JSON object matching this schema exactly:
{
  "milestones": [
    {
      "title": "Milestone Title",
      "description": "Milestone Description",
      "tasks": [
        {
          "title": "Granular Task Title",
          "description": "Task Description",
          "estimated_hours": 4.5
        }
      ]
    }
  ]
}

Example:
Input: <trinity><implementation>Set up Postgres Database...</implementation></trinity>
Output:
{
  "milestones": [
    {
      "title": "Database Initialization",
      "description": "Set up relational schemas",
      "tasks": [
        {
          "title": "Create User Tables",
          "description": "Define the User model and migration scripts.",
          "estimated_hours": 3.0
        }
      ]
    }
  ]
}

Do NOT write anything outside the JSON.
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

    async def generate_architecture_trinity(self, context: str, feature: str) -> str:
        if not await self.is_available():
            return "<trinity><implementation>Standard</implementation><architecture>Default</architecture><deployment>Basic</deployment></trinity>"

        prompt = f"<context>\n{context}\n</context>\n<feature>\n{feature}\n</feature>"
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": ARCH_TRINITY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2, "num_predict": 4096},
            )
            raw = response["message"]["content"].strip()
            # Extract just the <trinity> part if possible
            if "<trinity>" in raw and "</trinity>" in raw:
                start = raw.find("<trinity>")
                end = raw.find("</trinity>") + len("</trinity>")
                return raw[start:end]
            return raw
        except Exception as exc:
            logger.error("Ollama trinity generation failed: %s", exc)
            return ""

    async def generate_wbs_tasks(self, trinity_xml: str) -> dict[str, Any]:
        if not await self.is_available():
            return {}

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": WBS_TASK_SYSTEM_PROMPT},
                    {"role": "user", "content": trinity_xml},
                ],
                options={"temperature": 0.1, "num_predict": 4096},
            )
            raw = response["message"]["content"].strip()
            parsed = self._parse_json_response(raw)
            if isinstance(parsed, dict) and "milestones" in parsed:
                return parsed
            return {}
        except Exception as exc:
            logger.error("Ollama WBS task generation failed: %s", exc)
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
