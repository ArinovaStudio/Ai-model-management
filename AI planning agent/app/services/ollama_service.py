import json
import logging
from typing import Any, Optional

from ollama import AsyncClient

from app.config import settings

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert requirements analyst.
Extract software functional requirements and features explicitly mentioned in the provided text.
Ignore metadata, email headers, and greetings.

CRITICAL RULE FOR LLAMA 3: Do NOT invent features that are not explicitly requested or inferred directly from the user text. Only output technologies explicitly written in the text to the "related_technologies" array.

For each feature, output a JSON object with:
  - "title": (Max 5 words)
  - "description": (Detailed explanation of what the user wants)
  - "category": ("functional" or "feature")
  - "priority": ("high", "medium", or "low")
  - "related_technologies": (Array of strings representing tech explicitly named in the text)
  - "ambiguous": (boolean)
  - "clarification_prompt": (string or null if ambiguous is true)

Output ONLY a JSON array of these objects. Do not write anything outside the JSON array. No conversational text.
"""

CLARIFICATION_SYSTEM_PROMPT = """You are a requirements analyst helping to clarify ambiguous software requirements.
Given the original requirement and a user's clarification, rewrite the requirement clearly and completely.
Return ONLY a valid JSON object. Do not wrap in markdown code blocks like ```json ... ```. No conversational text, no extra spaces.

{
  "title": "clarified title",
  "description": "clarified detailed description matching user details",
  "category": "functional | non-functional | feature | user-story | bug-fix | enhancement",
  "priority": "critical | high | medium | low",
  "ambiguous": false,
  "clarification_prompt": null
}
"""

GAP_ANALYSIS_SYSTEM_PROMPT = """You are an expert project risk and gap analyst. Given a project's full context including its description, scope, constraints, timeline, feature requirements, and technology stack, perform a comprehensive gap analysis.

Produce three structured reports as a JSON object. Return ONLY valid JSON. Do not write markdown blocks or text fences.

Required JSON Structure:
{
  "assumptions": [
    {
      "title": "short title",
      "description": "Things the project assumes without explicit evidence in the text.",
      "category": "scope | timeline | technology | resource | process",
      "severity": "high | medium | low"
    }
  ],
  "clarifications": [
    {
      "title": "short title",
      "description": "Areas where information is missing, ambiguous, or incomplete.",
      "category": "ambiguity | incompleteness | missing_detail | contradiction",
      "severity": "high | medium | low",
      "related_feature": "title of related feature or null"
    }
  ],
  "risks": [
    {
      "title": "short title",
      "description": "Potential risks and blockages that could impact project success.",
      "category": "missing_dependency | unrealistic_deadline | unclear_scope | resource_gap | tech_incompatibility | requirement_gap",
      "severity": "critical | high | medium | low",
      "type": "risk | blockage"
    }
  ]
}
If a section has no items, use an empty array []. Do not add trailing commas.
"""

ARCH_TRINITY_SYSTEM_PROMPT = """You are a Principal Software Architect.
Given a project context and a specific feature request, define the implementation strategy, architecture, and deployment plan.

CRITICAL FOR LLAMA 3: You must restrict your strategy ONLY to the technology stack explicitly defined in the provided Project Context. Do not introduce alternative frameworks, databases, or language ecosystems.

Use the following XML output format EXACTLY:
<thinking>
Think step-by-step about the specific technical requirements, edge cases, and schemas required for this feature given the project's tech stack.
</thinking>
<trinity>
  <implementation>
    (Core engineering approach, explicit module dependencies, edge cases and failure handling protocols for this specific feature)
  </implementation>
  <architecture>
    (Logical sub-components, database entities, relational models, and detailed user/testing flows including end-to-end and final UAT criteria)
  </architecture>
  <deployment>
    (Server provisioning requirements, environment configuration variables, database migration/schema seeding steps, and verification sequences)
  </deployment>
</trinity>
"""

WBS_TASK_SYSTEM_PROMPT = """You are an Expert Technical Project Manager.
Given an Architectural Trinity for a feature, break it down into sequential milestones and granular tasks.

CRITICAL GENERATION RULES FOR LLAMA 3:
1. Do not use generic task names like "Implement API". Make them explicitly detailed to the feature details provided in the architectural text.
2. Break down the breakdown into exactly 5 logical, sequential Milestone phases:
   - Milestone 1: Core Foundation & Context Baseline
   - Milestone 2: Functional Feature Engineering & System Core
   - Milestone 3: Interface Integration & Frontend Synchronization
   - Milestone 4: Verification, Hardening, & Risk Mitigation (including E2E and UAT)
   - Milestone 5: Deployment Go-Live & Pipeline Sign-off

Return ONLY a valid JSON object matching this schema exactly. No trailing commas, conversational wrap-around text, or markdown code blocks:
{
  "milestones": [
    {
      "title": "Milestone Phase Title",
      "description": "Milestone Description detailing phase goals",
      "tasks": [
        {
          "title": "Granular, context-specific task title",
          "description": "Explicit step-by-step technical execution plan for this task",
          "estimated_hours": 4.5,
          "depends_on": ["Title of previous task this blocks on", "Can be empty [] for independent tasks"]
        }
      ]
    }
  ]
}
"""


PRODUCT_STRATEGY_PROMPT = """You are a Chief Product Officer. Given the project context, define the product strategy.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your strategy inside <thinking>...</thinking> XML tags. Discuss market positioning and differentiation.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. Do NOT wrap the JSON in markdown code blocks.

JSON SCHEMA:
{
  "productDefinition": "(string) Core product definition and vision",
  "valueProposition": "(string) Key value propositions",
  "differentiation": "(string) What differentiates this product",
  "marketPositioning": "(string) Target market and positioning",
  "useCases": ["(string) Primary use case 1", "(string) Primary use case 2"]
}
"""

FEATURE_SPEC_PROMPT = """You are a Lead Product Manager. Given the Product Strategy, define the feature specifications.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your feature choices inside <thinking>...</thinking> XML tags. Discuss trade-offs and priorities.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. Do NOT wrap the JSON in markdown code blocks.

JSON SCHEMA:
{
  "coreFeatures": [{"name": "(string)", "description": "(string)", "priority": "enum (high, medium, low)"}],
  "advancedFeatures": [{"name": "(string)", "description": "(string)", "priority": "enum (high, medium, low)"}],
  "prioritizationMatrix": {"mustHave": ["(string)"], "shouldHave": ["(string)"], "couldHave": ["(string)"]}
}
"""

SYSTEM_ARCHITECTURE_PROMPT = """You are a Principal Architect. Given the Feature Spec, define the system architecture.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your architectural choices inside <thinking>...</thinking> XML tags. Discuss scalability and microservice boundaries.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. Do NOT wrap the JSON in markdown code blocks.
4. For mermaidSystemFlow, output raw MermaidJS syntax ONLY. Do NOT wrap it in ```mermaid markdown blocks.

JSON SCHEMA:
{
  "highLevel": "(string) High-level architecture description",
  "microservices": [{"name": "(string)", "responsibility": "(string)", "techStack": ["(string)"]}],
  "mermaidSystemFlow": "(string) RAW MERMAID SYNTAX STARTING WITH 'graph TD'"
}
"""

DATABASE_DESIGN_PROMPT = """You are a Lead DBA. Given the system architecture, define the database design.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your schema inside <thinking>...</thinking> XML tags. Discuss relations and normalization.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. Do NOT wrap the JSON in markdown code blocks.
4. For mermaidERD, output raw MermaidJS syntax ONLY. Do NOT wrap it in ```mermaid markdown blocks.

JSON SCHEMA:
{
  "entityModeling": [{"entity": "(string)", "attributes": [{"name": "(string)", "type": "(string)", "constraints": "enum (PK, FK, NOT NULL, etc)"}]}],
  "normalization": "(string) Description of normalization approach",
  "mermaidERD": "(string) RAW MERMAID SYNTAX STARTING WITH 'erDiagram'"
}
"""

API_AND_SECURITY_PROMPT = """You are a Lead Security & API Engineer. Given the database design and architecture, define the API and security strategy.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your API boundaries and security inside <thinking>...</thinking> XML tags.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. Do NOT wrap the JSON in markdown code blocks.

JSON SCHEMA:
{
  "restEndpoints": [{"path": "(string)", "method": "enum (GET, POST, PUT, DELETE)", "description": "(string)", "auth": "enum (required, optional)"}],
  "authStrategy": "(string) Authentication and authorization approach",
  "dataProtection": "(string) Encryption, compliance, and data protection measures"
}
"""


class OllamaService:
    def __init__(self) -> None:
        self.client = AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.model = settings.OLLAMA_MODEL
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        try:
            await self.client.list()
            logger.info("Ollama service is available (model: %s)", self.model)
            return True
        except Exception as exc:
            logger.warning("Ollama service is not available: %s", exc)
            return False

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

    async def generate_intelligence_layer(self, system_prompt: str, context: str, temperature: float = 0.1) -> dict[str, Any]:
        if not await self.is_available():
            return {}

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                options={"temperature": temperature, "num_predict": 4096},
            )

            raw = response["message"]["content"].strip()
            parsed = self._parse_json_response(raw)
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Ollama intelligence layer returned non-dict: %s", type(parsed))
            return {}
        except Exception as exc:
            logger.error("Ollama intelligence layer generation failed: %s", exc)
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
