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
3. For each JSON string property, provide a highly detailed, professional, multi-sentence explanation. Do not use short, generic placeholder phrases. Explain the 'why' and the 'how'.
4. Do NOT wrap the JSON in markdown code blocks.

JSON SCHEMA:
{
  "productDefinition": "(string) Very detailed, high-fidelity core product definition and vision (minimum 3-4 sentences)",
  "valueProposition": "(string) Comprehensive value propositions detailing specific business and user benefits",
  "differentiation": "(string) Deep competitive differentiators explaining how this product wins in the market",
  "marketPositioning": "(string) Specific target customer segments, user personas, and strategic market positioning details",
  "useCases": ["(string) Detailed use case 1 with user goal and system response", "(string) Detailed use case 2 with user goal and system response"]
}
"""

FEATURE_SPEC_PROMPT = """You are a Lead Product Manager. Given the Product Strategy, define the feature specifications.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your feature choices inside <thinking>...</thinking> XML tags. Discuss trade-offs and priorities.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. For all feature descriptions, write a complete, detailed engineering spec paragraph rather than a single sentence.
4. Do NOT wrap the JSON in markdown code blocks.

JSON SCHEMA:
{
  "coreFeatures": [{"name": "(string) Feature name", "description": "(string) Rich, detailed multi-sentence description specifying functional inputs, outputs, and business logic", "priority": "enum (high, medium, low)"}],
  "advancedFeatures": [{"name": "(string) Feature name", "description": "(string) Rich, detailed multi-sentence description specifying functional inputs, outputs, and business logic", "priority": "enum (high, medium, low)"}],
  "prioritizationMatrix": {"mustHave": ["(string) Must-have feature summary"], "shouldHave": ["(string) Should-have feature summary"], "couldHave": ["(string) Could-have feature summary"]}
}
"""

SYSTEM_ARCHITECTURE_PROMPT = """You are a Principal Architect. Given the Feature Spec, define the system architecture.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your architectural choices inside <thinking>...</thinking> XML tags. Discuss scalability and microservice boundaries.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. For the highLevel architecture description, write a rich, multi-paragraph architectural brief describing design patterns, communication protocols, and technology integrations.
4. For mermaidSystemFlow, output raw MermaidJS syntax ONLY. Do NOT wrap it in ```mermaid markdown blocks.

JSON SCHEMA:
{
  "highLevel": "(string) Rich, comprehensive high-level architecture overview detailing patterns (e.g. MVC, microservices), communication (e.g. REST, gRPC), caching, and horizontal scalability plans",
  "microservices": [{"name": "(string)", "responsibility": "(string) Detailed explanation of this service's core responsibilities and technical capabilities", "techStack": ["(string)"]}],
  "mermaidSystemFlow": "(string) RAW MERMAID SYNTAX STARTING WITH 'graph TD'"
}
"""

DATABASE_DESIGN_PROMPT = """You are a Lead DBA. Given the system architecture, define the database design.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your schema inside <thinking>...</thinking> XML tags. Discuss relations and normalization.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. For the normalization field, write a highly detailed, professional DBA report describing indexes, query optimization, foreign key constraints, and scaling strategies.
4. For mermaidERD, output raw MermaidJS syntax ONLY. Do NOT wrap it in ```mermaid markdown blocks.

JSON SCHEMA:
{
  "entityModeling": [{"entity": "(string) Table/Entity name", "attributes": [{"name": "(string) Attribute name", "type": "(string) DB Type", "constraints": "enum (PK, FK, NOT NULL, etc)"}]}],
  "normalization": "(string) Deep architectural report on normalization (1NF, 2NF, 3NF), index strategies, indexing fields, and specific query patterns designed to optimize performance",
  "mermaidERD": "(string) RAW MERMAID SYNTAX STARTING WITH 'erDiagram'"
}
"""

API_AND_SECURITY_PROMPT = """You are a Lead Security & API Engineer. Given the database design and architecture, define the API and security strategy.
CRITICAL INSTRUCTIONS FOR LLAMA 3:
1. You must first reason about your API boundaries and security inside <thinking>...</thinking> XML tags.
2. Immediately after the </thinking> tag, you MUST output a strictly valid JSON object.
3. Ensure authStrategy and dataProtection fields are written as rich, comprehensive guidelines detailing specific industry standards, libraries, algorithms, and protocols.

JSON SCHEMA:
{
  "restEndpoints": [{"path": "(string) REST endpoint path", "method": "enum (GET, POST, PUT, DELETE)", "description": "(string) Rich explanation of request inputs, validation, and expected responses", "auth": "enum (required, optional)"}],
  "authStrategy": "(string) Detailed, multi-sentence authentication architecture (e.g., OAuth2, JWT with specific claim validation, rotation policies)",
  "dataProtection": "(string) Comprehensive data protection measures (e.g., AES-256 at rest, TLS 1.3 in transit, secrets management, PCI/HIPAA compliance strategy if applicable)"
}
"""


CORE_TECHNICAL_BLUEPRINT_PROMPT = """You are a Lead Solutions Architect. Your goal is to generate a comprehensive, highly detailed Core Technical Blueprint for the project in Markdown format.
You MUST output exactly the following 8 sections:

# 1. Product Definition
- **Product Name**: (Name)
- **Product Type**: (Type)
- **Core Value Proposition**: (Detailed, multi-sentence vision)
- **Primary Users**: (List key user personas and goals)

# 2. Solution Overview
- **High-level System Description**: (System context and core design)
- **End-to-End Workflow**: (Sequence of interactions from user request to system persistence)
- **Key Capabilities & Core Functionality**: (Bullet points detailing major functionalities)

# 3. System Architecture
Describe major system components, database schemas, interactions, and boundaries. Include a text-based ASCII flow chart or a Mermaid graph code block (e.g. ```mermaid graph TD ... ```) showing component connections.

# 4. Core Modules
For each major module/service, provide:
- **Module Name**
- **Purpose**: (What it does)
- **Responsibilities**: (List key functions)
- **Inputs & Outputs**: (Expected parameters and returns)
- **Dependencies**: (Libraries, modules, or services relied on)

# 5. Technical Stack
Provide specific technology choices for:
- **Frontend Technologies**: (Frameworks, libraries, state management)
- **Backend Technologies**: (Languages, frameworks, APIs)
- **Databases & Cache**: (Relational, document, caching layers)
- **AI/ML Technologies**: (Models, engines, or services if any)
- **Infrastructure**: (Container runtimes, orchestrators, serverless engines)
- **Third-Party Integrations**: (Payment gateways, notifications, authentication)

# 6. User & System Flows
Describe the user journey, system workflow, processing pipeline, and event lifecycle.

# 7. Implementation Strategy
- **MVP Scope**: (What is in scope for the initial release)
- **Development Phases**: (Phased release strategy)
- **Build Sequence**: (Logical ordering of module development)
- **Future Expansion Opportunities**: (Post-MVP roadmap concepts)

# 8. Security, Scalability & Risks
- **Security Considerations**: (Auth, encryption at rest/in transit, rate limiting)
- **Performance & Scalability Approach**: (Caching, DB indexing, container scaling)
- **Technical Risks & Mitigation Strategies**: (Provide a table linking Risk, Impact, and Mitigation Strategy)

Ensure the output is written professionally, technically, and rich in text context. Do not output anything outside the Markdown report.
"""

PROJECT_CONSTRAINTS_PROMPT = """You are a Principal Program Manager. Your goal is to generate a highly detailed Project Constraints report in Markdown format.
You MUST output exactly the following 7 sections:

# A. Budget Constraints
- **Approved Budget**: (Value)
- **Budget Flexibility**: (Fixed / Flexible / Unknown - with explanation)

# B. Timeline Constraints
- **Target Launch Date**: (Target date)
- **Hard Deadline**: (Yes / No - with explanation of critical path drivers)
- **Maximum Acceptable Delay**: (Timeline tolerance details)

# C. Resource Constraints
- **Available Team Size**: (Team count/structure)
- **Key Resource Limitations**: (Bottlenecks, specialist dependencies)
- **Hiring Allowed**: (Yes / No - with constraints/conditions)

# D. Technical Constraints
- **Required Technologies**: (Mandatory libraries or platforms)
- **Restricted Technologies**: (Excluded libraries, legacy codebases)
- **Mandatory Integrations**: (Legacy databases, external APIs)

# E. Compliance Constraints
- **Applicable Regulations**: (GDPR, PCI-DSS, HIPAA, etc.)
- **Security Requirements**: (Audit logging, MFA policies)
- **Data Restrictions**: (Locality, PII residency requirements)

# F. Business Constraints
- **Market Restrictions**: (Geographic availability, languages)
- **Pricing Constraints**: (Tiered subscription limits, transaction fees)
- **Brand & Strategic Constraints**: (Design guidelines, corporate alignments)

# G. Constraint Severity Table
Output a clean Markdown/unicode table mapping major constraints and their severity (High / Medium / Low).
Example format:
| Constraint Type | Value | Severity |
| --- | --- | --- |
| Budget | $150,000 | High |

Do not output anything outside the Markdown report.
"""

SUCCESS_METRICS_PROMPT = """You are a Head of Product Quality. Your goal is to generate a Success Metrics Framework in Markdown format.
You MUST output exactly the following 6 sections:

# A. Success Statement
Provide a single, testable success sentence.
Example: "Within 6 months, reduce project planning time from 5 hours to under 30 minutes while maintaining stakeholder approval rates above 80%."

# B. Primary Outcome Metric
Provide the single metric that matters most in a Markdown table.
Example format:
| Metric | Current | Target | Timeframe |
| --- | --- | --- | --- |

# C. Supporting Metrics
Provide a table containing 3 to 7 supporting metrics that explain why the primary metric moves.
Example format:
| Metric | Target |
| --- | --- |

# D. Guardrail Metrics
Provide a table containing guardrail metrics that must not degrade.
Example format:
| Metric | Minimum Acceptable |
| --- | --- |

# E. Measurement Definition
For every metric, provide:
- **Metric Name**
- **Exact Formula**: (e.g. Accepted outputs / Total generated)
- **Data Source**: (e.g. database logs)
- **Measurement Frequency**: (e.g. Weekly)
- **Owner**: (e.g. Product team)

# F. Success Review Criteria
Provide the explicit pass/fail review criteria (e.g., Primary achieved AND 80% supporting achieved AND no guardrail violated).

Do not output anything outside the Markdown report.
"""

ASSUMPTIONS_REGISTER_PROMPT = """You are a Senior Systems Analyst. Your goal is to generate a comprehensive Assumptions Register in Markdown format.
You MUST output exactly the following 8 fields for each assumption inside a structured Markdown table:

# Assumptions Register Table
Generate a clean Markdown table with the following columns:
1. **ID**: (e.g. A-01, A-02)
2. **Assumption Statement**: (Specific and testable hypothesis)
3. **Why We Believe It**: (One or two sentence evidence)
4. **Impact if Wrong**: (High / Medium / Low)
5. **Confidence Level**: (Percentage value, e.g. 70%)
6. **Validation Method**: (How to prove/disprove, e.g. user surveys)
7. **Validation Deadline**: (Milestone or target date)
8. **Current Status**: (Unvalidated / Testing / Validated / Invalidated)

Ensure the table is well-spaced and filled with high-fidelity, logical assumptions relevant to the project context. Do not output anything outside the Markdown report.
"""

DECISION_BRIEF_PROMPT = """You are a Principal Product Director. Your goal is to generate a Decision Brief in Markdown format to resolve one major architectural or product decision for the project.
You MUST output exactly the following 8 sections:

# A. Decision Statement
State the core decision to be made.
Example: "Should the first release generate complete PRDs or only planning outlines?"

# B. Context
Provide 5-10 lines of key facts and constraints driving this decision.

# C. Options
List between 2 to 4 feasible technical options (e.g., Option A, Option B).

# D. Evaluation Table
Provide a weighted evaluation matrix table. Columns: Criteria, Weight, Option A Score, Option B Score, etc.
Example format:
| Criteria | Weight | Option A Score | Option B Score |
| --- | --- | --- | --- |
| User Value | 40% | 8 | 6 |

# E. Key Tradeoffs
Contrast the options highlighting tradeoffs (e.g., Option A: Highest customer value, highest engineering effort vs. Option B: Faster launch, lower differentiation).

# F. Recommendation
Provide a one-paragraph clear recommendation explaining the "why".

# G. Decision Owner
Specify the role/owner (e.g., Head of Product).

# H. Decision Deadline
Specify the target date/milestone.

Do not output anything outside the Markdown report.
"""



class OllamaService:
    def __init__(self) -> None:
        self.client = AsyncClient(host=settings.OLLAMA_BASE_URL, timeout=180.0)
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
            parsed = self._parse_json_response(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # If the model wrapped the list in an object (e.g. {"requirements": [...]}), extract it
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
                return [parsed]
            return []

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
                format="json",
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
                format="json",
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
                format="json",
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
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to parse Ollama response as JSON: %.200s", raw)
            return []

    async def chat_rag(self, system_prompt: str, user_query: str) -> str:
        if not await self.is_available():
            return "Ollama is currently offline/unavailable."
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                options={"temperature": 0.2, "num_predict": 1024},
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.error("Ollama chat RAG failed: %s", exc)
            return "Failed to generate answer from context."

    def _default_clarified(self, original: dict, clarification: str) -> dict:
        return {
            "title": original.get("title", ""),
            "description": clarification,
            "category": original.get("category", "functional"),
            "priority": original.get("priority", "medium"),
            "ambiguous": False,
            "clarification_prompt": None,
        }

    async def generate_document_style(self, system_prompt: str, context: str) -> str:
        if not await self.is_available():
            return "Failed to generate document: Ollama offline."
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                options={"temperature": 0.3, "num_predict": 8192},
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.error("Failed to generate document-style content: %s", exc)
            return f"Error during generation: {exc}"


ollama_service = OllamaService()

