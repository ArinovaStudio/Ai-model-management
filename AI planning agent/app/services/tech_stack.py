import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

TECHNOLOGY_REGISTRY: dict[str, list[dict]] = {
    "backend": [
        {"name": "Python", "aliases": ["python", "python3"], "dependencies": []},
        {"name": "Django", "aliases": ["django"], "dependencies": ["Python"]},
        {"name": "FastAPI", "aliases": ["fastapi", "fast api"], "dependencies": ["Python"]},
        {"name": "Flask", "aliases": ["flask"], "dependencies": ["Python"]},
        {"name": "Node.js", "aliases": ["node", "node.js", "nodejs", "node js"], "dependencies": []},
        {"name": "Express", "aliases": ["express", "express.js", "expressjs"], "dependencies": ["Node.js"]},
        {"name": "NestJS", "aliases": ["nest", "nestjs", "nest.js"], "dependencies": ["Node.js"]},
        {"name": "Go", "aliases": ["golang", "go lang"], "dependencies": []},
        {"name": "Rust", "aliases": ["rust", "rustlang"], "dependencies": []},
        {"name": "Java", "aliases": ["java", "java 8", "java 11", "java 17"], "dependencies": []},
        {"name": "Spring Boot", "aliases": ["spring", "spring boot", "springboot"], "dependencies": ["Java"]},
        {"name": "Kotlin", "aliases": ["kotlin"], "dependencies": ["Java"]},
        {"name": "C#", "aliases": ["c#", "csharp", "c sharp", ".net"], "dependencies": []},
        {"name": ".NET Core", "aliases": [".net core", "dotnet core", "asp.net"], "dependencies": ["C#"]},
        {"name": "Ruby on Rails", "aliases": ["ruby", "rails", "ruby on rails"], "dependencies": []},
        {"name": "PHP", "aliases": ["php", "php 8"], "dependencies": []},
        {"name": "Laravel", "aliases": ["laravel"], "dependencies": ["PHP"]},
        {"name": "PostgreSQL", "aliases": ["postgresql", "postgres", "psql"], "dependencies": []},
        {"name": "MongoDB", "aliases": ["mongodb", "mongo", "nosql"], "dependencies": []},
        {"name": "Redis", "aliases": ["redis"], "dependencies": []},
        {"name": "MySQL", "aliases": ["mysql", "mariadb"], "dependencies": []},
        {"name": "GraphQL", "aliases": ["graphql", "graph ql"], "dependencies": []},
        {"name": "gRPC", "aliases": ["grpc"], "dependencies": []},
        {"name": "Kubernetes", "aliases": ["kubernetes", "k8s"], "dependencies": []},
        {"name": "Docker", "aliases": ["docker", "container"], "dependencies": []},
        {"name": "AWS", "aliases": ["aws", "amazon web services", "ec2", "s3", "lambda"], "dependencies": []},
        {"name": "Azure", "aliases": ["azure", "microsoft azure"], "dependencies": []},
        {"name": "GCP", "aliases": ["gcp", "google cloud", "google cloud platform"], "dependencies": []},
    ],
    "frontend": [
        {"name": "React", "aliases": ["react", "reactjs", "react.js"], "dependencies": ["JavaScript", "TypeScript"]},
        {"name": "Vue", "aliases": ["vue", "vuejs", "vue.js", "vue 3"], "dependencies": ["JavaScript", "TypeScript"]},
        {"name": "Angular", "aliases": ["angular", "angularjs", "angular 2"], "dependencies": ["TypeScript"]},
        {"name": "Svelte", "aliases": ["svelte", "sveltekit"], "dependencies": ["JavaScript", "TypeScript"]},
        {"name": "Next.js", "aliases": ["next", "nextjs", "next.js"], "dependencies": ["React"]},
        {"name": "Nuxt.js", "aliases": ["nuxt", "nuxtjs", "nuxt.js"], "dependencies": ["Vue"]},
        {"name": "TypeScript", "aliases": ["typescript", "ts"], "dependencies": ["JavaScript"]},
        {"name": "JavaScript", "aliases": ["javascript", "js", "ecmascript", "es6"], "dependencies": []},
        {"name": "Tailwind CSS", "aliases": ["tailwind", "tailwindcss", "tailwind css"], "dependencies": []},
        {"name": "Bootstrap", "aliases": ["bootstrap", "bootstrap 5"], "dependencies": []},
        {"name": "Material UI", "aliases": ["material ui", "mui", "material-ui"], "dependencies": ["React"]},
        {"name": "Redux", "aliases": ["redux", "redux toolkit"], "dependencies": ["React"]},
        {"name": "Webpack", "aliases": ["webpack"], "dependencies": []},
        {"name": "Vite", "aliases": ["vite"], "dependencies": []},
    ],
    "mobile": [
        {"name": "React Native", "aliases": ["react native", "reactnative"], "dependencies": ["React"]},
        {"name": "Flutter", "aliases": ["flutter", "dart"], "dependencies": []},
        {"name": "Swift", "aliases": ["swift", "swiftui"], "dependencies": []},
        {"name": "Kotlin Multiplatform", "aliases": ["kotlin multiplatform", "kmp"], "dependencies": ["Kotlin"]},
        {"name": "Android", "aliases": ["android", "android sdk"], "dependencies": ["Java", "Kotlin"]},
        {"name": "iOS", "aliases": ["ios", "iphone", "ipad"], "dependencies": ["Swift"]},
        {"name": "Ionic", "aliases": ["ionic", "ionic framework"], "dependencies": ["Angular", "React"]},
        {"name": "Xamarin", "aliases": ["xamarin", "xamarin forms"], "dependencies": ["C#"]},
    ],
    "qa": [
        {"name": "Jest", "aliases": ["jest"], "dependencies": ["JavaScript", "TypeScript"]},
        {"name": "Pytest", "aliases": ["pytest", "py test"], "dependencies": ["Python"]},
        {"name": "Mocha", "aliases": ["mocha", "mocha.js"], "dependencies": ["Node.js"]},
        {"name": "Cypress", "aliases": ["cypress", "cypress.io"], "dependencies": ["Node.js"]},
        {"name": "Selenium", "aliases": ["selenium"], "dependencies": []},
        {"name": "Playwright", "aliases": ["playwright"], "dependencies": ["Node.js"]},
        {"name": "JUnit", "aliases": ["junit", "junit 5"], "dependencies": ["Java"]},
        {"name": "TestNG", "aliases": ["testng"], "dependencies": ["Java"]},
        {"name": "SonarQube", "aliases": ["sonar", "sonarqube"], "dependencies": []},
        {"name": "ESLint", "aliases": ["eslint"], "dependencies": ["Node.js"]},
    ],
    "devops": [
        {"name": "Docker", "aliases": ["docker", "dockerfile"], "dependencies": []},
        {"name": "Kubernetes", "aliases": ["kubernetes", "k8s", "kube"], "dependencies": ["Docker"]},
        {"name": "Jenkins", "aliases": ["jenkins", "jenkins ci"], "dependencies": []},
        {"name": "GitHub Actions", "aliases": ["github actions", "gh actions"], "dependencies": []},
        {"name": "GitLab CI", "aliases": ["gitlab ci", "gitlab-ci"], "dependencies": []},
        {"name": "CircleCI", "aliases": ["circleci", "circle ci"], "dependencies": []},
        {"name": "Terraform", "aliases": ["terraform", "iac"], "dependencies": []},
        {"name": "Ansible", "aliases": ["ansible"], "dependencies": []},
        {"name": "Helm", "aliases": ["helm", "helm chart"], "dependencies": ["Kubernetes"]},
        {"name": "Prometheus", "aliases": ["prometheus"], "dependencies": []},
        {"name": "Grafana", "aliases": ["grafana"], "dependencies": []},
        {"name": "ELK Stack", "aliases": ["elk", "elasticsearch", "logstash", "kibana"], "dependencies": []},
    ],
    "uiux": [
        {"name": "Figma", "aliases": ["figma"], "dependencies": []},
        {"name": "Sketch", "aliases": ["sketch"], "dependencies": []},
        {"name": "Adobe XD", "aliases": ["adobe xd", "xd", "adobexd"], "dependencies": []},
        {"name": "InVision", "aliases": ["invision"], "dependencies": []},
        {"name": "Zeplin", "aliases": ["zeplin"], "dependencies": []},
        {"name": "Storybook", "aliases": ["storybook"], "dependencies": ["React", "Vue", "Angular"]},
    ],
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s.#]", "", text).lower().strip()


def detect_technologies(texts: list[str]) -> list[dict]:
    detected: dict[str, dict] = {}
    combined = " ".join(texts)
    normalized = _normalize(combined)

    for category, techs in TECHNOLOGY_REGISTRY.items():
        for tech in techs:
            for alias in tech["aliases"]:
                alias_norm = _normalize(alias)
                if alias_norm and alias_norm in normalized:
                    key = tech["name"].lower()
                    if key not in detected:
                        detected[key] = {
                            "name": tech["name"],
                            "category": category,
                            "version": None,
                            "is_detected": True,
                            "dependencies": list(tech["dependencies"]),
                        }
                    break

    return list(detected.values())


def resolve_dependencies(technologies: list[dict]) -> list[dict]:
    tech_names = {t["name"].lower() for t in technologies}

    for tech in technologies:
        resolved_deps: list[str] = []
        for dep in (tech.get("dependencies") or []):
            if dep.lower() in tech_names:
                resolved_deps.append(dep)
            else:
                logger.debug(
                    "Dependency %s for %s not detected in project; adding as inferred",
                    dep,
                    tech["name"],
                )
                resolved_deps.append(dep)
        tech["dependencies"] = resolved_deps

    return technologies


def detect_technologies_from_all(
    project_description: Optional[str],
    plan_doc_texts: list[str],
    doc_texts: list[str],
) -> list[dict]:
    all_texts: list[str] = []
    if project_description:
        all_texts.append(project_description)
    all_texts.extend(plan_doc_texts)
    all_texts.extend(doc_texts)

    detected = detect_technologies(all_texts)
    resolved = resolve_dependencies(detected)
    return resolved
