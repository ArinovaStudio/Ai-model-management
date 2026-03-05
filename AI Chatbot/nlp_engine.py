import spacy
from transformers import pipeline


class NLPProcessor:
    def __init__(self):
        print("Loading Intent Classifier...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli"
        )

        self.intent_mapping = {
            "checking active employees": "GET_ACTIVE_EMPLOYEES",
            "checking project status": "PROJECT_STATUS",
            "checking login status": "CHECK_LOGIN_STATUS",
            "checking logout time": "CHECK_CLOCK_OUT_TIME",
            "checking completed work": "CHECK_COMPLETED_WORK",
            "getting github profile": "GET_GITHUB_PROFILE",
            "checking last assigned task": "LAST_TASK_ASSIGNED",
            "checking last assigned project": "LAST_PROJECT_ASSIGNED",
            "getting all projects": "GET_ALL_PROJECTS",
            "checking who is working on project": "GET_PROJECT_TEAM"
        }

        self.intents = list(self.intent_mapping.keys())
        self.nlp = spacy.load("en_core_web_sm")

    # -------------------------------------------------------
    # INTENT DETECTION (HINGLISH + ENGLISH)
    # -------------------------------------------------------
    def identify_intent(self, query):
        query_lower = query.lower()

        # -------------------------------
        # ACTIVE EMPLOYEES (Hinglish)
        # -------------------------------
        if (
            ("active" in query_lower and "employee" in query_lower)
            or ("kitne" in query_lower and "active" in query_lower)
            or ("aaj" in query_lower and "active" in query_lower)
            or ("aaj" in query_lower and "login" in query_lower)
            or ("kis" in query_lower and "login" in query_lower)
            or ("kaun" in query_lower and "login" in query_lower)
        ):
            return "GET_ACTIVE_EMPLOYEES", 1.0

        # -------------------------------
        # SPECIFIC LOGIN CHECK
        # -------------------------------
        elif "login" in query_lower:
            return "CHECK_LOGIN_STATUS", 1.0

        # -------------------------------
        # LOGOUT
        # -------------------------------
        elif "logout" in query_lower:
            return "CHECK_CLOCK_OUT_TIME", 1.0

        # -------------------------------
        # PROJECT STATUS
        # -------------------------------
        elif "status" in query_lower or "progress" in query_lower:
            return "PROJECT_STATUS", 1.0

        # -------------------------------
        # COMPLETED WORK
        # -------------------------------
        elif "completed" in query_lower or "complete" in query_lower:
            return "CHECK_COMPLETED_WORK", 1.0

        # -------------------------------
        # GITHUB
        # -------------------------------
        elif "github" in query_lower:
            return "GET_GITHUB_PROFILE", 1.0

        # -------------------------------
        # LAST TASK
        # -------------------------------
        elif "last task" in query_lower:
            return "LAST_TASK_ASSIGNED", 1.0

        # -------------------------------
        # LAST PROJECT
        # -------------------------------
        elif "last project" in query_lower:
            return "LAST_PROJECT_ASSIGNED", 1.0

        # -------------------------------
        # ALL PROJECTS
        # -------------------------------
        elif "all projects" in query_lower:
            return "GET_ALL_PROJECTS", 1.0

        # -------------------------------
        # PROJECT TEAM
        # -------------------------------
        elif "working on" in query_lower or "kaun kaam kar raha" in query_lower:
            return "GET_PROJECT_TEAM", 1.0

        # -------------------------------
        # AI Fallback
        # -------------------------------
        result = self.classifier(
            query,
            self.intents,
            hypothesis_template="This text is about {}."
        )

        best_match = result["labels"][0]
        confidence = result["scores"][0]

        return self.intent_mapping[best_match], confidence

    # -------------------------------------------------------
    def extract_entities(self, query):
        doc = self.nlp(query)

        entities = {"assignees": []}

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "PRODUCT"]:
                entities["assignees"].append(ent.text)

        return entities

    # -------------------------------------------------------
    def process_query(self, query):
        intent, confidence = self.identify_intent(query)
        entities = self.extract_entities(query)

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "extracted_entities": entities
        }