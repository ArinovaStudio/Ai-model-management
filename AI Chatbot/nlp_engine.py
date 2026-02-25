import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline

class NLPProcessor:
    def __init__(self):
        print("Loading Intent Classifier (HuggingFace)...")
        self.classifier = pipeline("zero-shot-classification", 
                                   model="typeform/distilbert-base-uncased-mnli")
        
        # 1. EXPANDED INTENTS
        self.intent_mapping = {
            "checking the latest assigned task": "LAST_TASK_ASSIGNED",
            "checking completed work or finished tasks": "CHECK_COMPLETED_WORK",
            "checking overall project status or progress": "PROJECT_STATUS",
            "checking user login status or if online": "CHECK_LOGIN_STATUS",
            "checking what time an employee logged out": "CHECK_CLOCK_OUT_TIME",
            "getting a figma design link or project asset": "GET_PROJECT_ASSET",
            "checking the daily work summary of an employee": "GET_EMPLOYEE_SUMMARY",
            "checking if project daily summary is updated": "CHECK_PROJECT_SUMMARY",
            "getting an employee's github profile or username": "GET_GITHUB_PROFILE",
            "checking the latest assigned project": "LAST_PROJECT_ASSIGNED",
            "getting a list of currently active or logged in employees": "GET_ACTIVE_EMPLOYEES",
            "getting a list of all active or current projects": "GET_ALL_PROJECTS"
    
        }
        self.intents = list(self.intent_mapping.keys())

        print("Loading spaCy NER...")
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        priorities = ["high", "medium", "low", "critical", "urgent"]
        statuses = ["completed", "in progress", "pending", "to do", "done"]
        
        self.matcher.add("PRIORITY", [self.nlp.make_doc(text) for text in priorities])
        self.matcher.add("STATUS", [self.nlp.make_doc(text) for text in statuses])

    def identify_intent(self, query):
        """Identifies the intent using a hybrid Keyword + AI approach."""
        query_lower = query.lower()
        
        # KEYWORD OVERRIDES
        if "figma" in query_lower or "design" in query_lower or "asset" in query_lower:
            return "GET_PROJECT_ASSET", 1.0
        elif "github" in query_lower:
            return "GET_GITHUB_PROFILE", 1.0
        elif "logged out" in query_lower or "clock out" in query_lower:
            return "CHECK_CLOCK_OUT_TIME", 1.0
        elif "login" in query_lower or "logged in" in query_lower or "online" in query_lower:
            return "CHECK_LOGIN_STATUS", 1.0
        elif "summary" in query_lower and "project" in query_lower:
            return "CHECK_PROJECT_SUMMARY", 1.0
        elif "summary" in query_lower and ("work" in query_lower or "today" in query_lower):
            return "GET_EMPLOYEE_SUMMARY", 1.0
            
        # ---> CRITICAL FIX: 'project' rule MUST be above the generic 'assign' rule! <---
        elif "project" in query_lower and "assign" in query_lower:
            return "LAST_PROJECT_ASSIGNED", 1.0
            
        elif "assign" in query_lower or "last task" in query_lower or "latest task" in query_lower:
            return "LAST_TASK_ASSIGNED", 1.0
            
        elif "completed" in query_lower or "finished" in query_lower:
            return "CHECK_COMPLETED_WORK", 1.0
        elif "status" in query_lower or "progress" in query_lower:
            return "PROJECT_STATUS", 1.0
        elif "active" in query_lower and ("employee" in query_lower or "list" in query_lower):
            return "GET_ACTIVE_EMPLOYEES", 1.0
        elif "project" in query_lower and ("all" in query_lower or "list" in query_lower or "what are" in query_lower):
            return "GET_ALL_PROJECTS", 1.0

        # Fallback to AI
        result = self.classifier(query, self.intents, hypothesis_template="This text is about {}.")
        best_match_english = result['labels'][0]
        confidence = result['scores'][0]
        
        return self.intent_mapping[best_match_english], confidence

    def extract_entities(self, query):
        """Extracts names, dates, priorities, and statuses from the text."""
        doc = self.nlp(query)
        entities = {
            "assignees": [],     
            "dates": [],         
            "priorities": [],    
            "statuses": []       
        }

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                entities["assignees"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities["dates"].append(ent.text)

        matches = self.matcher(doc)
        for match_id, start, end in matches:
            rule_id = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            if rule_id == "PRIORITY":
                entities["priorities"].append(span.text)
            elif rule_id == "STATUS":
                entities["statuses"].append(span.text)

        return entities

    def process_query(self, query):
        intent, confidence = self.identify_intent(query)
        entities = self.extract_entities(query)
        
        return {
            "query": query,
            "intent": intent,
            "confidence": round(confidence, 4),
            "extracted_entities": entities
        }
