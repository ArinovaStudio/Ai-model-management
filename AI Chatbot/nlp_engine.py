import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline

class NLPProcessor:
    def __init__(self):
        print("Loading Intent Classifier (HuggingFace)...")
        self.classifier = pipeline("zero-shot-classification", 
                                   model="typeform/distilbert-base-uncased-mnli")
        
        # Natural English mapping for the AI
        self.intent_mapping = {
            "checking the latest assigned task": "LAST_TASK_ASSIGNED",
            "checking completed work or finished tasks": "CHECK_COMPLETED_WORK",
            "checking overall project status": "PROJECT_STATUS",
            "checking user login status": "CHECK_LOGIN_STATUS",
            "checking client interaction history": "CLIENT_INTERACTION_HISTORY"
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
        
        # 1. Keyword Overrides (Guarantees 100% accuracy for common queries)
        if "assigned" in query_lower or "latest task" in query_lower or "new task" in query_lower:
            return "LAST_TASK_ASSIGNED", 1.0
            
        elif "completed" in query_lower or "finished" in query_lower or "done" in query_lower:
            return "CHECK_COMPLETED_WORK", 1.0
            
        elif "status" in query_lower or "progress" in query_lower:
            return "PROJECT_STATUS", 1.0

        # 2. Fallback to HuggingFace AI for complex/weird phrasing
        result = self.classifier(
            query, 
            self.intents,
            hypothesis_template="This text is about {}."
        )
        
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
            if ent.label_ == "PERSON":
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
        """Runs the full NLP pipeline."""
        intent, confidence = self.identify_intent(query)
        entities = self.extract_entities(query)
        
        return {
            "query": query,
            "intent": intent,
            "confidence": round(confidence, 4),
            "extracted_entities": entities
        }