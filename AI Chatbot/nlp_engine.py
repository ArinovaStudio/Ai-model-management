import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline

class NLPProcessor:
    def __init__(self):
        # 1. Initialize HuggingFace Zero-Shot Classifier (DistilBERT base)
        print("Loading Intent Classifier (HuggingFace)...")
        self.classifier = pipeline("zero-shot-classification", 
                                   model="typeform/distilbert-base-uncased-mnli")
        
        # Define the intents based on your architecture document
        self.intents = [
            "CHECK_LOGIN_STATUS", 
            "LAST_TASK_ASSIGNED", 
            "PROJECT_STATUS", 
            "CLIENT_INTERACTION_HISTORY",
            "CHECK_COMPLETED_WORK"
        ]

        # 2. Initialize spaCy for NER
        print("Loading spaCy NER...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom rule-based matching for specific schema fields (Priority & Status)
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Define vocabulary for your schema enum-like fields
        priorities = ["high", "medium", "low", "critical", "urgent"]
        statuses = ["completed", "in progress", "pending", "to do", "done"]
        
        self.matcher.add("PRIORITY", [self.nlp.make_doc(text) for text in priorities])
        self.matcher.add("STATUS", [self.nlp.make_doc(text) for text in statuses])

    def identify_intent(self, query):
        """Identifies the intent of the admin query."""
        result = self.classifier(query, self.intents)
        return result['labels'][0], result['scores'][0]

    def extract_entities(self, query):
        """Extracts entities to map to Task and workDone schemas."""
        doc = self.nlp(query)
        entities = {
            "assignees": [],     # Maps to Task.assignee or workDone.userId
            "dates": [],         # Maps to dueDate, createdAt
            "priorities": [],    # Maps to Task.priority
            "statuses": []       # Maps to Task.status
        }

        # Extract standard entities (Names and Dates)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["assignees"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities["dates"].append(ent.text)

        # Extract custom entities (Priority and Status)
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
        """Main pipeline combining Intent Classification and NER."""
        intent, confidence = self.identify_intent(query)
        entities = self.extract_entities(query)
        
        return {
            "query": query,
            "intent": intent,
            "confidence": round(confidence, 4),
            "extracted_entities": entities
        }

# ==========================================
# INTERACTIVE TESTING MODE (NO BACKEND NEEDED)
# ==========================================
if __name__ == "__main__":
    print("\n--- Starting NLP Development Environment ---")
    processor = NLPProcessor()
    print("NLP Engine Ready!\n")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nAdmin Query >> ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("Analyzing...")
        result = processor.process_query(user_input)
        intent = result['intent']
        entities = result['extracted_entities']
        
        print(f"\n[NLP RESULTS]")
        print(f"➜ Intent: {intent} (Confidence: {result['confidence']})")
        print(f"➜ Entities: {entities}")
        
        print(f"\n[SIMULATED SCHEMA MAPPING]")
        # Simulate mapping to your Task schema
        if intent == "LAST_TASK_ASSIGNED":
            print("➜ Target Table: Task")
            mock_where = {}
            if entities["assignees"]:
                mock_where["assignee"] = entities["assignees"][0]
            if entities["priorities"]:
                mock_where["priority"] = entities["priorities"][0]
            if entities["statuses"]:
                mock_where["status"] = entities["statuses"][0]
                
            print(f"➜ Mock Prisma Query: db.task.find_many(where={mock_where}, order={{'createdAt': 'desc'}}, take=1)")

        # Simulate mapping to your workDone schema
        elif intent == "CHECK_COMPLETED_WORK":
            print("➜ Target Table: workDone")
            mock_where = {}
            if entities["assignees"]:
                mock_where["completedBy"] = {"name": entities["assignees"][0]} # Relational lookup simulation
            
            print(f"➜ Mock Prisma Query: db.workdone.find_many(where={mock_where})")
            
        elif intent == "PROJECT_STATUS":
             print("➜ Target Table: Task")
             print(f"➜ Mock Prisma Query: db.task.group_by(by=['status'])")