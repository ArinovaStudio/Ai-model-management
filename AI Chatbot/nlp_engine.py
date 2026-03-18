import spacy


class NLPProcessor:

    def __init__(self):

        print("Loading NLP model...")
        self.nlp = spacy.load("en_core_web_sm")

        # words that indicate database queries
        self.db_keywords = [
            "login", "logout", "attendance", "report",
            "active", "employee", "online",
            "ticket", "meeting", "leave",
            "performance", "score",
            "payment", "salary", "payout",
            "github", "task", "project"
        ]

    def is_database_query(self, text):

        q = text.lower()

        for word in self.db_keywords:
            if word in q:
                return True

        return False

    def extract_name(self, text):

        doc = self.nlp(text)

        # Try NER
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text

        # fallback method
        words = text.split()

        ignore = [
            "login", "logout", "attendance", "report",
            "show", "give", "list", "time",
            "of", "for", "in"
        ]

        for w in words:
            if w.lower() not in ignore and w[0].isalpha():
                return w

        return None