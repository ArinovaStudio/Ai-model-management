import spacy
from transformers import pipeline

class NLP:
    def __init__(self):
        self.model = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        self.nlp = spacy.load("en_core_web_sm")
        
        self.tables = {
            "tasks assignments and work": "task",
            "projects and ongoing status": "project",
            "users and login status": "user",
            "work hours attendance and clock out times": "workhours",
            "completed work history": "workdone",
            "employee salary, payment and payouts": "payoutschedule", 
            "client invoices and payment requests": "paymentrequest",
            "leaves, time off, vacations": "leavereq",
            "support tickets, issues, help": "ticket",
            "meetings, calls, appointments": "meeting",
            "bank details, accounts": "bankdetails"
        }

        self.fast_routes = {
            "payment": "payoutschedule", "salary": "payoutschedule", "payout": "payoutschedule",
            "attendance": "workhours", "login": "workhours", "logout": "workhours", "report": "workhours",
            "leave": "leavereq", "ticket": "ticket", "meeting": "meeting", "invoice": "paymentrequest"
        }

    def get_query_details(self, text):
        txt = text.lower()
        tbl = None

        for k, v in self.fast_routes.items():
            if k in txt:
                tbl = v
                break
        
        if not tbl:
            res = self.model(text, list(self.tables.keys()))
            tbl = self.tables[res['labels'][0]]

        doc = self.nlp(text)
        names = [e.text for e in doc.ents if e.label_ in ['PERSON', 'ORG', 'GPE']]
        
        # [BUG #6 FIXED] - Filter names using capitalized tokens and remove question words
        if not names:
            ignore_words = ["login", "logout", "attendance", "report", "show", "give", "list", "time", "of", "for", "in", "the"]
            question_words = ['what', 'who', 'when', 'how', 'show', 'give', 'list']
            
            candidates = [
                w for w in text.split()
                if w[0].isupper()
                and w.lower() not in ignore_words
                and len(w) > 2
                and w.isalpha()
            ]
            candidates = [w for w in candidates if w.lower() not in question_words]
            
            if candidates:
                names.append(candidates[0])
                    
        # Client name fallback
        if not names and 'client' in txt:
            words = txt.split()
            for i, w in enumerate(words):
                if 'client' in w and i > 0:
                    names.append(words[i-1])

        months = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
        mnt_val = None
        for m in months:
            if m in txt:
                mnt_val = months[m]
                break

        return {
            "table": tbl,
            "target": names[0] if names else None,
            "month": mnt_val,
            "raw": txt
        }
