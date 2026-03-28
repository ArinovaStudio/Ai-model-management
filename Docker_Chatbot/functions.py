import calendar
import pytz
from datetime import datetime
from thefuzz import process
from deep_translator import GoogleTranslator

# Import from your new utils file
from utils import is_hinglish

async def handle_query(query: str, ctx: dict, db, nlp) -> str:
    """Processes the query using specific logic first, then falls back to dynamic execution."""
    
    try: 
        eq = GoogleTranslator(source='auto', target='en').translate(query)
    except: 
        eq = query

    parsed = nlp.get_query_details(eq)
    tbl = parsed['table']
    tgt = parsed['target']
    mnt = parsed['month']
    raw_q = parsed['raw']

    # --- Short Term Memory ---
    if tgt: 
        ctx['last_tgt'] = tgt
    elif ctx.get('last_tgt') and any(p in eq.lower() for p in ["he", "she", "him", "her", "their"]):
        tgt = ctx['last_tgt']

    # --- Fuzzy Matching for User Names ---
    users = await db.user.find_many()
    u_names = [u.name for u in users if u.name]
    
    if tgt and u_names:
        m, s = process.extractOne(tgt, u_names)
        if s > 70: 
            tgt = m

    usr = None
    if tgt:
        try: 
            usr = await db.user.find_first(where={"name": {"contains": tgt, "mode": "insensitive"}})
        except: 
            pass

    if "active" in raw_q or "online" in raw_q:
        actv = await db.user.find_many(where={"isLogin": True})
        return "Active employees: " + ", ".join([u.name for u in actv]) if actv else "No employees are currently active."

    if tbl == "workhours" and "login" in raw_q and "time" in raw_q:
        if not usr: return "User not found."
        rec = await db.workhours.find_first(where={"userId": usr.id}, order={"date": "desc"})
        if not rec or not rec.clockIn: return "Login record not found."
        
        try:
            l_time = datetime.strptime(rec.clockIn, "%I:%M %p")
            if "usa" in raw_q or "pst" in raw_q:
                dt_conv = pytz.timezone("Asia/Kolkata").localize(l_time).astimezone(pytz.timezone("US/Pacific"))
                return f"{usr.name} logged in at {dt_conv.strftime('%I:%M %p')} PST"
            return f"{usr.name} logged in at {l_time.strftime('%I:%M %p')} IST"
        except: 
            return f"{usr.name} logged in at {rec.clockIn}"

    if tbl == "workhours" and ("attendance" in raw_q or "report" in raw_q):
        if not usr: return "Please specify an employee for the report."
        cur_m = mnt if mnt else datetime.today().month
        yr = datetime.today().year
        end_d = datetime.today().day if cur_m == datetime.today().month else calendar.monthrange(yr, cur_m)[1]

        recs = await db.workhours.find_many(where={
            "userId": usr.id,
            "date": {"gte": datetime(yr, cur_m, 1), "lte": datetime(yr, cur_m, end_d)}
        })
        
        prs = sorted(list(set([r.date.day for r in recs if r.date])))
        absnt = [d for d in range(1, end_d + 1) if d not in prs]
        status = "(Ongoing)" if cur_m == datetime.today().month else ""
        return f"{calendar.month_name[cur_m]} report for {usr.name} {status} -> Present: {prs} | Absent: {absnt}"

    if tbl == "payoutschedule" and usr:
        try:
            sched = await db.payoutschedule.find_first(where={"userId": usr.id})
            hist = []
            try: hist = await db.payout.find_many(where={"userId": usr.id})
            except: pass
            
            tot = sum([float(p.amount) for p in hist if hasattr(p, 'amount') and p.amount])

            if sched:
                nd = "Not set"
                if hasattr(sched, 'nextPayoutDate') and sched.nextPayoutDate:
                    try: nd = sched.nextPayoutDate.strftime("%d %b %Y")
                    except: nd = str(sched.nextPayoutDate).split(' ')[0]
                
                na = getattr(sched, 'nextAmount', 0)
                return f"{usr.name.upper()}'s payout: Next Date: {nd} Next Amount: ₹{na} Total Paid: ₹{tot}"
            elif tot > 0: 
                return f"{usr.name.upper()}'s payout: Total Paid: ₹{tot}"
            else: 
                return f"No payout records found for {usr.name}."
        except Exception as e:
            print(f"Payout logic error: {e}")

    allowed = ["task", "project", "user", "workhours", "workdone", "paymentrequest", "payoutschedule", "leavereq", "ticket", "meeting", "bankdetails", "clientfeedback"]
    if tbl not in allowed: 
        return "I'm not sure which data to fetch for this."

    try:
        db_mdl = getattr(db, tbl)
        data = None
        query_executed = False
        
        if usr:
            for fk in ["userId", "employeeId", "user_id", "clientId"]:
                try:
                    data = await db_mdl.find_many(where={fk: usr.id}, take=2, order={"createdAt": "desc"})
                    query_executed = True
                    if data: break
                except: pass

        if not query_executed or not data:
            for c in ["assignee", "clientName", "name"]:
                try:
                    data = await db_mdl.find_many(where={c: {"contains": tgt, "mode": "insensitive"}}, take=2, order={"createdAt": "desc"})
                    query_executed = True
                    if data: break
                except: pass

        if query_executed and not data:
            return f"No records found in {tbl} for {tgt}."

        if not data:
            try: data = await db_mdl.find_many(take=2, order={"createdAt": "desc"})
            except: data = await db_mdl.find_many(take=2)
            
        if not data: 
            return f"The {tbl} table is empty."

        res_strs = []
        for r in data:
            d = r.__dict__ if hasattr(r, '__dict__') else {}
            c_f = {k: v for k, v in d.items() if not k.startswith('_') and v is not None and k not in ['id', 'createdAt', 'updatedAt', 'userId']}
            res_strs.append("• " + ", ".join([f"{k}: {v}" for k, v in list(c_f.items())[:5]]))

        return " | ".join(res_strs)

    except Exception as e:
        print(f"Dynamic fetch error: {e}")
        return "Internal fetch error."