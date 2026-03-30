import json, calendar, pytz
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp_engine import NLP
from prisma import Prisma
import uvicorn
from thefuzz import process
from deep_translator import GoogleTranslator

app = FastAPI(title="Voice Assistant AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

nlp = NLP()
db = Prisma()

@app.on_event("startup")
async def start_db(): await db.connect()

@app.on_event("shutdown")
async def stop_db(): await db.disconnect()

# ==========================================
# WEB UI FOR SPEAK CHATBOT
# ==========================================
html_content = """
<!DOCTYPE html>
<html>
<head><style>body{font-family:sans-serif; background:#f4f7f6; display:flex; justify-content:center; padding-top:50px;} .box{width:600px; background:white; padding:20px; border-radius:10px; box-shadow:0 4px 15px rgba(0,0,0,0.1);} #log{height:400px; overflow-y:auto; border:1px solid #ddd; padding:15px; margin-bottom:15px; background:#f9fbfc;} input{flex-grow:1; padding:10px; border:1px solid #ccc; border-radius:4px;} button{padding:10px 20px; background:#007bff; color:white; border:none; cursor:pointer;} #mic{background:#28a745; font-size:18px; border-radius:50%;} .rec{background:#dc3545 !important;}</style></head>
<body>
    <div class="box">
        <h2>🎙️ Voice AI Hub</h2>
        <div id="log"></div>
        <div style="display:flex; gap:10px;">
            <button id="mic" onclick="startMic()">🎤</button>
            <input type="text" id="msg" placeholder="Speak or type..."/>
            <button onclick="sendMsg()">Send</button>
        </div>
    </div>
    <script>
        let ws = new WebSocket("ws://localhost:8000/ws/chat");
        let log = document.getElementById('log'), micBtn = document.getElementById('mic'), inp = document.getElementById('msg');
        ws.onmessage = e => { log.innerHTML += "<p>🤖 " + JSON.parse(e.data).reply + "</p>"; log.scrollTop = log.scrollHeight; };
        function sendMsg() { if(!inp.value) return; log.innerHTML += "<p>👤 " + inp.value + "</p>"; ws.send(JSON.stringify({query: inp.value})); inp.value = ''; }
        function startMic() {
            let rec = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            rec.lang = "en-IN";
            rec.onstart = () => micBtn.classList.add("rec");
            rec.onresult = e => { inp.value = e.results[0][0].transcript; sendMsg(); };
            rec.onend = () => micBtn.classList.remove("rec");
            rec.start();
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get_ui(): return HTMLResponse(html_content)

# ==========================================
# CORE PROCESSING LOGIC
# ==========================================
async def handle_query(query: str, ctx: dict):
    try: eq = GoogleTranslator(source='auto', target='en').translate(query)
    except: eq = query

    parsed = nlp.get_query_details(eq)
    tbl = parsed['table']
    tgt = parsed['target']
    mnt = parsed['month']
    raw_q = parsed['raw']

    if tgt: ctx['last_tgt'] = tgt
    elif ctx.get('last_tgt') and any(p in eq.lower() for p in ["he", "she", "him", "her", "their"]):
        tgt = ctx['last_tgt']

    users = await db.user.find_many()
    u_names = [u.name for u in users if u.name]
    
    if tgt and u_names:
        m, s = process.extractOne(tgt, u_names)
        if s > 70: tgt = m

    usr = None
    if tgt:
        try: usr = await db.user.find_first(where={"name": {"contains": tgt, "mode": "insensitive"}})
        except: pass

    # --- TEAMMATE'S SPECIFIC LOGIC BLOCKS (Catches Math & Formatting) ---
    
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
        except: return f"{usr.name} logged in at {rec.clockIn}"

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

    # 🟢 THE FIX: BULLETPROOF PAYOUT LOGIC
    if tbl == "payoutschedule" and usr:
        try:
            # Removed order={"createdAt": "desc"} so it doesn't crash on schemas without that column!
            sched = await db.payoutschedule.find_first(where={"userId": usr.id})
            
            # Safely calculate total past payouts
            hist = []
            try: hist = await db.payout.find_many(where={"userId": usr.id})
            except: pass
            
            tot = 0
            for p in hist:
                if hasattr(p, 'amount') and p.amount:
                    try: tot += float(p.amount)
                    except: pass

            if sched:
                # Safely format the date, even if it's stored as a string instead of a datetime object
                nd = "Not set"
                if hasattr(sched, 'nextPayoutDate') and sched.nextPayoutDate:
                    try: nd = sched.nextPayoutDate.strftime("%d %b %Y")
                    except: nd = str(sched.nextPayoutDate).split(' ')[0]
                
                na = getattr(sched, 'nextAmount', 0)
                
                # Formatted exactly like your requested screenshot!
                return f"{usr.name.upper()}'s payout: Next Date: {nd} Next Amount: ₹{na} Total Paid: ₹{tot}"
            
            elif tot > 0: 
                return f"{usr.name.upper()}'s payout: Total Paid: ₹{tot}"
            else: 
                return f"No payout records found for {usr.name}."
                
        except Exception as e:
            print(f"Payout logic error: {e}")
            return f"Error reading payout for {usr.name}."

    # --- ADARSH'S DYNAMIC ARCHITECTURE FALLBACK ---
    allowed = ["task", "project", "user", "workhours", "workdone", "paymentrequest", "payoutschedule", "leavereq", "ticket", "meeting", "bankdetails", "clientfeedback"]
    if tbl not in allowed: return "I'm not sure which data to fetch for this."

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
            
        if not data: return f"The {tbl} table is empty."

        res_strs = []
        for r in data:
            d = r.__dict__ if hasattr(r, '__dict__') else {}
            c_f = {k: v for k, v in d.items() if not k.startswith('_') and v is not None and k not in ['id', 'createdAt', 'updatedAt', 'userId']}
            res_strs.append("• " + ", ".join([f"{k}: {v}" for k, v in list(c_f.items())[:5]]))

        return " | ".join(res_strs)

    except Exception as e:
        print(f"err: {e}")
        return "Internal fetch error."


# ==========================================
# ENDPOINTS
# ==========================================
class Req(BaseModel): query: str
mem = {"last_tgt": None}

@app.post("/api/chat")
async def chat_api(req: Req):
    return {"reply": await handle_query(req.query, mem)}

@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await ws.accept()
    ws_mem = {"last_tgt": None}
    try:
        while True:
            d = json.loads(await ws.receive_text())
            if "query" in d: await ws.send_json({"reply": await handle_query(d["query"], ws_mem)})
    except: pass

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)