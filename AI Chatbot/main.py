from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prisma import Prisma
from datetime import datetime
import calendar
import pytz
import uvicorn

from nlp_engine import NLPProcessor

app = FastAPI(title="AI Office Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Prisma()
nlp = NLPProcessor()

class ChatRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()

def basic_reply(text):
    msg = text.lower()

    if "hi" in msg or "hello" in msg:
        return "Hello! How can I help you?"

    if "how are you" in msg:
        return "I'm doing great."

    if "thank" in msg:
        return "You're welcome."

    if "bye" in msg:
        return "Goodbye!"

    return "You can ask about employees, attendance, meetings, tickets or payments."

async def find_user(query):
    users = await db.user.find_many()
    q = query.lower()

    for u in users:
        full = u.name.lower()
        first = full.split()[0]

        if first in q or full in q:
            return u

    return None

def detect_month(query):
    months = {
        "january":1,"february":2,"march":3,"april":4,
        "may":5,"june":6,"july":7,"august":8,
        "september":9,"october":10,"november":11,"december":12
    }

    q = query.lower()

    for m in months:
        if m in q:
            return months[m]

    return datetime.today().month

async def process_query(query):

    q = query.lower()

    if not nlp.is_database_query(query):
        return basic_reply(query)

    # Active employees
    if "active" in q or "online" in q:
        users = await db.user.find_many(where={"isLogin": True})
        if users:
            return "Active employees: " + ", ".join([u.name for u in users])
        return "No employees are currently active."

    # Login time
    if "login" in q and "time" in q:

        user = await find_user(query)
        if not user:
            return "User not found."

        record = await db.workhours.find_first(
            where={"userId": user.id},
            order={"date": "desc"}
        )

        if not record or not record.clockIn:
            return "Login record not found."

        login_time = datetime.strptime(record.clockIn, "%I:%M %p")

        if "usa" in q or "pst" in q:
            ist = pytz.timezone("Asia/Kolkata")
            pst = pytz.timezone("US/Pacific")

            dt = ist.localize(login_time)
            converted = dt.astimezone(pst)

            return f"{user.name} logged in at {converted.strftime('%I:%M %p')} PST"

        return f"{user.name} logged in at {login_time.strftime('%I:%M %p')} IST"

    # Logout
    if "logout" in q:

        user = await find_user(query)
        if not user:
            return "User not found."

        record = await db.workhours.find_first(
            where={"userId": user.id},
            order={"date": "desc"}
        )

        if record and record.clockOut:
            return f"{user.name} logged out at {record.clockOut}"

        return "Logout record not found."

    # Attendance
    if "attendance" in q or "report" in q:

        user = await find_user(query)
        if not user:
            return "User not found."

        month = detect_month(query)
        year = datetime.today().year

        start = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]

        today = datetime.today()
        end_day = today.day if month == today.month else last_day

        records = await db.workhours.find_many(
            where={
                "userId": user.id,
                "date": {
                    "gte": start,
                    "lte": datetime(year, month, end_day)
                }
            }
        )

        present = sorted(list(set([r.date.day for r in records])))
        absent = [d for d in range(1, end_day + 1) if d not in present]

        status = "(Month ongoing)" if month == today.month else ""

        return (
            f"{calendar.month_name[month]} attendance for {user.name} {status}\n"
            f"Present: {present}\nAbsent: {absent}"
        )

    # Meeting
    if "meeting" in q:
        meetings = await db.meeting.find_many(order={"scheduledAt": "asc"})
        if meetings:
            m = meetings[0]
            return f"Next meeting: {m.title} at {m.scheduledAt}"
        return "No meetings scheduled."

    # Tickets
    if "ticket" in q:
        tickets = await db.ticket.find_many()
        return f"There are {len(tickets)} open tickets."

    # Leave
    if "leave" in q:
        user = await find_user(query)
        if not user:
            return "Employee not found."

        leaves = await db.leavereq.find_many(where={"userId": user.id})

        if not leaves:
            return f"{user.name} has no leave history."

        history = [f"{l.startDate.date()} to {l.endDate.date()}" for l in leaves]
        return f"{user.name}'s leave history: {', '.join(history)}"

    # GitHub
    if "github" in q:
        user = await find_user(query)
        if not user:
            return "Employee not found."

        if not user.githubProfile:
            return f"{user.name} has no GitHub profile."

        return f"{user.name}'s GitHub: {user.githubProfile}"

    # Payout
    if "payment" in q or "salary" in q or "payout" in q:

        user = await find_user(query)
        if not user:
            return "Employee not found."

        schedule = await db.payoutschedule.find_unique(
            where={"userId": user.id}
        )

        payouts = await db.payout.find_many(where={"userId": user.id})
        total_paid = sum([float(p.amount) for p in payouts])

        if schedule:
            next_date = schedule.nextPayoutDate.strftime("%d %b %Y") if schedule.nextPayoutDate else "Not set"
            next_amount = schedule.nextAmount or 0

            return (
                f"{user.name}'s payout:\n"
                f"Next Date: {next_date}\n"
                f"Next Amount: ₹{next_amount}\n"
                f"Total Paid: ₹{total_paid}"
            )

        if total_paid > 0:
            return f"{user.name} total payout: ₹{total_paid}"

        return f"No payout records found for {user.name}."

    return "I couldn't find that information."

@app.post("/api/chat")
async def chat(request: ChatRequest):
    reply = await process_query(request.query)
    return {"reply": reply}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    await websocket.accept()

    while True:
        query = await websocket.receive_text()
        response = await process_query(query)
        await websocket.send_text(response)


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)