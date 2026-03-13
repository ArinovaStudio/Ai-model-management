from fastapi import FastAPI
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

@app.post("/api/chat")
async def chat(request: ChatRequest):

    query = request.query
    q = query.lower()

    if not nlp.is_database_query(query):
        return {"reply": basic_reply(query)}

   
    if "active" in q or "online" in q:

        users = await db.user.find_many(where={"isLogin": True})

        if users:
            names = ", ".join([u.name for u in users])
            return {"reply": f"Active employees: {names}"}

        return {"reply": "No employees are currently active."}


    if "login" in q and "time" in q:

        user = await find_user(query)

        if not user:
            return {"reply": "User not found."}

        record = await db.workhours.find_first(
            where={"userId": user.id},
            order={"date": "desc"}
        )

        if not record or not record.clockIn:
            return {"reply": "Login record not found."}

        login_str = record.clockIn
        login_time = datetime.strptime(login_str, "%I:%M %p")

        if "usa" in q or "pst" in q:

            ist = pytz.timezone("Asia/Kolkata")
            pst = pytz.timezone("US/Pacific")

            dt = ist.localize(login_time)
            converted = dt.astimezone(pst)

            return {"reply": f"{user.name} logged in at {converted.strftime('%I:%M %p')} PST"}

        return {"reply": f"{user.name} logged in at {login_time.strftime('%I:%M %p')} IST"}

    if "logout" in q:

        user = await find_user(query)

        if not user:
            return {"reply": "User not found."}

        record = await db.workhours.find_first(
            where={"userId": user.id},
            order={"date": "desc"}
        )

        if record and record.clockOut:
            return {"reply": f"{user.name} logged out at {record.clockOut}"}

        return {"reply": "Logout record not found."}

    if "attendance" in q or "report" in q:

        user = await find_user(query)

        if not user:
            return {"reply": "User not found."}

        month = detect_month(query)
        year = datetime.today().year

        start = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]

        today = datetime.today()

        if month == today.month:
            end_day = today.day
        else:
            end_day = last_day

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

        status = "(Month still ongoing)" if month == today.month else ""

        return {
            "reply":
            f"{calendar.month_name[month]} attendance report for {user.name} {status}\n"
            f"Present days: {present}\n"
            f"Absent days: {absent}"
        }

    if "meeting" in q:

        meetings = await db.meeting.find_many(order={"scheduledAt": "asc"})

        if meetings:
            m = meetings[0]
            return {"reply": f"Next meeting: {m.title} at {m.scheduledAt}"}

        return {"reply": "No meetings scheduled."}

    if "ticket" in q:

        tickets = await db.ticket.find_many()

        return {"reply": f"There are {len(tickets)} open tickets."}

    if "leave" in q:

        user = await find_user(query)

        if not user:
            return {"reply": "Employee not found."}

        leaves = await db.leavereq.find_many(where={"userId": user.id})

        if not leaves:
            return {"reply": f"{user.name} has no leave history."}

        history = [
            f"{l.startDate.date()} to {l.endDate.date()}"
            for l in leaves
        ]

        return {"reply": f"{user.name}'s leave history: {', '.join(history)}"}


    if "github" in q:

        user = await find_user(query)

        if not user:
            return {"reply": "Employee not found."}

        if not user.githubProfile:
            return {"reply": f"{user.name} has not added a GitHub profile."}

        return {"reply": f"{user.name}'s GitHub profile: {user.githubProfile}"}

    if "payment" in q or "salary" in q or "payout" in q:

        user = await find_user(query)

        if not user:
            return {"reply": "Employee not found."}

        payouts = await db.payout.find_many(where={"userId": user.id})
        requests = await db.paymentrequest.find_many(where={"requestedToId": user.id})

        total = 0

        for p in payouts:
            total += float(p.amount)

        for r in requests:
            total += float(r.amount)

        if total == 0:
            return {"reply": f"No payout records found for {user.name}."}

        return {"reply": f"{user.name} has received total payments of ₹{total}"}


    return {"reply": "I couldn't find that information."}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)