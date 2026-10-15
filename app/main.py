"""
Employee Attendance & Analytics API
Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded; real environment variables win)
"""
import calendar
import os
import re
from contextlib import asynccontextmanager
from datetime import date as date_cls, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
PRESENT_STATUSES = ["PRESENT", "WFH", "ON_DUTY"]
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# --------------------------------------------------------------------------- #
# Startup: indexes (idempotent)
# --------------------------------------------------------------------------- #
def ensure_indexes():
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_emp_code"
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_emp_date"
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)], name="date_desc_emp"
    )
    db.attendance_logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)],
        name="status_date_emp",
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="emp_punch_in"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Helpers: time conversion and rules R1-R5
# --------------------------------------------------------------------------- #
def from_ms(ms: int) -> datetime:
    """Epoch milliseconds -> UTC datetime, cut down to whole seconds (R1)."""
    return datetime.fromtimestamp(ms // 1000, tz=timezone.utc)


def to_ms(dt: datetime) -> int:
    """Datetime from MongoDB -> epoch milliseconds (exact integer maths)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def now_utc_seconds() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def now_utc_ms() -> datetime:
    n = datetime.now(timezone.utc)
    return n.replace(microsecond=n.microsecond // 1000 * 1000)


def parse_hhmm(value: str) -> time:
    h, m = map(int, value.split(":"))
    return time(h, m)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return parse_hhmm(shift_end) <= parse_hhmm(shift_start)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """The IST date a punch-in belongs to (R1), including overnight shifts."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(shift_start, shift_end) and local.time() < parse_hhmm(shift_end):
        day = day - timedelta(days=1)
    return day.isoformat()


def shift_start_dt(date_str: str, shift_start: str) -> datetime:
    day = datetime.fromisoformat(date_str).date()
    return datetime.combine(day, parse_hhmm(shift_start), tzinfo=IST)


def shift_end_dt(date_str: str, shift_start: str, shift_end: str) -> datetime:
    day = datetime.fromisoformat(date_str).date()
    if is_overnight(shift_start, shift_end):
        day = day + timedelta(days=1)
    return datetime.combine(day, parse_hhmm(shift_end), tzinfo=IST)


def round_half_up(value, places: int = 2) -> float:
    q = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str) -> int:
    """R2: late only if strictly more than 10:00 after shift start."""
    seconds = int((punch_in - shift_start_dt(date_str, shift_start)).total_seconds())
    return seconds // 60 if seconds > 600 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, rounded to 2 decimals half-up."""
    seconds = int((punch_out - punch_in).total_seconds())
    hours = Decimal(seconds) / Decimal(3600)
    return float(hours.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(punch_out: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R3: whole minutes after shift end, only if at least 30."""
    seconds = int((punch_out - shift_end_dt(date_str, shift_start, shift_end)).total_seconds())
    minutes = seconds // 60
    return minutes if minutes >= 30 else 0


def compute_half_day(work_hours: float) -> bool:
    """R5: rounded work hours below 4.50."""
    return work_hours < 4.5


# --------------------------------------------------------------------------- #
# Helpers: validation and output shaping
# --------------------------------------------------------------------------- #
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_date(value: str) -> str:
    if not DATE_RE.match(value):
        raise ValueError("must be YYYY-MM-DD")
    date_cls.fromisoformat(value)
    return value


def date_param(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    try:
        return check_date(value)
    except ValueError:
        raise HTTPException(422, f"{name} must be a valid YYYY-MM-DD date")


def validate_month(month: str) -> str:
    if not MONTH_RE.match(month):
        raise HTTPException(422, "month must be YYYY-MM")
    return month


def employee_out(d: dict) -> dict:
    return {
        "emp_code": d["emp_code"],
        "name": d["name"],
        "email": d["email"],
        "department": d["department"],
        "shift_start": d["shift_start"],
        "shift_end": d["shift_end"],
        "joined_on": d["joined_on"],
        "created_at": to_ms(d["created_at"]) if d.get("created_at") else None,
    }


def history_out(h: dict) -> dict:
    changes = {}
    for key, val in h.get("changes", {}).items():
        if key in ("punch_in", "punch_out"):
            val = {
                "from": to_ms(val["from"]) if val.get("from") else None,
                "to": to_ms(val["to"]) if val.get("to") else None,
            }
        changes[key] = val
    return {"at": to_ms(h["at"]), "by": h["by"], "reason": h["reason"], "changes": changes}


def attendance_out(d: dict) -> dict:
    return {
        "emp_code": d["emp_code"],
        "date": d["date"],
        "status": d["status"],
        "punch_in": to_ms(d["punch_in"]) if d.get("punch_in") else None,
        "punch_out": to_ms(d["punch_out"]) if d.get("punch_out") else None,
        "work_hours": d.get("work_hours"),
        "late_minutes": d.get("late_minutes", 0),
        "overtime_minutes": d.get("overtime_minutes", 0),
        "half_day": d.get("half_day", False),
        "history": [history_out(h) for h in d.get("history", [])],
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
EpochMs = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]
HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM)
    shift_end: str = Field(default="18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def joined_on_is_date(cls, v: str) -> str:
        return check_date(v)

    @model_validator(mode="after")
    def shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "MongoDB is not reachable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_utc_ms()
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {"department": department} if department else {}
    total = db.employees.count_documents(q)
    cursor = (
        db.employees.find(q)
        .sort("emp_code", ASCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "items": [employee_out(d) for d in cursor],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc_seconds()
    d = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, d, emp["shift_start"]),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return attendance_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc_seconds()
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )
    if not rec:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    if ts <= rec["punch_in"]:
        raise HTTPException(422, "punched_at must be after punch_in")
    if ts - rec["punch_in"] > timedelta(hours=24):
        raise HTTPException(422, "punched_at is more than 24 hours after punch_in")
    hours = compute_work_hours(rec["punch_in"], ts)
    update = {
        "punch_out": ts,
        "work_hours": hours,
        "overtime_minutes": compute_overtime(ts, rec["date"], emp["shift_start"], emp["shift_end"]),
        "half_day": compute_half_day(hours),
    }
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": update},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return attendance_out(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    date_from = date_param(date_from, "date_from")
    date_to = date_param(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    q = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    total = db.attendance_logs.count_documents(q)
    cursor = (
        db.attendance_logs.find(q)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "items": [attendance_out(d) for d in cursor],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    date = date_param(date, "date")
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        raise HTTPException(404, "no attendance record for that date")
    new_status = body.status or rec["status"]
    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "ABSENT/LEAVE cannot have punch times")
        new_in = new_out = None
    else:
        new_in = from_ms(body.punch_in) if body.punch_in is not None else rec.get("punch_in")
        new_out = from_ms(body.punch_out) if body.punch_out is not None else rec.get("punch_out")
        if new_in is None:
            raise HTTPException(422, "a presence status requires a punch_in")
        if attendance_date(new_in, emp["shift_start"], emp["shift_end"]) != date:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if new_out is not None:
            if new_out <= new_in:
                raise HTTPException(422, "punch_out must be after punch_in")
            if new_out - new_in > timedelta(hours=24):
                raise HTTPException(422, "punch_out is more than 24 hours after punch_in")
    late = compute_late_minutes(new_in, date, emp["shift_start"]) if new_in else 0
    if new_in and new_out:
        hours = compute_work_hours(new_in, new_out)
        overtime = compute_overtime(new_out, date, emp["shift_start"], emp["shift_end"])
        half = compute_half_day(hours)
    else:
        hours, overtime, half = None, 0, False
    new = {
        "status": new_status,
        "punch_in": new_in,
        "punch_out": new_out,
        "work_hours": hours,
        "late_minutes": late,
        "overtime_minutes": overtime,
        "half_day": half,
    }
    old = {
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes", 0),
        "overtime_minutes": rec.get("overtime_minutes", 0),
        "half_day": rec.get("half_day", False),
    }
    changes = {k: {"from": old[k], "to": new[k]} for k in new if old[k] != new[k]}
    if not changes:
        raise HTTPException(422, "the request changes nothing")
    entry = {
        "at": now_utc_ms(),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }
    hist = rec.get("history")
    guard = {
        "_id": rec["_id"],
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "history": {"$size": len(hist)} if hist is not None else {"$exists": False},
    }
    updated = db.attendance_logs.find_one_and_update(
        guard,
        {"$set": new, "$push": {"history": entry}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "record was changed by someone else, please retry")
    return attendance_out(updated)


# --------------------------------------------------------------------------- #
# Analytics helpers
# --------------------------------------------------------------------------- #
def month_bounds(month: str):
    y, m = int(month[:4]), int(month[5:7])
    return y, m, calendar.monthrange(y, m)[1]


def working_days_in_month(month: str, joined_on: str) -> int:
    """R7: Mon-Fri days of the month, counted from joined_on for mid-month joiners."""
    y, m, last = month_bounds(month)
    first = max(date_cls(y, m, 1), date_cls.fromisoformat(joined_on))
    end = date_cls(y, m, last)
    count, day = 0, first
    while day <= end:
        if day.weekday() < 5:
            count += 1
        day += timedelta(days=1)
    return count


def monthly_pipeline(emp_code: str, month: str) -> list:
    _, _, last = month_bounds(month)
    return [
        {"$match": {
            "emp_code": emp_code,
            "date": {"$gte": f"{month}-01", "$lte": f"{month}-{last:02d}"},
        }},
        {"$addFields": {
            "dow": {"$isoDayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}}},
        }},
        {"$group": {
            "_id": None,
            "present_days": {"$sum": {"$cond": [
                {"$and": [{"$in": ["$status", PRESENT_STATUSES]}, {"$lte": ["$dow", 5]}]},
                {"$cond": [{"$eq": [{"$ifNull": ["$half_day", False]}, True]}, 0.5, 1]},
                0,
            ]}},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "employee not found")
    rows = list(db.attendance_logs.aggregate(monthly_pipeline(emp_code, month)))
    r = rows[0] if rows else {}
    working = working_days_in_month(month, emp["joined_on"])
    present = round_half_up(r.get("present_days", 0))
    pct = None
    if working > 0:
        pct = round_half_up(Decimal(str(present)) / Decimal(working) * 100)
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working,
        "present_days": present,
        "leave_days": r.get("leave_days", 0),
        "late_count": r.get("late_count", 0),
        "total_late_minutes": r.get("total_late_minutes", 0),
        "total_overtime_minutes": r.get("total_overtime_minutes", 0),
        "attendance_pct": pct,
    }



# --------------------------------------------------------------------------- #
# Missing analytics endpoints and MongoDB explain
# --------------------------------------------------------------------------- #

@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    _, _, last = month_bounds(month)
    end_date = f"{month}-{last:02d}"

    match = {"joined_on": {"$lte": end_date}}
    if department is not None:
        match["department"] = department

    pipeline = [
        {"$match": match},
        {"$group": {
            "_id": "$department",
            "headcount": {"$sum": 1},
            "codes": {"$push": "$emp_code"},
        }},
        {"$lookup": {
            "from": "attendance_logs",
            "let": {"codes": "$codes"},
            "pipeline": [
                {"$match": {
                    "date": {"$gte": f"{month}-01", "$lte": end_date}
                }},
                {"$match": {"$expr": {"$in": ["$emp_code", "$$codes"]}}},
                {"$addFields": {
                    "dow": {"$isoDayOfWeek": {
                        "$dateFromString": {
                            "dateString": "$date",
                            "format": "%Y-%m-%d",
                        }
                    }}
                }},
                {"$group": {
                    "_id": None,
                    "present_days": {"$sum": {"$cond": [
                        {"$and": [
                            {"$in": ["$status", PRESENT_STATUSES]},
                            {"$lte": ["$dow", 5]},
                        ]},
                        {"$cond": [
                            {"$eq": [{"$ifNull": ["$half_day", False]}, True]},
                            0.5, 1,
                        ]},
                        0,
                    ]}},
                    "hours": {"$avg": {"$cond": [
                        {"$and": [
                            {"$in": ["$status", PRESENT_STATUSES]},
                            {"$ne": ["$work_hours", None]},
                        ]},
                        "$work_hours", None,
                    ]}},
                    "late_count": {"$sum": {"$cond": [
                        {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                        1, 0,
                    ]}},
                    "total_late_minutes": {"$sum": {
                        "$ifNull": ["$late_minutes", 0]
                    }},
                    "leave_count": {"$sum": {"$cond": [
                        {"$eq": ["$status", "LEAVE"]}, 1, 0,
                    ]}},
                    "on_duty_count": {"$sum": {"$cond": [
                        {"$eq": ["$status", "ON_DUTY"]}, 1, 0,
                    ]}},
                }},
            ],
            "as": "stats",
        }},
        {"$project": {
            "_id": 0,
            "department": "$_id",
            "headcount": 1,
            "present_days": {
                "$ifNull": [{"$arrayElemAt": ["$stats.present_days", 0]}, 0]
            },
            "avg_work_hours": {
                "$arrayElemAt": ["$stats.hours", 0]
            },
            "late_count": {
                "$ifNull": [{"$arrayElemAt": ["$stats.late_count", 0]}, 0]
            },
            "total_late_minutes": {
                "$ifNull": [
                    {"$arrayElemAt": ["$stats.total_late_minutes", 0]}, 0
                ]
            },
            "leave_count": {
                "$ifNull": [{"$arrayElemAt": ["$stats.leave_count", 0]}, 0]
            },
            "on_duty_count": {
                "$ifNull": [{"$arrayElemAt": ["$stats.on_duty_count", 0]}, 0]
            },
        }},
        {"$sort": {"department": 1}},
    ]

    items = list(db.employees.aggregate(pipeline))
     
    for item in items:
         item["present_days"] = round_half_up(
             item.get("present_days", 0)
        )

         avg_hours = item.get("avg_work_hours")
         if avg_hours is not None:
            item["avg_work_hours"] = round_half_up(avg_hours)
         else:
            item["avg_work_hours"] = None
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    _, _, last = month_bounds(month)

    pipeline = [
        {"$match": {
            "date": {"$gte": f"{month}-01", "$lte": f"{month}-{last:02d}"},
            "late_minutes": {"$gt": 0},
        }},
        {"$group": {
            "_id": "$emp_code",
            "total_late_minutes": {"$sum": "$late_minutes"},
            "late_count": {"$sum": 1},
        }},
        {"$lookup": {
            "from": "employees",
            "localField": "_id",
            "foreignField": "emp_code",
            "as": "employee",
        }},
        {"$unwind": "$employee"},
    ]

    if department is not None:
        pipeline.append({"$match": {"employee.department": department}})

    pipeline.extend([
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$setWindowFields": {
            "sortBy": {"total_late_minutes": -1},
            "output": {"rank": {"$rank": {}}},
        }},
        {"$match": {"rank": {"$lte": limit}}},
        {"$project": {
            "_id": 0,
            "rank": 1,
            "emp_code": "$_id",
            "name": "$employee.name",
            "department": "$employee.department",
            "total_late_minutes": 1,
            "late_count": 1,
        }},
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}},
    ])

    return {
        "month": month,
        "items": list(db.attendance_logs.aggregate(pipeline)),
    }


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
):
    from_date = date_param(from_date, "from")
    to_date = date_param(to_date, "to")

    if to_date < from_date:
        raise HTTPException(422, "'to' must not be before 'from'")

    start = date_cls.fromisoformat(from_date)
    end = date_cls.fromisoformat(to_date)
    day_count = (end - start).days + 1

    if day_count > 92:
        raise HTTPException(422, "date range must not exceed 92 days")

    pipeline = [
        {"$match": {"department": department}},
        {"$group": {
            "_id": None,
            "employees": {"$push": {
                "emp_code": "$emp_code",
                "joined_on": "$joined_on",
            }},
        }},
        {"$project": {
            "_id": 0,
            "employees": 1,
            "days": {"$map": {
                "input": {"$range": [0, day_count]},
                "as": "offset",
                "in": {"$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": {"$dateAdd": {
                        "startDate": {
                            "$dateFromString": {"dateString": from_date}
                        },
                        "unit": "day",
                        "amount": "$$offset",
                    }},
                    "timezone": "UTC",
                }},
            }},
        }},
        {"$unwind": "$days"},
        {"$lookup": {
            "from": "attendance_logs",
            "let": {"day": "$days", "employees": "$employees.emp_code"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$date", "$$day"]},
                    {"$in": ["$emp_code", "$$employees"]},
                ]}}},
                {"$group": {
                    "_id": None,
                    "present_count": {"$sum": {"$cond": [
                        {"$in": ["$status", PRESENT_STATUSES]},
                        {"$cond": [
                            {"$eq": [{"$ifNull": ["$half_day", False]}, True]},
                            0.5, 1,
                        ]},
                        0,
                    ]}},
                    "late_count": {"$sum": {"$cond": [
                        {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                        1, 0,
                    ]}},
                }},
            ],
            "as": "stats",
        }},
        {"$set": {
            "headcount": {"$size": {"$filter": {
                "input": "$employees",
                "as": "employee",
                "cond": {"$lte": ["$$employee.joined_on", "$days"]},
            }}},
            "is_working_day": {"$lte": [
                {"$isoDayOfWeek": {
                    "$dateFromString": {"dateString": "$days"}
                }},
                5,
            ]},
            "present_count": {"$ifNull": [
                {"$arrayElemAt": ["$stats.present_count", 0]}, 0
            ]},
            "late_count": {"$ifNull": [
                {"$arrayElemAt": ["$stats.late_count", 0]}, 0
            ]},
        }},
        {"$set": {
            "attendance_rate": {"$cond": [
                {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                {"$divide": [
                    {"$floor": {"$add": [
                        {"$multiply": [
                            {"$divide": ["$present_count", "$headcount"]},
                            10000,
                        ]},
                        0.5,
                    ]}},
                    10000,
                ]},
                None,
            ]},
        }},
        {"$setWindowFields": {
            "sortBy": {"days": 1},
            "output": {
                "avg_raw": {"$avg": "$attendance_rate",
                            "window": {"documents": [-6, 0]}},
            },
        }},
        {"$project": {
            "_id": 0,
            "date": "$days",
            "is_working_day": 1,
            "headcount": 1,
            "present_count": 1,
            "late_count": 1,
            "attendance_rate": 1,
            "moving_avg_7d": {"$cond": [
                {"$eq": ["$avg_raw", None]},
                None,
                {"$divide": [
                    {"$floor": {"$add": [
                        {"$multiply": ["$avg_raw", 10000]}, 0.5
                    ]}},
                    10000,
                ]},
            ]},
        }},
        {"$sort": {"date": 1}},
    ]

    # No employees means the requested department does not exist.
    if not db.employees.find_one({"department": department}, {"_id": 1}):
        raise HTTPException(404, "department not found")

    return {
        "department": department,
        "items": list(db.employees.aggregate(pipeline)),
    }


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(
        None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"
    ),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    allowed = {
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    }
    if endpoint not in allowed:
        raise HTTPException(422, "unsupported endpoint")

    def require(value, name):
        if value is None:
            raise HTTPException(422, f"{name} is required")
        return value

    def explain(collection, pipeline=None, query=None, sort=None,
                skip=None, take=None):
        if pipeline is not None:
            command = {
                "aggregate": collection.name,
                "pipeline": pipeline,
                "cursor": {},
            }
        else:
            command = {"find": collection.name, "filter": query or {}}
            if sort:
                command["sort"] = sort
            if skip is not None:
                command["skip"] = skip
            if take is not None:
                command["limit"] = take

        result = db.command(
            "explain", command, verbosity="executionStats"
        )
        return {
            "endpoint": endpoint,
            "collection": collection.name,
            "explain": result,
        }

    if endpoint == "attendance_list":
        date_from = date_param(date_from, "date_from")
        date_to = date_param(date_to, "date_to")
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must not be after date_to")

        query = {}
        if emp_code:
            query["emp_code"] = emp_code
        if status:
            query["status"] = status
        if date_from or date_to:
            query["date"] = {}
            if date_from:
                query["date"]["$gte"] = date_from
            if date_to:
                query["date"]["$lte"] = date_to

        return explain(
            db.attendance_logs,
            query=query,
            sort=[("date", -1), ("emp_code", 1)],
            skip=(page - 1) * page_size,
            take=page_size,
        )

    if endpoint == "employee_monthly":
        emp_code = require(emp_code, "emp_code")
        month = require(month, "month")
        validate_month(month)
        return explain(
            db.attendance_logs,
            pipeline=monthly_pipeline(emp_code, month),
        )

    if endpoint == "department_summary":
        month = require(month, "month")
        validate_month(month)
        _, _, last = month_bounds(month)
        match = {"joined_on": {"$lte": f"{month}-{last:02d}"}}
        if department is not None:
            match["department"] = department
        return explain(
            db.employees,
            pipeline=[
                {"$match": match},
                {"$group": {"_id": "$department", "count": {"$sum": 1}}},
            ],
        )

    if endpoint == "late_leaderboard":
        month = require(month, "month")
        validate_month(month)
        _, _, last = month_bounds(month)
        pipeline = [
            {"$match": {
                "date": {"$gte": f"{month}-01", "$lte": f"{month}-{last:02d}"},
                "late_minutes": {"$gt": 0},
            }},
            {"$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
            }},
        ]
        return explain(db.attendance_logs, pipeline=pipeline)

    department = require(department, "department")
    from_date = date_param(require(from_date, "from"), "from")
    to_date = date_param(require(to_date, "to"), "to")
    if to_date < from_date or (
        date_cls.fromisoformat(to_date) -
        date_cls.fromisoformat(from_date)
    ).days >= 92:
        raise HTTPException(422, "invalid trend date range")

    if not db.employees.find_one({"department": department}, {"_id": 1}):
        raise HTTPException(404, "department not found")

    return explain(
        db.attendance_logs,
        query={"date": {"$gte": from_date, "$lte": to_date}},
    )
