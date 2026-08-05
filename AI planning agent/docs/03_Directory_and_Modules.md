# 03. Codebase Directory and Modules

Here is the structured folder-by-folder breakdown of the **AI Planning Agent** workspace:

---

## 1. Directory Tree Map
```
AI planning agent/
├── app/
│   ├── db/
│   │   └── prisma.py                 # Prisma client initializer and in-memory cache helpers
│   ├── models/
│   │   └── schemas.py                # Pydantic schemas for request/response serialization
│   ├── routers/
│   │   ├── analysis.py               # Routes for requirements extraction
│   │   ├── docs.py                   # Documentation viewing routes
│   │   ├── enhancement.py            # Targeted section modifier, rollbacks, and ID deduplication
│   │   ├── estimation.py             # Effort and cost estimation routes
│   │   ├── intelligence.py           # Architecture specification routes
│   │   ├── projects.py               # Project CRUD and status routes
│   │   ├── resource_planning.py      # Resource allocations routes
│   │   ├── schedule.py               # Sprints and gantt timeline scheduling routes
│   │   ├── validation.py             # Timeline feasibility validations routes
│   │   └── wbs.py                    # WBS tasks and milestones generator routes
│   ├── services/
│   │   ├── analysis_engine.py        # Requirements processor
│   │   ├── estimation_engine.py      # Task effort estimator (adds order indices to tasks/groups)
│   │   ├── gap_engine.py             # Final project constraint checks
│   │   ├── intelligence_engine.py    # System specification builder
│   │   ├── ollama_service.py         # LLM prompt communications wrapper
│   │   ├── resource_engine.py        # Team allocation builder
│   │   ├── schedule_engine.py        # Sprints, Gantt scheduling and milestone assigner
│   │   ├── validation_engine.py      # timeline validation check engine
│   │   └── wbs_engine.py             # WBS tasks builder
│   ├── config.py                     # App settings load configuration (.env mapping)
│   └── main.py                       # FastAPI application entrypoint and middleware registry
├── docs/                             # Project-specific documentation suite
├── frontend/                         # Workspace HTML, CSS, and Vanilla JS Controllers
│   ├── js/
│   │   ├── api.js                    # HTTP fetch calls wrapper
│   │   ├── dashboard.js              # Projects landing page controller
│   │   └── workspace.js              # Plan generation steppers controller
│   ├── css/
│   │   └── index.css                 # Styling stylesheet
│   └── workspace.html                # Workspace generation interface view
├── prisma/
│   └── schema.prisma                 # Prisma SQLite/PostgreSQL schema configuration
├── backups/                          # Temporary local backups stored during enhancements
├── .env                              # Environment configuration file
├── profile_launcher.py               # Monkeypatched profiling and instrumentation launcher script
└── requirements.txt                  # Python dependency list
```

---

## 2. Key Code Organization Rules
* **Controllers and Routes (`app/routers/`):** Expose HTTP endpoints, validate request formats via Pydantic (`app/models/schemas.py`), and hand off task processing to services.
* **Services (`app/services/`):** House all the mathematical algorithms, prompt building logic, and data processing heuristics. They do not interact directly with routes.
* **Database Caching (`app/db/prisma.py`):** Acts as the centralized Prisma client provider and stores local cache matrices to preserve memory state during sequential pipeline executions.
