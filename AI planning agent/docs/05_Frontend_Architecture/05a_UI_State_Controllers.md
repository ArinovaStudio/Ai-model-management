# 05a. UI State Controllers

The frontend interface for the AI Planning Agent workspace is structured around a step-by-step wizard. UI updates, state coordination, and DOM rendering are handled by Vanilla JavaScript controllers located in `frontend/js/`.

---

## 1. Workspace Stepper (`workspace.js`)
The `workspace.js` controller manages the core project planning wizard:
* **Step Progression:** Coordinates transitions between stages (e.g. Requirements, WBS, Resource Planning, Sprints, Timeline Validation).
* **DOM Rendering Engine:** Parses API JSON responses on-the-fly and builds dynamic layouts (such as the Gantt chart and validation warning cards).
* **State Syncing:** Collects user feedbacks and timeline updates from the DOM and synchronizes them with the backend database.

---

## 2. Dashboards Controller (`dashboard.js`)
* **Purpose:** Handles the project landing interface.
* **Responsibilities:**
  * Lists active projects and retrieves status metrics.
  * Manages the "Create New Project" modal form.
  * Passes selected project details to the main workspace router.

---

## 3. Live Section Editing Interface
When a user clicks "Enhance" on a plan section, the controller:
1. Gathers the natural language inputs and target section markers.
2. Dispatches an asynchronous POST request to the `/enhance` endpoint.
3. If successful, invalidates local UI step state and calls the API to pull fresh database records.
4. Renders the newly modified structure and refreshes timeline dependencies.
5. If the request fails, displays the rollback error message to the user without altering the UI layout.
