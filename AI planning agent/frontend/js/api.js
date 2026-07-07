const defaultBase = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
    ? 'http://127.0.0.1:8000/api/v1/projects' 
    : '/api/v1/projects';
const API_BASE_URL = window.API_BASE_URL || defaultBase;

const api = {
    async getProjects() {
        const res = await fetch(API_BASE_URL);
        if (!res.ok) throw new Error('Failed to fetch projects');
        return res.json();
    },

    async getProject(id) {
        const res = await fetch(`${API_BASE_URL}/${id}`);
        if (!res.ok) throw new Error('Failed to fetch project');
        return res.json();
    },

    async createProject(data) {
        const res = await fetch(API_BASE_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) throw new Error('Failed to create project');
        return res.json();
    },

    async updateProjectInfo(id, data) {
        const res = await fetch(`${API_BASE_URL}/${id}/info`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) throw new Error('Failed to update project info');
        return res.json();
    },

    async uploadDocs(id, file, docType = 'plandoc') {
        const content = await file.text();
        const payload = {
            title: file.name,
            content: content,
            doc_type: docType
        };
        const res = await fetch(`${API_BASE_URL}/${id}/docs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Failed to upload document');
        return res.json();
    },

    async analyzeProject(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true })
        });
        if (!res.ok) throw new Error('Failed to analyze project');
        return res.json();
    },

    async generateWBS(id, prefs = {}) {
        const res = await fetch(`${API_BASE_URL}/${id}/wbs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true, ...prefs })
        });
        if (!res.ok) throw new Error('Failed to generate WBS');
        return res.json();
    },

    async generateEstimates(id, prefs = {}) {
        const res = await fetch(`${API_BASE_URL}/${id}/estimate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true, ...prefs })
        });
        if (!res.ok) throw new Error('Failed to generate estimates');
        return res.json();
    },

    async generateResourcePlan(id, prefs = {}) {
        const res = await fetch(`${API_BASE_URL}/${id}/resource-plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true, ...prefs })
        });
        if (!res.ok) throw new Error('Failed to generate resource plan');
        return res.json();
    },

    async generateSchedule(id, prefs = {}) {
        const res = await fetch(`${API_BASE_URL}/${id}/schedule`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true, ...prefs })
        });
        if (!res.ok) throw new Error('Failed to generate schedule');
        return res.json();
    },

    async getEstimates(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/estimate`);
        if (!res.ok) throw new Error('Failed to get estimates');
        return res.json();
    },

    async getSchedule(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/schedule`);
        if (!res.ok) throw new Error('Failed to get schedule');
        return res.json();
    },

    async getResourcePlan(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/resource-plan`);
        if (!res.ok) throw new Error('Failed to get resource plan');
        return res.json();
    },

    async validatePlan(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true })
        });
        if (!res.ok) throw new Error('Failed to validate plan');
        return res.json();
    },

    async getValidation(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/validate`);
        if (!res.ok) throw new Error('Failed to get validation');
        return res.json();
    },

    async runGapAnalysis(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/gap-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true })
        });
        if (!res.ok) throw new Error('Failed to run gap analysis');
        return res.json();
    },

    async getGapAnalysis(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/gap-analysis`);
        if (!res.ok) throw new Error('Failed to get gap analysis');
        return res.json();
    },

    async searchContext(id, query) {
        const res = await fetch(`${API_BASE_URL}/${id}/search?query=${encodeURIComponent(query)}`);
        if (!res.ok) throw new Error('Search failed');
        return res.json();
    }
};

window.api = api;
