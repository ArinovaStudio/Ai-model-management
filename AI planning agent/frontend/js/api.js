const API_BASE_URL = 'http://127.0.0.1:8000/api/v1/projects';

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

    async generateWBS(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/wbs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true })
        });
        if (!res.ok) throw new Error('Failed to generate WBS');
        return res.json();
    },

    async generateEstimates(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/estimate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true })
        });
        if (!res.ok) throw new Error('Failed to generate estimates');
        return res.json();
    },

    async generateResourcePlan(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/resource-plan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true })
        });
        if (!res.ok) throw new Error('Failed to generate resource plan');
        return res.json();
    },

    async generateSchedule(id) {
        const res = await fetch(`${API_BASE_URL}/${id}/schedule`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reindex: true })
        });
        if (!res.ok) throw new Error('Failed to generate schedule');
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

    async searchContext(id, query) {
        const res = await fetch(`${API_BASE_URL}/${id}/search?query=${encodeURIComponent(query)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (!res.ok) throw new Error('Search failed');
        return res.json();
    }
};

window.api = api;
