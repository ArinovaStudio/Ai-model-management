const dashboard = {
    async init() {
        await this.loadProjects();
    },

    async loadProjects() {
        try {
            const projects = await window.api.getProjects();
            this.renderProjects(projects);
        } catch (error) {
            console.error('Failed to load projects:', error);
            alert('Could not load projects. Ensure the backend is running.');
        }
    },

    renderProjects(projects) {
        const grid = document.getElementById('project-grid');
        const emptyState = document.getElementById('empty-state');
        
        grid.innerHTML = '';
        
        if (projects.length === 0) {
            emptyState.style.display = 'block';
            return;
        }
        
        emptyState.style.display = 'none';
        
        projects.forEach(p => {
            const date = new Date(p.createdAt).toLocaleDateString();
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <h3 style="margin-bottom: 0.25rem;">${p.name}</h3>
                    <span class="badge badge-success">Active</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--color-sedate-gray); margin-bottom: 1.5rem;">
                    <p>Created: ${date}</p>
                    <p>Phase: ${p.currentPhase}</p>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--color-sedate-gray-light); padding-top: 1rem;">
                    <div style="font-size: 0.8rem; font-weight: 500;">
                        <span>Tasks: ${p.wbs_tasks_count}</span>
                    </div>
                    <a href="workspace.html?id=${p.id}" class="btn btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;">Open Workspace &rarr;</a>
                </div>
            `;
            grid.appendChild(card);
        });
    },

    openNewProjectModal() {
        document.getElementById('new-project-modal').classList.add('active');
    },

    closeNewProjectModal() {
        document.getElementById('new-project-modal').classList.remove('active');
        document.getElementById('new-project-form').reset();
    },

    async handleCreateProject(e) {
        e.preventDefault();
        const btn = document.getElementById('submit-btn');
        btn.textContent = 'Creating...';
        btn.disabled = true;

        const data = {
            name: document.getElementById('project-name').value,
            start_date: new Date().toISOString(),
            metadata: {
                client_name: document.getElementById('project-client').value,
                project_type: document.getElementById('project-type').value,
                budget: parseFloat(document.getElementById('project-budget').value)
            }
        };

        try {
            const res = await window.api.createProject(data);
            this.closeNewProjectModal();
            // Go straight to workspace
            window.location.href = `workspace.html?id=${res.id}`;
        } catch (error) {
            console.error('Create error:', error);
            alert('Failed to create project.');
            btn.textContent = 'Create Project';
            btn.disabled = false;
        }
    }
};

document.addEventListener('DOMContentLoaded', () => dashboard.init());
window.dashboard = dashboard;
