const workspace = {
    projectId: null,
    projectData: null,

    async init() {
        const urlParams = new URLSearchParams(window.location.search);
        this.projectId = urlParams.get('id');
        
        if (!this.projectId) {
            window.location.href = 'index.html';
            return;
        }

        await this.loadProject();
    },

    async loadProject() {
        try {
            this.projectData = await window.api.getProject(this.projectId);
            document.getElementById('project-title').textContent = this.projectData.name || 'Project Workspace';
            
            // If project already has WBS/Schedule generated, show plan state
            if (this.projectData.wbs_tasks_count > 0 || this.projectData.milestones_count > 0) {
                this.showPlanState();
                
                try {
                    const val = await window.api.getValidation(this.projectId);
                    this.updateConfidenceBanner(val.confidence.score);
                    // The getValidation endpoint might not return warnings from the DB history accurately if not stored, but let's try
                    if (val.warnings) this.renderWarnings(val.warnings);
                } catch(e) { console.warn("No validation data found"); }

                await this.loadWBS();
            }
        } catch (error) {
            console.error('Failed to load project:', error);
            document.getElementById('project-title').textContent = 'Error loading project';
        }
    },

    async handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const statusDiv = document.getElementById('upload-status');
        statusDiv.textContent = 'Uploading...';
        
        try {
            await window.api.uploadDocs(this.projectId, file);
            statusDiv.textContent = `Uploaded: ${file.name}`;
            document.getElementById('dropzone').classList.add('active');
        } catch (error) {
            statusDiv.textContent = 'Upload failed. See console.';
            statusDiv.style.color = 'red';
        }
    },

    async runPipeline() {
        const btn = document.getElementById('generate-btn');
        const stepper = document.getElementById('pipeline-stepper');
        
        // Extract preferences
        const sprintDurVal = parseInt(document.getElementById('sprint-duration').value, 10);
        const prodHrsVal = parseFloat(document.getElementById('productive-hours').value);
        
        const prefs = {
            sprint_duration_days: isNaN(sprintDurVal) ? 14 : sprintDurVal,
            productive_hours_per_day: isNaN(prodHrsVal) ? 6.0 : prodHrsVal
        };

        btn.disabled = true;
        btn.textContent = 'Pipeline Running...';
        stepper.style.display = 'block';

        try {
            this.updateStep(1, 'active');
            await window.api.analyzeProject(this.projectId);
            this.updateStep(1, 'done');

            this.updateStep(2, 'active');
            await window.api.generateWBS(this.projectId, prefs);
            this.updateStep(2, 'done');

            this.updateStep(3, 'active');
            await window.api.generateEstimates(this.projectId, prefs);
            this.updateStep(3, 'done');

            this.updateStep(4, 'active');
            await window.api.generateResourcePlan(this.projectId, prefs);
            this.updateStep(4, 'done');

            this.updateStep(5, 'active');
            await window.api.generateSchedule(this.projectId, prefs);
            this.updateStep(5, 'done');

            this.updateStep(6, 'active');
            const validation = await window.api.validatePlan(this.projectId);
            this.updateStep(6, 'done');

            this.updateStep(7, 'active');
            try {
                await window.api.runGapAnalysis(this.projectId);
            } catch(e) { console.warn("Gap analysis failed/skipped", e); }
            this.updateStep(7, 'done');

            // Set Confidence Banner and Warnings
            this.updateConfidenceBanner(validation.confidence.score);
            this.renderWarnings(validation.warnings);
            
            // Transition to Plan State
            setTimeout(() => {
                this.showPlanState();
                this.loadWBS();
            }, 1500);

        } catch (error) {
            console.error('Pipeline failed:', error);
            alert('Pipeline failed at a step. Check console.');
            btn.disabled = false;
            btn.textContent = 'Retry Generation';
        }
    },

    updateStep(stepNum, status) {
        const stepEl = document.getElementById(`step-${stepNum}`);
        if (status === 'active') {
            stepEl.classList.add('active');
        } else if (status === 'done') {
            stepEl.classList.remove('active');
            stepEl.classList.add('done');
            stepEl.querySelector('.step-icon').innerHTML = '✓';
        }
    },

    updateConfidenceBanner(score) {
        const banner = document.getElementById('confidence-banner');
        banner.style.display = 'inline-block';
        banner.textContent = `Confidence Score: ${score}/100`;
        
        if (score >= 80) banner.classList.add('badge-success');
        else if (score >= 50) banner.classList.add('badge-warning');
        else banner.classList.add('badge-neutral');
    },

    renderWarnings(warnings) {
        const container = document.getElementById('warnings-container');
        if (!warnings || warnings.length === 0) {
            container.style.display = 'none';
            return;
        }
        container.style.display = 'flex';
        container.innerHTML = '';
        warnings.forEach(w => {
            const color = w.severity === 'critical' ? '#dc3545' : (w.severity === 'high' ? '#ffc107' : '#17a2b8');
            const textColor = w.severity === 'high' ? '#000' : '#fff';
            container.innerHTML += `
                <div style="background: ${color}; color: ${textColor}; padding: 0.75rem 1rem; border-radius: var(--radius-md); font-size: 0.85rem; font-weight: 500;">
                    <strong>${w.category.toUpperCase()}:</strong> ${w.message}
                </div>
            `;
        });
    },

    showPlanState() {
        document.getElementById('setup-state').style.display = 'none';
        document.getElementById('plan-state').style.display = 'block';
    },

    switchTab(tabName) {
        // Update tab styling
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        event.target.classList.add('active');

        // Update content panes
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        document.getElementById(`tab-${tabName}`).classList.add('active');

        // Lazy load data if needed
        if (tabName === 'schedule') this.loadSchedule();
        if (tabName === 'resources') this.loadResources();
        if (tabName === 'risks') this.loadGapAnalysis();
    },

    async loadWBS() {
        // Actually fetches from estimation endpoint to get effort data too
        const container = document.getElementById('wbs-container');
        container.innerHTML = '<p>Loading WBS...</p>';
        try {
            // Using estimation endpoint since it includes task hours
            const res = await window.api.getEstimates(this.projectId);
            container.innerHTML = '';
            
            res.groups.forEach(g => {
                const epicHtml = `
                    <div class="tree-epic">
                        <div class="tree-epic-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                            <span>▾ ${g.group_name}</span>
                            <span class="badge badge-neutral">${g.total_effort_hours} hrs</span>
                        </div>
                        <div style="display: block;">
                            ${g.tasks.map(t => {
                                const depHtml = t.dependencies && t.dependencies.length > 0 
                                    ? '<span title="Depends on other tasks" style="cursor: help; margin-left: 0.5rem; font-size: 0.8rem;">🔗</span>' 
                                    : '';
                                return `
                                <div class="tree-task">
                                    <span>${t.task_title}${depHtml}</span>
                                    <span style="color: var(--text-muted); font-size: 0.8rem;">${t.effort_hours}h</span>
                                </div>
                                `
                            }).join('')}
                        </div>
                    </div>
                `;
                container.innerHTML += epicHtml;
            });
        } catch (e) {
            container.innerHTML = '<p>Failed to load WBS.</p>';
        }
    },

    async loadSchedule() {
        const container = document.getElementById('schedule-container');
        if (container.innerHTML !== '') return; // Already loaded

        container.innerHTML = '<p>Loading Schedule...</p>';
        try {
            const res = await window.api.getSchedule(this.projectId);
            container.innerHTML = '';
            
            // Group by sprint number
            const sprintMap = {};
            res.sprints.forEach(s => {
                if (!sprintMap[s.sprint_number]) {
                    sprintMap[s.sprint_number] = {
                        start_date: s.start_date,
                        end_date: s.end_date,
                        task_count: 0
                    };
                }
                sprintMap[s.sprint_number].task_count += s.task_count;
            });

            Object.keys(sprintMap).sort((a,b) => a-b).forEach(sprintNum => {
                const s = sprintMap[sprintNum];
                const start = new Date(s.start_date).toLocaleDateString();
                const end = new Date(s.end_date).toLocaleDateString();
                
                container.innerHTML += `
                    <div class="sprint-block">
                        <h4 style="margin-bottom: 0.5rem; color: var(--color-sedate-gray);">Sprint ${sprintNum}</h4>
                        <p style="font-size: 0.85rem; font-weight: 500;">${start} - ${end}</p>
                        <p style="font-size: 0.8rem; margin-top: 0.5rem;">${s.task_count} tasks assigned</p>
                    </div>
                `;
            });

            // Day-Wise Tasks visualization
            if (res.tasks && res.tasks.length > 0) {
                container.innerHTML += `<h4 style="margin-top: 2rem; margin-bottom: 1rem; color: var(--color-sedate-gray); border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom: 0.5rem;">Day-Wise Schedule</h4>`;
                
                // Group tasks by their start date string
                const dayMap = {};
                res.tasks.forEach(t => {
                    if (!t.scheduled_start) return;
                    const dateStr = new Date(t.scheduled_start).toLocaleDateString();
                    if (!dayMap[dateStr]) dayMap[dateStr] = [];
                    dayMap[dateStr].push(t);
                });

                // Sort dates chronologically
                const sortedDates = Object.keys(dayMap).sort((a, b) => new Date(a) - new Date(b));

                sortedDates.forEach(dateStr => {
                    const tasks = dayMap[dateStr];
                    let tasksHtml = tasks.map(t => {
                        const endStr = new Date(t.scheduled_end).toLocaleDateString();
                        return `
                            <div style="padding: 0.75rem; background: var(--bg-main); border: 1px solid var(--color-sedate-gray-light); border-radius: var(--radius-sm); margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-weight: 500; font-size: 0.9rem;">${t.title}</div>
                                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">Duration: ${t.duration_days} day(s) | Ends: ${endStr}</div>
                                </div>
                            </div>
                        `;
                    }).join('');

                    container.innerHTML += `
                        <div style="margin-bottom: 1.5rem; border-left: 3px solid var(--color-sage); padding-left: 1rem;">
                            <h5 style="margin-bottom: 0.75rem; color: var(--text-muted);">${dateStr}</h5>
                            ${tasksHtml}
                        </div>
                    `;
                });
            }

            const mContainer = document.getElementById('milestones-container');
            if (mContainer && res.milestones) {
                mContainer.innerHTML = '';
                res.milestones.forEach(m => {
                    const due = new Date(m.due_date).toLocaleDateString();
                    mContainer.innerHTML += `
                        <div style="padding-left: 1rem; border-left: 2px solid var(--color-dusty-rose); position: relative;">
                            <div style="position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: var(--color-dusty-rose);"></div>
                            <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">${m.title}</div>
                            <div style="font-size: 0.8rem; color: var(--text-muted);">Due: ${due}</div>
                        </div>
                    `;
                });
            }
        } catch (e) {
            container.innerHTML = '<p>Failed to load schedule.</p>';
        }
    },

    async loadResources() {
        const tbody = document.getElementById('resources-tbody');
        if (tbody.innerHTML !== '') return;

        tbody.innerHTML = '<tr><td colspan="3">Loading...</td></tr>';
        try {
            const res = await window.api.getResourcePlan(this.projectId);
            tbody.innerHTML = '';
            
            res.allocations.forEach(r => {
                tbody.innerHTML += `
                    <tr>
                        <td><strong>${r.role}</strong></td>
                        <td>${r.allocated_hours} hrs</td>
                        <td>${r.required_count} devs</td>
                    </tr>
                `;
            });
        } catch (e) {
            tbody.innerHTML = '<tr><td colspan="3">Failed to load resources.</td></tr>';
        }
    },

    async loadGapAnalysis() {
        const container = document.getElementById('risks-container');
        if (container.innerHTML !== '') return;
        
        container.innerHTML = '<p>Loading Gap Analysis...</p>';
        try {
            const res = await window.api.getGapAnalysis(this.projectId);
            container.innerHTML = '';
            
            const r = res.reports;
            if ((!r.assumptions || r.assumptions.length === 0) && 
                (!r.clarifications || r.clarifications.length === 0) && 
                (!r.risks || r.risks.length === 0)) {
                container.innerHTML = '<div class="card" style="text-align:center; color: var(--text-muted);">No risks or gaps identified (or LLM extraction failed).</div>';
                return;
            }

            const renderItems = (items, type) => {
                if (!items || items.length === 0) return '';
                return items.map(item => `
                    <div style="background: var(--bg-main); border: 1px solid var(--color-sedate-gray-light); padding: 1rem; border-radius: var(--radius-md); margin-bottom: 0.5rem;">
                        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem;">${item.title || 'Untitled'}</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">${item.description || ''}</div>
                        <span class="badge badge-neutral" style="font-size: 0.75rem;">Severity: ${item.severity || 'Medium'}</span>
                    </div>
                `).join('');
            };

            container.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                    <div>
                        <h4 style="margin-bottom: 1rem; color: var(--color-dusty-rose);">Project Risks</h4>
                        ${renderItems(r.risks, 'risk') || '<p style="color:var(--text-muted); font-size:0.85rem;">None</p>'}
                    </div>
                    <div>
                        <h4 style="margin-bottom: 1rem; color: var(--color-sage);">Assumptions & Clarifications</h4>
                        ${renderItems(r.assumptions, 'assumption') || ''}
                        ${renderItems(r.clarifications, 'clarification') || ''}
                        ${(!r.assumptions?.length && !r.clarifications?.length) ? '<p style="color:var(--text-muted); font-size:0.85rem;">None</p>' : ''}
                    </div>
                </div>
            `;
        } catch (e) {
            container.innerHTML = '<div class="card" style="text-align:center; color: var(--text-muted);">No Gap Analysis available for this project.</div>';
        }
    },

    async handleSearch(e) {
        e.preventDefault();
        const input = document.getElementById('chat-input');
        const query = input.value.trim();
        if (!query) return;

        const chatBox = document.getElementById('chat-messages');
        
        // Append user msg
        chatBox.innerHTML += `<div class="msg msg-user">${query}</div>`;
        input.value = '';
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const res = await window.api.searchContext(this.projectId, query);
            
            let aiText = "I found some context in your documents:<br><br>";
            res.results.forEach(r => {
                aiText += `<div style="padding: 0.5rem; background: var(--bg-card); border-radius: 4px; margin-bottom: 0.5rem; font-size: 0.8rem;">
                    <strong>${r.title}</strong> (Score: ${r.similarity_score})<br>
                    "${r.content_snippet}"
                </div>`;
            });
            if (res.results.length === 0) aiText = "I couldn't find any relevant context for that.";

            chatBox.innerHTML += `<div class="msg msg-ai">${aiText}</div>`;
        } catch (error) {
            chatBox.innerHTML += `<div class="msg msg-ai" style="color: red;">Search failed.</div>`;
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    }
};

document.addEventListener('DOMContentLoaded', () => workspace.init());
window.workspace = workspace;
