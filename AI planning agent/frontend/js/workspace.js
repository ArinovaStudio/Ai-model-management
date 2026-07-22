const workspace = {
    projectId: null,
    projectData: null,
    projectIntelligence: null,

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

            try {
                this.projectIntelligence = await window.api.getIntelligence(this.projectId);
            } catch(e) {
                console.warn("No intelligence data found yet");
            }

            if (this.projectData.wbs_tasks_count > 0 || this.projectData.milestones_count > 0) {
                this.showPlanState();
                
                try {
                    const val = await window.api.getValidation(this.projectId);
                    this.updateConfidenceBanner(val.confidence.score);
                    if (val.warnings) this.renderWarnings(val.warnings);
                } catch(e) { console.warn("No validation data found"); }

                try {
                    const sched = await window.api.getSchedule(this.projectId);
                    this.updateExpectedDeadlineBanner(sched.expected_deadline);
                } catch(e) { console.warn("No schedule found"); }

                await this.loadWBS();
            }
            await this.loadChatHistory();
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
            const descInput = document.getElementById('project-description-input');
            const archPattern = document.getElementById('project-architecture').value;
            const updatePayload = { project_type: archPattern };
            if (descInput && descInput.value.trim()) {
                updatePayload.description = descInput.value.trim();
            }
            await window.api.updateProjectInfo(this.projectId, updatePayload);

            this.updateStep(1, 'active');
            await window.api.analyzeProject(this.projectId);
            this.updateStep(1, 'done');

            console.log("Generating Project Intelligence...");
            btn.textContent = 'Generating Strategy & Architecture...';
            try {
                const intelligenceData = await window.api.generateIntelligence(this.projectId);
                this.projectIntelligence = intelligenceData;
                console.log("Intelligence generated:", intelligenceData);
            } catch(e) {
                console.warn("Intelligence generation failed/skipped", e);
            }

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

            try {
                const sched = await window.api.getSchedule(this.projectId);
                this.updateExpectedDeadlineBanner(sched.expected_deadline);
            } catch(e) { console.warn("No schedule found"); }
            
            // Clear cached layouts to force re-render on tab click
            document.getElementById('strategy-container').innerHTML = '';
            document.getElementById('architecture-container').innerHTML = '';
            document.getElementById('schedule-container').innerHTML = '';
            document.getElementById('wbs-container').innerHTML = '';
            document.getElementById('milestones-container').innerHTML = '';
            const docDisp = document.getElementById('doc-display-content');
            if (docDisp) docDisp.innerHTML = '';

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

    updateExpectedDeadlineBanner(deadlineStr) {
        const banner = document.getElementById('expected-deadline-banner');
        if (deadlineStr) {
            banner.style.display = 'inline-block';
            const date = new Date(deadlineStr).toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' });
            banner.textContent = `Expected Deadline: ${date}`;
        } else {
            banner.style.display = 'none';
        }
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
        this.activeTab = 'wbs';
    },

    switchTab(tabName) {
        this.activeTab = tabName;
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
        if (tabName === 'strategy') this.loadStrategy();
        if (tabName === 'architecture') this.loadArchitecture();
        if (tabName === 'documentation') this.loadDocumentation();
    },

    async loadWBS() {
        // Actually fetches from estimation endpoint to get effort data too
        const container = document.getElementById('wbs-container');
        container.innerHTML = '<p>Loading WBS...</p>';
        try {
            // Using estimation endpoint since it includes task hours, and wbs tasks to get descriptions
            const [res, wbsTasks] = await Promise.all([
                window.api.getEstimates(this.projectId),
                window.api.getWBSTasks(this.projectId)
            ]);
            container.innerHTML = '';
            
            // 1. Build Global Task ID -> Title Map & Task ID -> Description Map
            const taskIdToTitle = {};
            const taskIdToDesc = {};
            res.groups.forEach(g => g.tasks.forEach(t => taskIdToTitle[t.task_id] = t.task_title));
            wbsTasks.forEach(t => taskIdToDesc[t.id] = t.description);

            res.groups.forEach(g => {
                // 2. Group by Milestone
                const milestoneGroups = {};
                const generalTasks = [];
                
                g.tasks.forEach(t => {
                    const match = t.task_title.match(/^\[(Milestone\s*\d+[^\]]*)\]/i);
                    if (match) {
                        const mName = match[1];
                        if (!milestoneGroups[mName]) milestoneGroups[mName] = [];
                        const cleanTitle = t.task_title.replace(/^\[Milestone\s*\d+[^\]]*\]\s*/i, '');
                        milestoneGroups[mName].push({...t, cleanTitle});
                    } else {
                        generalTasks.push({...t, cleanTitle: t.task_title});
                    }
                });

                let tasksHtml = '';
                
                // Render Milestones
                const sortedMilestoneNames = Object.keys(milestoneGroups).sort((a, b) => {
                    const matchA = a.match(/Milestone\s*(\d+)/i);
                    const matchB = b.match(/Milestone\s*(\d+)/i);
                    const numA = matchA ? parseInt(matchA[1], 10) : 999;
                    const numB = matchB ? parseInt(matchB[1], 10) : 999;
                    return numA - numB;
                });

                sortedMilestoneNames.forEach(mName => {
                    const mTasks = milestoneGroups[mName];
                    tasksHtml += `
                        <div style="margin-left: 1.5rem; margin-top: 0.5rem; border-left: 2px solid var(--color-sedate-gray-light); padding-left: 1rem;">
                            <div style="font-weight: 600; font-size: 0.85rem; color: var(--color-sedate-gray); cursor: pointer;" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                                ▾ ${mName}
                            </div>
                            <div style="display: block;">
                                ${mTasks.map(t => {
                                    const depTitles = (t.dependencies || []).map(id => taskIdToTitle[id] || id).join(', ');
                                    const depHtml = (t.dependencies && t.dependencies.length > 0) 
                                        ? `<span title="Depends on: ${depTitles}" style="cursor: help; margin-left: 0.5rem; font-size: 0.8rem;">🔗</span>` 
                                        : '';
                                    return `
                                    <div class="tree-task" style="margin-left: 0; cursor: pointer; flex-direction: column; align-items: flex-start; gap: 0.25rem;" onclick="const desc = this.querySelector('.task-desc-detail'); desc.style.display = desc.style.display === 'none' ? 'block' : 'none'; event.stopPropagation();">
                                        <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
                                            <span style="font-weight: 500;">${t.cleanTitle}${depHtml}</span>
                                            <span style="color: var(--text-muted); font-size: 0.8rem;">${t.effort_hours}h</span>
                                        </div>
                                        <div class="task-desc-detail" style="display: none; font-size: 0.8rem; color: var(--text-muted); padding: 0.5rem; background: rgba(0,0,0,0.02); border-left: 2px solid var(--color-sedate-gray); width: 100%; border-radius: var(--radius-sm); box-sizing: border-box; margin-top: 0.25rem;">
                                            ${taskIdToDesc[t.task_id] || 'No further description available.'}
                                        </div>
                                    </div>
                                    `
                                }).join('')}
                            </div>
                        </div>
                    `;
                });

                // Render General Tasks (if any)
                if (generalTasks.length > 0) {
                    tasksHtml += generalTasks.map(t => {
                        const depTitles = (t.dependencies || []).map(id => taskIdToTitle[id] || id).join(', ');
                        const depHtml = (t.dependencies && t.dependencies.length > 0) 
                            ? `<span title="Depends on: ${depTitles}" style="cursor: help; margin-left: 0.5rem; font-size: 0.8rem;">🔗</span>` 
                            : '';
                        return `
                        <div class="tree-task" style="cursor: pointer; flex-direction: column; align-items: flex-start; gap: 0.25rem;" onclick="const desc = this.querySelector('.task-desc-detail'); desc.style.display = desc.style.display === 'none' ? 'block' : 'none'; event.stopPropagation();">
                            <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
                                <span style="font-weight: 500;">${t.cleanTitle}${depHtml}</span>
                                <span style="color: var(--text-muted); font-size: 0.8rem;">${t.effort_hours}h</span>
                            </div>
                            <div class="task-desc-detail" style="display: none; font-size: 0.8rem; color: var(--text-muted); padding: 0.5rem; background: rgba(0,0,0,0.02); border-left: 2px solid var(--color-sedate-gray); width: 100%; border-radius: var(--radius-sm); box-sizing: border-box; margin-top: 0.25rem;">
                                ${taskIdToDesc[t.task_id] || 'No further description available.'}
                            </div>
                        </div>
                        `
                    }).join('');
                }

                const epicHtml = `
                    <div class="tree-epic">
                        <div class="tree-epic-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                            <span>▾ ${g.group_name}</span>
                            <span class="badge badge-neutral">${g.total_effort_hours} hrs</span>
                        </div>
                        <div style="display: block;">
                            ${tasksHtml}
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
            
            if (res.required_headcount) {
                container.innerHTML += `
                    <div style="background: #e9ecef; color: #495057; padding: 0.75rem 1rem; border-radius: var(--radius-md); font-size: 0.85rem; font-weight: 500; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem; border-left: 4px solid var(--color-sage);">
                        💡 <strong>AI Capacity Calculation:</strong> To meet the deadline with the estimated task volume, the schedule has been automatically leveled for a minimum of <strong>${res.required_headcount} concurrent developers</strong>.
                    </div>
                `;
            }
            
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

    async loadStrategy() {
        const container = document.getElementById('strategy-container');
        if (container.innerHTML !== '') return;

        container.innerHTML = '<p>Loading Product Strategy...</p>';
        try {
            if (!this.projectIntelligence) {
                this.projectIntelligence = await window.api.getIntelligence(this.projectId);
            }
            const pi = this.projectIntelligence;
            const strat = pi.productStrategy || {};
            const feats = pi.featureSpecs || {};

            if (!strat.productDefinition && (!feats.coreFeatures || feats.coreFeatures.length === 0)) {
                container.innerHTML = '<div class="card" style="text-align:center; color: var(--text-muted);">No product strategy details generated yet. Run the pipeline first.</div>';
                return;
            }

            const coreList = (feats.coreFeatures || []).map(f => `
                <div style="background: var(--bg-main); border: 1px solid var(--color-sedate-gray-light); padding: 1rem; border-radius: var(--radius-md); margin-bottom: 0.5rem;">
                    <div style="font-weight:600; font-size:0.95rem;">${f.name}</div>
                    <div style="font-size:0.85rem; color:var(--text-muted); margin-top:0.25rem;">${f.description}</div>
                </div>
            `).join('');

            const advList = (feats.advancedFeatures || []).map(f => `
                <div style="background: var(--bg-main); border: 1px solid var(--color-sedate-gray-light); padding: 1rem; border-radius: var(--radius-md); margin-bottom: 0.5rem;">
                    <div style="font-weight:600; font-size:0.95rem;">${f.name} <span class="badge badge-neutral" style="font-size:0.7rem; padding:0.1rem 0.4rem;">${f.priority}</span></div>
                    <div style="font-size:0.85rem; color:var(--text-muted); margin-top:0.25rem;">${f.description}</div>
                </div>
            `).join('');

            container.innerHTML = `
                <div style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.5rem;">
                    <div class="card">
                        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                            <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">Product Vision & Strategy</h4>
                            <div style="display:flex; gap:0.25rem;">
                                <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'productStrategy')">✨ Enhance</button>
                                <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'productStrategy')">↩️ Revert</button>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 1.5rem;">
                            <h5 style="color:var(--color-sage); margin-bottom:0.5rem;">Product Definition</h5>
                            <p style="font-size:0.9rem; line-height:1.5;">${strat.productDefinition || 'N/A'}</p>
                        </div>
                        
                        <div style="margin-bottom: 1.5rem;">
                            <h5 style="color:var(--color-sage); margin-bottom:0.5rem;">Core Value Proposition</h5>
                            <p style="font-size:0.9rem; line-height:1.5;">${strat.valueProposition || 'N/A'}</p>
                        </div>

                        <div style="margin-bottom: 1.5rem;">
                            <h5 style="color:var(--color-sage); margin-bottom:0.5rem;">Market Differentiation</h5>
                            <p style="font-size:0.9rem; line-height:1.5;">${strat.differentiation || 'N/A'}</p>
                        </div>

                        <div>
                            <h5 style="color:var(--color-sage); margin-bottom:0.5rem;">Target Market & Positioning</h5>
                            <p style="font-size:0.9rem; line-height:1.5;">${strat.marketPositioning || 'N/A'}</p>
                        </div>
                    </div>

                    <div>
                        <div class="card" style="margin-bottom:1.5rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1rem;">
                                <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">Core Scope (Must Have)</h4>
                                <div style="display:flex; gap:0.25rem;">
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'featureSpecs')">✨ Enhance Specs</button>
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'featureSpecs')">↩️ Revert</button>
                                </div>
                            </div>
                            ${coreList || '<p style="color:var(--text-muted); font-size:0.85rem;">No core features defined</p>'}
                        </div>
                        
                        <div class="card">
                            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1rem;">
                                <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">Advanced Features</h4>
                                <div style="display:flex; gap:0.25rem;">
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'featureSpecs')">✨ Enhance Specs</button>
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'featureSpecs')">↩️ Revert</button>
                                </div>
                            </div>
                            ${advList || '<p style="color:var(--text-muted); font-size:0.85rem;">No advanced features defined</p>'}
                        </div>
                    </div>
                </div>
            `;
        } catch(e) {
            container.innerHTML = '<div class="card" style="text-align:center; color: var(--text-muted);">No Product Strategy available.</div>';
        }
    },

    async loadArchitecture() {
        const container = document.getElementById('architecture-container');
        if (container.innerHTML !== '') return;

        container.innerHTML = '<p>Loading Architecture & Design...</p>';
        try {
            if (!this.projectIntelligence) {
                this.projectIntelligence = await window.api.getIntelligence(this.projectId);
            }
            const pi = this.projectIntelligence;
            const arch = pi.systemArchitecture || {};
            const dbDesign = pi.databaseDesign || {};
            const apiDesign = pi.apiDesign || {};
            const sec = pi.security || {};

            if (!arch.highLevel && !arch.microservices) {
                container.innerHTML = '<div class="card" style="text-align:center; color: var(--text-muted);">No system architecture generated yet. Run the pipeline first.</div>';
                return;
            }

            // Build microservices rows
            const serviceRows = (arch.microservices || []).map(s => `
                <tr>
                    <td><strong>${s.name}</strong></td>
                    <td>${(s.techStack || []).join(', ')}</td>
                    <td>${s.responsibility}</td>
                </tr>
            `).join('');

            // Build Database Design display
            let entities = dbDesign.entities || [];
            if (entities.length === 0 && dbDesign.entityModeling && dbDesign.entityModeling.length > 0) {
                entities = dbDesign.entityModeling.map(e => ({
                    name: e.entity || e.name || 'Table',
                    description: e.description || '',
                    fields: (e.attributes || e.fields || []).map(f => ({
                        name: f.name || 'field',
                        type: f.type || 'text',
                        constraints: f.constraints || ''
                    }))
                }));
            }

            let dbHtml = '<p style="font-size:0.9rem; color:var(--text-muted);">No database schemas defined.</p>';
            if (entities && entities.length > 0) {
                dbHtml = entities.map(e => `
                    <div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-main); border: 1px solid var(--color-sedate-gray-light); border-radius: var(--radius-md);">
                        <div style="font-weight:600; font-size:0.95rem; margin-bottom:0.5rem; color:var(--color-dusty-rose);">Table: ${e.name}</div>
                        <div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.5rem;">${e.description || ''}</div>
                        <table style="width:100%; border-collapse:collapse; font-size:0.8rem;">
                            <thead>
                                <tr style="border-bottom:1px solid var(--color-sedate-gray-light); text-align:left;">
                                    <th style="padding:0.25rem;">Field</th>
                                    <th style="padding:0.25rem;">Type</th>
                                    <th style="padding:0.25rem;">Constraints</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${(e.fields || []).map(f => `
                                    <tr style="border-bottom:1px dashed rgba(0,0,0,0.05);">
                                        <td style="padding:0.25rem;"><code>${f.name}</code></td>
                                        <td style="padding:0.25rem; color:var(--text-muted);">${f.type}</td>
                                        <td style="padding:0.25rem; color:var(--text-muted);">${f.constraints || ''}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `).join('');
            } else if (typeof dbDesign === 'object' && Object.keys(dbDesign).length > 0) {
                dbHtml = `<pre style="font-size:0.85rem; background:var(--bg-main); padding:1rem; border-radius:4px; overflow-x:auto;">${JSON.stringify(dbDesign, null, 2)}</pre>`;
            }

            // Build API Design display
            let apiHtml = '<p style="font-size:0.9rem; color:var(--text-muted);">No APIs defined.</p>';
            if (apiDesign.endpoints && apiDesign.endpoints.length > 0) {
                apiHtml = apiDesign.endpoints.map(ep => `
                    <div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-main); border: 1px solid var(--color-sedate-gray-light); border-radius: var(--radius-md);">
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                            <span style="font-weight:600; font-size:0.95rem;"><code>${ep.path}</code></span>
                            <span class="badge badge-success" style="font-size:0.75rem;">${ep.method}</span>
                        </div>
                        <div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.5rem;">${ep.description}</div>
                        ${ep.requestPayload ? `<div style="font-size:0.8rem; margin-top:0.25rem;"><strong>Request:</strong> <code>${JSON.stringify(ep.requestPayload)}</code></div>` : ''}
                        ${ep.responsePayload ? `<div style="font-size:0.8rem; margin-top:0.25rem;"><strong>Response:</strong> <code>${JSON.stringify(ep.responsePayload)}</code></div>` : ''}
                    </div>
                `).join('');
            } else if (typeof apiDesign === 'object' && Object.keys(apiDesign).length > 0) {
                apiHtml = `<pre style="font-size:0.85rem; background:var(--bg-main); padding:1rem; border-radius:4px; overflow-x:auto;">${JSON.stringify(apiDesign, null, 2)}</pre>`;
            }

            // Build Security
            let secHtml = '<p style="font-size:0.9rem; color:var(--text-muted);">No security specs defined.</p>';
            if (sec.authentication || sec.authorization || sec.dataProtection || sec.authStrategy) {
                secHtml = `
                    <div style="font-size:0.9rem; line-height:1.5;">
                        <h5 style="margin-bottom:0.25rem; color:var(--color-sage);">Authentication</h5>
                        <p style="margin-bottom:1rem; color:var(--text-muted);">${sec.authentication || sec.authStrategy || 'N/A'}</p>
                        
                        <h5 style="margin-bottom:0.25rem; color:var(--color-sage);">Authorization</h5>
                        <p style="margin-bottom:1rem; color:var(--text-muted);">${sec.authorization || sec.authStrategy || 'N/A'}</p>

                        <h5 style="margin-bottom:0.25rem; color:var(--color-sage);">Data Protection & Encryption</h5>
                        <p style="color:var(--text-muted);">${sec.dataProtection || 'N/A'}</p>
                    </div>
                `;
            } else if (typeof sec === 'object' && Object.keys(sec).length > 0) {
                secHtml = `<pre style="font-size:0.85rem; background:var(--bg-main); padding:1rem; border-radius:4px; overflow-x:auto;">${JSON.stringify(sec, null, 2)}</pre>`;
            }

            // Render Layout
            container.innerHTML = `
                <div style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.5rem;">
                    <div>
                        <!-- System Flow Diagram -->
                        ${arch.mermaidSystemFlow ? `
                            <div class="card" style="margin-bottom: 1.5rem;">
                                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                                    <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">System Flow Diagram</h4>
                                    <div style="display:flex; gap:0.25rem;">
                                        <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'systemArchitecture')">✨ Enhance</button>
                                        <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'systemArchitecture')">↩️ Revert</button>
                                    </div>
                                </div>
                                <div class="mermaid" style="display:flex; justify-content:center; padding:1rem; background:white; border-radius:var(--radius-md); overflow-x:auto;">
                                    ${this.cleanMermaidFlow(arch.mermaidSystemFlow)}
                                </div>
                            </div>
                        ` : ''}

                        <!-- Microservices Table -->
                        <div class="card" style="padding: 0; overflow: hidden; margin-bottom:1.5rem;">
                            <div style="padding: 1.5rem 1.5rem 0 1.5rem;">
                                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                                    <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">Functionality stack</h4>
                                    <div style="display:flex; gap:0.25rem;">
                                        <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'systemArchitecture')">✨ Enhance</button>
                                        <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'systemArchitecture')">↩️ Revert</button>
                                    </div>
                                </div>
                            </div>
                            <table class="table" style="margin: 0;">
                                <thead>
                                    <tr>
                                        <th>Component / Service</th>
                                        <th>Tech Stack</th>
                                        <th>Responsibility</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${serviceRows || '<tr><td colspan="3">No sub-systems or services defined</td></tr>'}
                                </tbody>
                            </table>
                        </div>

                        <!-- Database Design -->
                        <div class="card">
                            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                                <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">Database Design & Schemas</h4>
                                <div style="display:flex; gap:0.25rem;">
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'databaseDesign')">✨ Enhance</button>
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'databaseDesign')">↩️ Revert</button>
                                </div>
                            </div>
                            ${dbHtml}
                        </div>
                    </div>

                    <div>
                        <!-- High Level Architecture -->
                        <div class="card" style="margin-bottom: 1.5rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                                <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">Architecture Overview</h4>
                                <div style="display:flex; gap:0.25rem;">
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'systemArchitecture')">✨ Enhance</button>
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'systemArchitecture')">↩️ Revert</button>
                                </div>
                            </div>
                            <p style="font-size:0.95rem; line-height:1.5; font-weight:500; color:var(--color-sedate-gray);">${arch.highLevel || 'N/A'}</p>
                        </div>

                        <!-- API Design -->
                        <div class="card" style="margin-bottom:1.5rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                                <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">REST API Reference</h4>
                                <div style="display:flex; gap:0.25rem;">
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'apiDesign')">✨ Enhance</button>
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'apiDesign')">↩️ Revert</button>
                                </div>
                            </div>
                            ${apiHtml}
                        </div>

                        <!-- Security -->
                        <div class="card">
                            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                                <h4 style="margin:0; color: var(--color-sedate-gray); font-size:1.1rem;">Security & Compliance</h4>
                                <div style="display:flex; gap:0.25rem;">
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('intelligence', 'security')">✨ Enhance</button>
                                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('intelligence', 'security')">↩️ Revert</button>
                                </div>
                            </div>
                            ${secHtml}
                        </div>
                    </div>
                </div>
            `;

            // Initialize Mermaid Flowcharts
            setTimeout(() => {
                if (window.mermaid) {
                    try {
                        window.mermaid.init(undefined, document.querySelectorAll('.mermaid'));
                    } catch(err) {
                        console.error("Mermaid rendering failed:", err);
                    }
                }
            }, 100);

        } catch(e) {
            container.innerHTML = '<div class="card" style="text-align:center; color: var(--text-muted);">No Architecture & Design specification available.</div>';
        }
    },

    async loadChatHistory() {
        try {
            const history = await window.api.getChatHistory(this.projectId);
            const chatBox = document.getElementById('chat-messages');
            if (!chatBox) return;

            // Retain the default greeting message
            chatBox.innerHTML = `<div class="msg msg-ai">Hi! I am your Project Context Assistant. Ask me anything about this project's requirements, technologies, system flow, database design, or constraints.</div>`;
            
            history.forEach(m => {
                if (m.role === 'user') {
                    chatBox.innerHTML += `<div class="msg msg-user">${m.content}</div>`;
                } else {
                    chatBox.innerHTML += `<div class="msg msg-ai">${this.parseMarkdown(m.content)}</div>`;
                }
            });
            chatBox.scrollTop = chatBox.scrollHeight;
        } catch (e) {
            console.error('Failed to load chat history:', e);
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
            const res = await window.api.sendChatMessage(this.projectId, query);
            
            let aiText = "";
            if (res.answer) {
                aiText += `<div style="margin-bottom: 0.75rem; line-height: 1.4;">${this.parseMarkdown(res.answer)}</div>`;
            }

            if (res.results && res.results.length > 0) {
                const randomId = 'sources_' + Math.random().toString(36).substr(2, 9);
                aiText += `<details style="margin-top: 0.5rem; font-size: 0.8rem; border-top: 1px dashed var(--color-sedate-gray-light); padding-top: 0.5rem;">
                    <summary style="cursor: pointer; color: var(--color-sky-blue); font-weight: bold;">View Sources (${res.results.length})</summary>
                    <div id="${randomId}" style="margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.5rem;">`;
                
                res.results.forEach(r => {
                    aiText += `<div style="padding: 0.5rem; background: var(--bg-card); border: 1px solid var(--color-sedate-gray-light); border-radius: 4px;">
                        <strong>${r.title}</strong> <span style="font-size: 0.75rem; color: var(--text-muted); float: right;">(Score: ${r.similarity_score})</span><br>
                        <span style="font-style: italic; color: var(--text-muted);">"${r.content_snippet}"</span>
                    </div>`;
                });
                
                aiText += `</div></details>`;
            } else {
                if (!res.answer) {
                    aiText = "I couldn't find any relevant context for that.";
                }
            }

            chatBox.innerHTML += `<div class="msg msg-ai">${aiText}</div>`;
        } catch (error) {
            chatBox.innerHTML += `<div class="msg msg-ai" style="color: red;">Search failed.</div>`;
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    },

    loadedDocs: {},
    docPollInterval: null,

    async loadDocumentation() {
        if (this.docPollInterval) {
            clearInterval(this.docPollInterval);
            this.docPollInterval = null;
        }

        const pollDocs = async () => {
            try {
                const docs = await window.api.getPlanDocs(this.projectId);
                this.loadedDocs = {};
                let stillCompiling = false;

                docs.forEach(doc => {
                    this.loadedDocs[doc.title] = doc.content;
                    if (!this.docObjects) this.docObjects = {};
                    this.docObjects[doc.title] = doc.id;
                    if (doc.content === "Generating...") {
                        stillCompiling = true;
                    }
                });

                const navItems = {
                    'Core Technical Blueprint': 'nav-blueprint',
                    'Project Constraints': 'nav-constraints',
                    'Success Metrics Framework': 'nav-metrics',
                    'Assumptions Register': 'nav-assumptions',
                    'Decision Brief': 'nav-decision'
                };

                Object.keys(navItems).forEach(title => {
                    const elId = navItems[title];
                    const navEl = document.getElementById(elId);
                    if (navEl) {
                        const statusEl = navEl.querySelector('.doc-status');
                        if (statusEl) {
                            const content = this.loadedDocs[title];
                            if (!content) {
                                statusEl.textContent = 'Queueing...';
                                statusEl.style.color = 'var(--text-muted)';
                                stillCompiling = true;
                            } else if (content === 'Generating...') {
                                statusEl.innerHTML = '<span class="spinner" style="display:inline-block; width:10px; height:10px; border:2px solid var(--color-dusty-rose); border-radius:50%; border-top-color:transparent; animation:spin 1s linear infinite;"></span> Generating...';
                                statusEl.style.color = 'var(--color-dusty-rose)';
                                stillCompiling = true;
                            } else {
                                statusEl.textContent = 'Ready';
                                statusEl.style.color = 'var(--color-sage)';
                            }
                        }
                    }
                });

                const activeNav = document.querySelector('.doc-nav-item.active');
                if (activeNav) {
                    const activeTitle = activeNav.querySelector('span').textContent.trim();
                    const activeContent = this.loadedDocs[activeTitle];
                    const container = document.getElementById('doc-display-content');
                    if (activeContent && activeContent !== 'Generating...') {
                        if (container.querySelector('.loader-msg') || container.innerHTML.includes('Generating...')) {
                            container.innerHTML = this.parseMarkdown(activeContent);
                        }
                    } else if (activeContent === 'Generating...') {
                        container.innerHTML = `<div class="loader-msg"><h3>${activeTitle}</h3><p style="color: var(--color-dusty-rose); margin-top: 1rem;"><span class="spinner" style="display:inline-block; width:12px; height:12px; border:2px solid var(--color-dusty-rose); border-radius:50%; border-top-color:transparent; animation:spin 1s linear infinite; margin-right: 5px;"></span> AI solutions architect is compiling this document in the background...</p></div>`;
                    }
                }

                if (!stillCompiling && this.docPollInterval) {
                    clearInterval(this.docPollInterval);
                    this.docPollInterval = null;
                }
            } catch (e) {
                console.error('Failed to poll plan documents:', e);
            }
        };

        const container = document.getElementById('doc-display-content');
        container.innerHTML = '<p style="color: var(--text-muted);">Loading Project Documentation...</p>';
        await pollDocs();

        this.switchDoc('Core Technical Blueprint');

        let needsInterval = false;
        const requiredDocs = ['Core Technical Blueprint', 'Project Constraints', 'Success Metrics Framework', 'Assumptions Register', 'Decision Brief'];
        requiredDocs.forEach(t => {
            const content = this.loadedDocs[t];
            if (!content || content === 'Generating...') {
                needsInterval = true;
            }
        });

        if (needsInterval) {
            this.docPollInterval = setInterval(pollDocs, 4000);
        }
    },

    switchDoc(title) {
        document.querySelectorAll('.doc-nav-item').forEach(el => {
            el.classList.remove('active');
            const spanText = el.querySelector('span').textContent.trim();
            if (spanText === title) {
                el.classList.add('active');
            }
        });

        const container = document.getElementById('doc-display-content');
        const docContent = this.loadedDocs[title];
        if (!docContent) {
            container.innerHTML = `<h3>${title}</h3><p style="color: var(--text-muted); margin-top: 1rem;">This document is queued for generation. Please wait...</p>`;
            return;
        }

        if (docContent === 'Generating...') {
            container.innerHTML = `<div class="loader-msg"><h3>${title}</h3><p style="color: var(--color-dusty-rose); margin-top: 1rem;"><span class="spinner" style="display:inline-block; width:12px; height:12px; border:2px solid var(--color-dusty-rose); border-radius:50%; border-top-color:transparent; animation:spin 1s linear infinite; margin-right: 5px;"></span> AI solutions architect is compiling this document in the background...</p></div>`;
            return;
        }

        const docId = this.docObjects ? this.docObjects[title] : '';
        const headerHtml = `
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid var(--color-sedate-gray-light); padding-bottom:0.5rem; margin-bottom:1.5rem;">
                <h3 style="margin:0;">${title}</h3>
                ${docId ? `
                <div style="display:flex; gap:0.25rem;">
                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardEnhance('documentation', '${docId}')">✨ Enhance Document</button>
                    <button class="btn btn-neutral" style="font-size:0.75rem; padding:0.25rem 0.5rem; cursor:pointer; background:transparent; border:1px solid var(--color-sedate-gray-light); border-radius:var(--radius-sm);" onclick="workspace.triggerCardRevert('documentation', '${docId}')">↩️ Revert</button>
                </div>
                ` : ''}
            </div>
        `;
        container.innerHTML = headerHtml + this.parseMarkdown(docContent);
    },

    parseMarkdown(markdown) {
        if (!markdown) return '';
        let html = markdown;

        // Escape HTML
        html = html
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Headings
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Code blocks
        html = html.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Horizontal Rule
        html = html.replace(/^---$/gim, '<hr style="border: 0; border-top: 1px solid var(--color-sedate-gray-light); margin: 1.5rem 0;">');

        // Tables
        const lines = html.split('\n');
        let inTable = false;
        let tableHTML = '';
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].trim();
            if (line.startsWith('|')) {
                if (!inTable) {
                    inTable = true;
                    tableHTML = '<table class="table">';
                }
                const cells = line.split('|').slice(1, -1);
                const isHeaderDivider = cells.every(c => c.trim().startsWith('-'));
                if (isHeaderDivider) {
                    lines[i] = '';
                    continue;
                }
                
                const isHeaderRow = (tableHTML === '<table class="table">');
                tableHTML += '<tr>';
                cells.forEach(cell => {
                    const tag = isHeaderRow ? 'th' : 'td';
                    tableHTML += `<${tag}>${cell.trim()}</${tag}>`;
                });
                tableHTML += '</tr>';
                lines[i] = '';
            } else {
                if (inTable) {
                    inTable = false;
                    tableHTML += '</table>';
                    lines[i - 1] = tableHTML;
                }
            }
        }
        if (inTable) {
            tableHTML += '</table>';
            lines[lines.length - 1] = tableHTML;
        }
        html = lines.filter(l => l !== '').join('\n');

        // Bullet Lists
        html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
        html = html.replace(/^\s*\*\s+(.*$)/gim, '<li>$1</li>');
        
        // Wrap bullet lists
        html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

        // Line breaks
        html = html.replace(/\n\n/g, '<p></p>');

        return html;
    },

    triggerCardEnhance(type, key) {
        let currentContent = '';
        if (type === 'intelligence') {
            const data = this.projectIntelligence ? this.projectIntelligence[key] : null;
            currentContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        } else if (type === 'documentation') {
            let title = 'Core Technical Blueprint';
            if (this.docObjects) {
                for (const t of Object.keys(this.docObjects)) {
                    if (this.docObjects[t] === key) {
                        title = t;
                        break;
                    }
                }
            }
            currentContent = this.loadedDocs ? this.loadedDocs[title] : '';
        }

        document.getElementById('enhance-section-type').value = type;
        document.getElementById('enhance-section-key').value = key;
        document.getElementById('enhance-instruction-input').value = '';
        document.getElementById('enhance-modal').style.display = 'flex';
    },

    async submitEnhancement() {
        const sectionType = document.getElementById('enhance-section-type').value;
        const sectionKey = document.getElementById('enhance-section-key').value;
        const instruction = document.getElementById('enhance-instruction-input').value.trim();

        if (!instruction) {
            alert("Please enter editing instructions.");
            return;
        }

        const submitBtn = document.getElementById('enhance-submit-btn');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Regenerating...';

        try {
            let currentContent = '';
            if (sectionType === 'intelligence') {
                const data = this.projectIntelligence ? this.projectIntelligence[sectionKey] : null;
                currentContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
            } else if (sectionType === 'documentation') {
                let title = 'Core Technical Blueprint';
                if (this.docObjects) {
                    for (const t of Object.keys(this.docObjects)) {
                        if (this.docObjects[t] === sectionKey) {
                            title = t;
                            break;
                        }
                    }
                }
                currentContent = this.loadedDocs ? this.loadedDocs[title] : '';
            } else if (sectionType === 'wbs') {
                const res = await window.api.getWBSTasks(this.projectId);
                currentContent = JSON.stringify(res, null, 2);
            } else if (sectionType === 'resources') {
                const res = await window.api.getResourcePlan(this.projectId);
                currentContent = JSON.stringify(res.allocations, null, 2);
            }

            const res = await window.api.enhanceSection(this.projectId, {
                section_type: sectionType,
                section_key: sectionKey,
                user_instruction: instruction,
                current_content: currentContent
            });

            if (res.status === 'success') {
                alert("Section enhanced successfully!");
                document.getElementById('enhance-modal').style.display = 'none';
                
                // Reload project intelligence block in memory to sync new values
                if (sectionType === 'intelligence') {
                    this.projectIntelligence = await window.api.getIntelligence(this.projectId);
                }

                // Force refresh current view
                const tab = this.activeTab || 'wbs';
                if (tab === 'strategy') {
                    document.getElementById('strategy-container').innerHTML = '';
                    await this.loadStrategy();
                } else if (tab === 'architecture') {
                    document.getElementById('architecture-container').innerHTML = '';
                    await this.loadArchitecture();
                } else if (tab === 'documentation') {
                    await this.loadDocumentation();
                } else if (tab === 'wbs' || tab === 'schedule' || tab === 'resources') {
                    document.getElementById('wbs-container').innerHTML = '';
                    document.getElementById('schedule-container').innerHTML = '';
                    const tbody = document.getElementById('resources-tbody');
                    if (tbody) tbody.innerHTML = '';
                    if (tab === 'wbs') await this.loadWBS();
                    if (tab === 'schedule') await this.loadSchedule();
                    if (tab === 'resources') await this.loadResources();
                }
            } else {
                alert("Failed to enhance section: " + res.detail);
            }
        } catch (e) {
            console.error(e);
            alert("Error enhancing section: " + e.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Enhance';
        }
    },

    async triggerCardRevert(type, key) {
        if (!confirm("Are you sure you want to revert this section to its original generated state? Any custom enhancements to this section will be discarded.")) {
            return;
        }

        try {
            const res = await window.api.revertSection(this.projectId, {
                section_type: type,
                section_key: key
            });

            if (res.status === 'success') {
                alert("Section reverted to original state successfully!");
                
                // Reload project intelligence block in memory to sync restored values
                if (type === 'intelligence') {
                    this.projectIntelligence = await window.api.getIntelligence(this.projectId);
                }

                // Force refresh current view
                const tab = this.activeTab || 'wbs';
                if (tab === 'strategy') {
                    document.getElementById('strategy-container').innerHTML = '';
                    await this.loadStrategy();
                } else if (tab === 'architecture') {
                    document.getElementById('architecture-container').innerHTML = '';
                    await this.loadArchitecture();
                } else if (tab === 'documentation') {
                    await this.loadDocumentation();
                } else if (tab === 'wbs' || tab === 'schedule' || tab === 'resources') {
                    document.getElementById('wbs-container').innerHTML = '';
                    document.getElementById('schedule-container').innerHTML = '';
                    const tbody = document.getElementById('resources-tbody');
                    if (tbody) tbody.innerHTML = '';
                    if (tab === 'wbs') await this.loadWBS();
                    if (tab === 'schedule') await this.loadSchedule();
                    if (tab === 'resources') await this.loadResources();
                }
            } else {
                alert("Failed to revert section: " + res.message);
            }
        } catch (e) {
            console.error(e);
            alert("Error reverting section: " + e.message);
        }
    },

    cleanMermaidFlow(code) {
        if (!code) return '';
        let cleaned = code.trim();
        
        // Remove code fences
        cleaned = cleaned.replace(/^```mermaid\s*/i, '');
        cleaned = cleaned.replace(/```$/g, '');
        cleaned = cleaned.replace(/```/g, '');
        
        // Split by lines
        const lines = cleaned.split('\n');
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];
            
            // Fix invalid Graphviz/Sequence arrows e.g. ->> to --> or -> to -->
            line = line.replace(/->>/g, '-->');
            line = line.replace(/([^-])->([^-])/g, '$1-->$2');
            
            // Fix invalid trailing > in label bars e.g. |RESTful API|> to |RESTful API|
            line = line.replace(/\|([^|]+)\|\s*>/g, '|$1|');
            
            // Wrap all unquoted brackets [Label] in double quotes to escape special characters like ->
            // Support hyphenated node IDs (e.g. web-server) using [a-zA-Z0-9_-]
            line = line.replace(/([a-zA-Z0-9_-]+)\s*\[([^"\[\]\n]+?)\]/g, '$1["$2"]');
            
            // Wrap all unquoted parentheses (Label) in double quotes
            line = line.replace(/([a-zA-Z0-9_-]+)\s*\(([^"\(\)\n]+?)\)/g, '$1("$2")');
            
            // Fix unquoted subgraph titles with spaces/special characters
            if (line.trim().startsWith("subgraph ")) {
                let parts = line.trim().split(/\s+/);
                if (parts.length > 2) {
                    let rest = parts.slice(1).join(" ");
                    if (!rest.startsWith('"') && !rest.endsWith('"')) {
                        line = `subgraph "${rest}"`;
                    }
                }
            }
            
            lines[i] = line;
        }
        
        return lines.join('\n').trim();
    }
};

document.addEventListener('DOMContentLoaded', () => workspace.init());
window.workspace = workspace;
