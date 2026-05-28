class DashboardManager {
    constructor() {
        this.authScore = 0.0;
        this.confidence = 0.0;
        this.anomalyRisk = 'Low';
        this.securityScore = 85;
        this.behaviorChart = null;
        this.driftChart = null;
        this.patternsChart = null;
        this.timeChart = null;
        this.currentSection = 'dashboard';
        this.securityAlerts = [];
        this.initializeElements();
        this.setupEventListeners();
        this.initializeCharts();
        this.startListening();
    }

    initializeElements() {
        this.navLinks = document.querySelectorAll('.nav-link');
        this.contentSections = document.querySelectorAll('.content-section');
        this.pageTitle = document.getElementById('pageTitle');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.sidebar = document.querySelector('.sidebar');
        this.authStatus = document.getElementById('authStatus');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.notificationBtn = document.getElementById('notificationBtn');
        this.notificationBadge = document.getElementById('notificationBadge');
        this.notificationDropdown = document.getElementById('notificationDropdown');
        this.notificationList = document.getElementById('notificationList');
        this.markAllRead = document.getElementById('markAllRead');
        this.securityScoreEl = document.getElementById('securityScore');
        this.securityScoreCircle = document.getElementById('securityScoreCircle');
        this.authScoreEl = document.getElementById('authScore');
        this.confidenceLevelEl = document.getElementById('confidenceLevel');
        this.anomalyRiskEl = document.getElementById('anomalyRisk');
        this.keystrokeSamplesEl = document.getElementById('keystrokeSamples');
        this.mouseSamplesEl = document.getElementById('mouseSamples');
        this.recentActivityList = document.getElementById('recentActivityList');
        this.activityTableBody = document.getElementById('activityTableBody');
        this.activityFilter = document.getElementById('activityFilter');
        this.dateFilter = document.getElementById('dateFilter');
        this.refreshActivity = document.getElementById('refreshActivity');
        this.runSecurityCheck = document.getElementById('runSecurityCheck');
        this.exportLogs = document.getElementById('exportLogs');
        this.enableRealTimeAuth = document.getElementById('enableRealTimeAuth');
        this.enableAnomalyAlerts = document.getElementById('enableAnomalyAlerts');
        this.authThreshold = document.getElementById('authThreshold');
        this.anomalySensitivity = document.getElementById('anomalySensitivity');
        this.securityAlertModal = document.getElementById('securityAlertModal');
        this.alertTitle = document.getElementById('alertTitle');
        this.alertMessage = document.getElementById('alertMessage');
        this.alertDetails = document.getElementById('alertDetails');
        this.acknowledgeAlert = document.getElementById('acknowledgeAlert');
        this.softLockOverlay = document.getElementById('softLockOverlay');
        this.pinSection = document.getElementById('pinSection');
        this.pinInput = document.getElementById('pinInput');
        this.pinSubmitBtn = document.getElementById('pinSubmitBtn');
        this.pinError = document.getElementById('pinError');
        this.overrideSection = document.getElementById('overrideSection');
        this.tiredModeBtn = document.getElementById('tiredModeBtn');
        this.guestModeBtn = document.getElementById('guestModeBtn');
        this.guestDuration = document.getElementById('guestDuration');
        this.activeOverrideStatus = document.getElementById('activeOverrideStatus');
        this.overrideStatusText = document.getElementById('overrideStatusText');
        this.newPinInput = document.getElementById('newPinInput');
        this.confirmPinInput = document.getElementById('confirmPinInput');
        this.savePinBtn = document.getElementById('savePinBtn');
        this.clearPinBtn = document.getElementById('clearPinBtn');
        this.pinSetupStatus = document.getElementById('pinSetupStatus');
    }

    setupEventListeners() {
        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.showSection(link.dataset.section);
            });
        });
        this.sidebarToggle.addEventListener('click', () => this.sidebar.classList.toggle('open'));
        this.notificationBtn.addEventListener('click', () => this.toggleNotificationDropdown());
        this.markAllRead.addEventListener('click', () => this.markAllNotificationsRead());
        this.runSecurityCheck.addEventListener('click', () => this.runSecurityCheck());
        this.exportLogs.addEventListener('click', () => this.exportLogs());
        this.refreshActivity.addEventListener('click', () => this.loadActivityLog());
        this.activityFilter.addEventListener('change', () => this.loadActivityLog());
        this.dateFilter.addEventListener('change', () => this.loadActivityLog());
        this.authThreshold.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
            this.updateSettings();
        });
        this.anomalySensitivity.addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = e.target.value;
            this.updateSettings();
        });
        this.acknowledgeAlert.addEventListener('click', () => { this.securityAlertModal.style.display = 'none'; });
        document.addEventListener('click', (e) => {
            if (!this.notificationBtn.contains(e.target)) this.notificationDropdown.style.display = 'none';
        });
        this.pinSubmitBtn.addEventListener('click', () => this.handlePinSubmit());
        this.pinInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.handlePinSubmit(); });
        this.tiredModeBtn.addEventListener('click', () => this.handleTiredMode());
        this.guestModeBtn.addEventListener('click', () => this.handleGuestMode());
        this.savePinBtn.addEventListener('click', () => this.handleSavePin());
        this.clearPinBtn.addEventListener('click', () => this.handleClearPin());
    }

    startListening() {
        eel.on_auth_update(this.handleAuthResult.bind(this));
        eel.on_security_alert(this.handleSecurityAlert.bind(this));
        eel.on_soft_lock(this.onSoftLock.bind(this));
        setInterval(() => this.pollLockState(), 10000);
    }

    initializeCharts() {
        const behaviorCtx = document.getElementById('behaviorChart');
        if (behaviorCtx) {
            this.behaviorChart = new Chart(behaviorCtx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'Authentication Score', data: [], borderColor: 'rgb(99,102,241)', backgroundColor: 'rgba(99,102,241,0.1)', tension: 0.4, fill: true }] },
                options: { responsive: true, scales: { y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: 'rgba(255,255,255,0.7)' } }, x: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: 'rgba(255,255,255,0.7)' } } }, plugins: { legend: { labels: { color: 'rgba(255,255,255,0.7)' } } } }
            });
        }
        const driftCtx = document.getElementById('driftChart');
        if (driftCtx) {
            this.driftChart = new Chart(driftCtx, {
                type: 'radar',
                data: { labels: ['Typing Speed', 'Key Timing', 'Mouse Velocity', 'Click Patterns', 'Movement Efficiency'], datasets: [{ label: 'Current', data: [0.8, 0.9, 0.7, 0.85, 0.75], borderColor: 'rgb(99,102,241)', backgroundColor: 'rgba(99,102,241,0.2)', pointBackgroundColor: 'rgb(99,102,241)' }, { label: 'Baseline', data: [0.8, 0.8, 0.8, 0.8, 0.8], borderColor: 'rgb(16,185,129)', backgroundColor: 'rgba(16,185,129,0.1)', pointBackgroundColor: 'rgb(16,185,129)' }] },
                options: { responsive: true, scales: { r: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.1)' }, angleLines: { color: 'rgba(255,255,255,0.1)' }, pointLabels: { color: 'rgba(255,255,255,0.7)' }, ticks: { color: 'rgba(255,255,255,0.5)' } } }, plugins: { legend: { labels: { color: 'rgba(255,255,255,0.7)' } } } }
            });
        }
    }

    handleAuthResult(data) {
        this.authScore = data.authenticity_score || 0;
        this.confidence = (data.confidence || 0) * 100;
        this.anomalyRisk = data.anomaly_score < 0.3 ? 'Low' : data.anomaly_score < 0.7 ? 'Medium' : 'High';
        this.updateDisplay();
        this.addChartData(data);
    }

    handleSecurityAlert(data) {
        this.addAlert(data);
        this.updateNotificationBadge();
        if (data.level >= 2) this.showAlertModal(data);
    }

    updateDisplay() {
        if (this.authScoreEl) this.authScoreEl.textContent = this.authScore.toFixed(2);
        if (this.confidenceLevelEl) this.confidenceLevelEl.textContent = Math.round(this.confidence) + '%';
        if (this.anomalyRiskEl) {
            this.anomalyRiskEl.textContent = this.anomalyRisk;
            this.anomalyRiskEl.className = 'metric-value ' + this.anomalyRisk.toLowerCase();
        }
        const newScore = Math.round(this.authScore * 100);
        if (this.securityScoreEl && newScore !== this.securityScore) {
            this.securityScore = newScore;
            this.securityScoreEl.textContent = this.securityScore;
            if (this.securityScoreCircle) this.securityScoreCircle.style.setProperty('--score', this.securityScore);
        }
        const dot = this.statusIndicator ? this.statusIndicator.querySelector('.status-dot') : null;
        if (this.authScore >= 0.8) { if (dot) dot.style.background = 'var(--secondary-color)'; if (this.statusText) this.statusText.textContent = 'Secure'; }
        else if (this.authScore >= 0.6) { if (dot) dot.style.background = 'var(--accent-color)'; if (this.statusText) this.statusText.textContent = 'Monitoring'; }
        else { if (dot) dot.style.background = 'var(--danger-color)'; if (this.statusText) this.statusText.textContent = 'Alert'; }
    }

    addChartData(data) {
        if (this.behaviorChart) {
            const c = this.behaviorChart;
            const now = new Date().toLocaleTimeString();
            c.data.labels.push(now);
            c.data.datasets[0].data.push(data.authenticity_score);
            if (c.data.labels.length > 20) { c.data.labels.shift(); c.data.datasets[0].data.shift(); }
            c.update('none');
        }
        if (this.timeChart) {
            const c = this.timeChart;
            const now = new Date().toLocaleTimeString();
            c.data.labels.push(now);
            c.data.datasets[0].data.push(data.authenticity_score);
            c.data.datasets[1].data.push(data.anomaly_score);
            if (c.data.labels.length > 30) { c.data.labels.shift(); c.data.datasets[0].data.shift(); c.data.datasets[1].data.shift(); }
            c.update('none');
        }
    }

    addAlert(alert) {
        this.securityAlerts.unshift({ ...alert, id: Date.now(), timestamp: new Date(), read: false });
        if (this.securityAlerts.length > 50) this.securityAlerts = this.securityAlerts.slice(0, 50);
        this.updateNotificationDropdown();
    }

    updateNotificationBadge() {
        const unread = this.securityAlerts.filter(a => !a.read).length;
        if (unread > 0) { this.notificationBadge.textContent = unread; this.notificationBadge.style.display = 'block'; }
        else this.notificationBadge.style.display = 'none';
    }

    updateNotificationDropdown() {
        if (this.securityAlerts.length === 0) {
            this.notificationList.innerHTML = '<div class="no-notifications"><i class="fas fa-check-circle"></i><p>No alerts</p></div>';
        } else {
            this.notificationList.innerHTML = this.securityAlerts.slice(0, 10).map(a =>
                `<div class="notification-item ${a.read ? 'read' : 'unread'}">
                    <div class="notification-icon ${this.getLevelClass(a.level)}"><i class="fas ${this.getIcon(a.level)}"></i></div>
                    <div class="notification-content"><div class="notification-title">${a.message}</div><div class="notification-time">${this.formatTime(a.timestamp)}</div></div>
                </div>`
            ).join('');
        }
        this.updateNotificationBadge();
    }

    showAlertModal(alert) {
        this.alertTitle.textContent = `Alert - Level ${alert.level}`;
        this.alertMessage.textContent = alert.message;
        this.alertDetails.innerHTML = `<div class="alert-detail-item"><strong>Confidence:</strong> ${Math.round((alert.confidence || 0) * 100)}%</div><div class="alert-detail-item"><strong>Time:</strong> ${new Date().toLocaleString()}</div>`;
        this.securityAlertModal.style.display = 'flex';
    }

    showSection(sectionId) {
        this.navLinks.forEach(l => l.parentElement.classList.remove('active'));
        const activeLink = document.querySelector(`[data-section="${sectionId}"]`);
        if (activeLink) activeLink.parentElement.classList.add('active');
        this.contentSections.forEach(s => s.classList.remove('active'));
        const activeSection = document.getElementById(`${sectionId}Section`);
        if (activeSection) activeSection.classList.add('active');
        const titles = { dashboard: 'Dashboard', security: 'Security Status', analytics: 'Behavioral Analytics', activity: 'Activity Log', settings: 'Settings' };
        this.pageTitle.textContent = titles[sectionId] || 'Dashboard';
        this.currentSection = sectionId;
        if (sectionId === 'activity') this.loadActivityLog();
    }

    toggleNotificationDropdown() {
        this.notificationDropdown.style.display = this.notificationDropdown.style.display === 'block' ? 'none' : 'block';
    }

    markAllNotificationsRead() {
        this.securityAlerts.forEach(a => a.read = true);
        this.updateNotificationDropdown();
    }

    runSecurityCheck() {
        this.addActivityItem({ type: 'success', message: 'Security check completed', timestamp: new Date().toISOString() });
        eel.run_immediate_auth()();
    }

    exportLogs() {
        const data = 'Locksy activity log exported at ' + new Date().toISOString();
        const blob = new Blob([data], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'locksy-activity.txt'; a.click();
        URL.revokeObjectURL(url);
    }

    loadActivityLog() {
        eel.get_activity_log()().then(events => {
            if (this.activityTableBody) {
                this.activityTableBody.innerHTML = (events || []).map(e =>
                    `<tr><td>${e.time || ''}</td><td>${e.type || ''}</td><td>${e.description || ''}</td><td><span class="risk-level ${(e.risk || 'low').toLowerCase()}">${e.risk || 'Low'}</span></td></tr>`
                ).join('');
            }
        });
    }

    addActivityItem(activity) {
        if (this.recentActivityList) {
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.innerHTML = `<div class="activity-icon ${activity.type}"><i class="fas ${this.getActivityIcon(activity.type)}"></i></div><div class="activity-details"><span class="activity-text">${activity.message}</span><span class="activity-time">${this.formatTime(new Date(activity.timestamp))}</span></div>`;
            this.recentActivityList.insertBefore(item, this.recentActivityList.firstChild);
            while (this.recentActivityList.children.length > 5) this.recentActivityList.removeChild(this.recentActivityList.lastChild);
        }
    }

    updateSettings() { console.log('Settings updated'); }
    getLevelClass(l) { return ['info', 'low', 'medium', 'high', 'critical'][l] || 'info'; }
    getIcon(l) { return ['fa-info', 'fa-exclamation', 'fa-exclamation-triangle', 'fa-exclamation-circle', 'fa-times-circle'][l] || 'fa-info'; }
    getActivityIcon(t) { return { login: 'fa-sign-in-alt', logout: 'fa-sign-out-alt', anomaly: 'fa-exclamation-triangle', drift: 'fa-chart-line', success: 'fa-check' }[t] || 'fa-info'; }
    formatTime(ts) {
        const now = new Date(), d = now - new Date(ts);
        const m = Math.floor(d / 60000);
        if (m < 1) return 'Just now';
        if (m < 60) return `${m}m ago`;
        const h = Math.floor(m / 60);
        if (h < 24) return `${h}h ago`;
        return `${Math.floor(h / 24)}d ago`;
    }

    onSoftLock(data) {
        this.softLockOverlay.style.display = 'flex';
        this.activeOverrideStatus.style.display = 'none';
        this.pinError.style.display = 'none';
        this.overrideSection.style.display = 'none';
        this.pinInput.value = '';
        this.pinInput.focus();

        if (data.has_pin) {
            this.pinSection.style.display = 'block';
        } else {
            this.pinSection.style.display = 'none';
            this.overrideSection.style.display = 'block';
        }
    }

    async handlePinSubmit() {
        const pin = this.pinInput.value.trim();
        if (!pin) return;

        try {
            const valid = await eel.verify_pin(pin)();
            if (valid) {
                this.pinError.style.display = 'none';
                this.pinSection.style.display = 'none';
                this.overrideSection.style.display = 'block';
                await eel.dismiss_soft_lock()();
            } else {
                this.pinError.style.display = 'block';
                this.pinInput.value = '';
                this.pinInput.focus();
            }
        } catch (err) {
            this.pinError.textContent = 'Error verifying PIN.';
            this.pinError.style.display = 'block';
        }
    }

    async handleTiredMode() {
        try {
            const result = await eel.enable_tired_mode()();
            if (result.success) {
                this.softLockOverlay.style.display = 'none';
            }
        } catch (err) {
            console.error('Tired mode error:', err);
        }
    }

    async handleGuestMode() {
        const minutes = parseInt(this.guestDuration.value, 10);
        try {
            const result = await eel.enable_guest_mode(minutes)();
            if (result.success) {
                this.softLockOverlay.style.display = 'none';
            }
        } catch (err) {
            console.error('Guest mode error:', err);
        }
    }

    async pollLockState() {
        try {
            const state = await eel.get_lock_state()();
            if (state.soft_lock_active) {
                this.softLockOverlay.style.display = 'flex';
            }
            if (state.tired_mode_until) {
                const until = new Date(state.tired_mode_until);
                this.activeOverrideStatus.style.display = 'block';
                this.overrideStatusText.textContent = `Tired mode active until ${until.toLocaleTimeString()}`;
            } else if (state.monitoring_paused_until) {
                const until = new Date(state.monitoring_paused_until);
                this.activeOverrideStatus.style.display = 'block';
                this.overrideStatusText.textContent = `Guest mode active until ${until.toLocaleTimeString()}`;
            } else {
                this.activeOverrideStatus.style.display = 'none';
            }
            this.updatePinSettingsUI(state.has_pin);
        } catch (err) {
            console.error('Poll lock state error:', err);
        }
    }

    updatePinSettingsUI(hasPin) {
        if (!this.savePinBtn) return;
        if (hasPin) {
            this.savePinBtn.textContent = 'Update PIN';
            this.clearPinBtn.style.display = 'inline-flex';
        } else {
            this.savePinBtn.textContent = 'Save PIN';
            this.clearPinBtn.style.display = 'none';
        }
    }

    async handleSavePin() {
        const pin = this.newPinInput.value.trim();
        const confirm = this.confirmPinInput.value.trim();

        if (!pin || pin.length < 4) {
            this.pinSetupStatus.textContent = 'PIN must be at least 4 characters.';
            this.pinSetupStatus.className = 'pin-setup-status error';
            return;
        }
        if (pin !== confirm) {
            this.pinSetupStatus.textContent = 'PINs do not match.';
            this.pinSetupStatus.className = 'pin-setup-status error';
            return;
        }

        try {
            const result = await eel.set_local_pin(pin)();
            if (result.success) {
                this.pinSetupStatus.textContent = 'PIN saved successfully.';
                this.pinSetupStatus.className = 'pin-setup-status success';
                this.newPinInput.value = '';
                this.confirmPinInput.value = '';
                this.updatePinSettingsUI(true);
            } else {
                this.pinSetupStatus.textContent = result.error || 'Failed to save PIN.';
                this.pinSetupStatus.className = 'pin-setup-status error';
            }
        } catch (err) {
            this.pinSetupStatus.textContent = 'Error saving PIN.';
            this.pinSetupStatus.className = 'pin-setup-status error';
        }
    }

    async handleClearPin() {
        try {
            const result = await eel.set_local_pin('')();
            if (result.success) {
                this.pinSetupStatus.textContent = 'PIN removed.';
                this.pinSetupStatus.className = 'pin-setup-status success';
                this.updatePinSettingsUI(false);
            }
        } catch (err) {
            this.pinSetupStatus.textContent = 'Error removing PIN.';
            this.pinSetupStatus.className = 'pin-setup-status error';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const dm = new DashboardManager();
    console.log('Locksy Dashboard Initialized');
    window.dashboardManager = dm;
});
