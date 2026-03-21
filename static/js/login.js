/**
 * Login Page JavaScript
 * Handles authentication, registration, and UI interactions
 */

class LoginManager {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.setupPasswordStrengthChecker();
        this.setupFloatingAnimation();
    }

    initializeElements() {
        // Forms
        this.loginForm = document.getElementById('loginForm');
        this.registerForm = document.getElementById('registerForm');
        
        // Buttons
        this.loginBtn = document.getElementById('loginBtn');
        this.registerBtn = document.getElementById('registerBtn');
        this.showRegisterForm = document.getElementById('showRegisterForm');
        this.backToLogin = document.getElementById('backToLogin');
        
        // Password toggles
        this.togglePassword = document.getElementById('togglePassword');
        this.toggleRegPassword = document.getElementById('toggleRegPassword');
        
        // Form fields
        this.usernameField = document.getElementById('username');
        this.passwordField = document.getElementById('password');
        this.regPasswordField = document.getElementById('regPassword');
        this.confirmPasswordField = document.getElementById('confirmPassword');
        
        // Messages
        this.errorMessage = document.getElementById('errorMessage');
        this.successMessage = document.getElementById('successMessage');
        
        // Overlays
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.successOverlay = document.getElementById('successOverlay');
        
        // Password strength elements
        this.strengthBar = document.getElementById('strengthBar');
        this.strengthText = document.getElementById('strengthText');
    }

    setupEventListeners() {
        // Form submissions
        this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        this.registerForm.addEventListener('submit', (e) => this.handleRegister(e));
        
        // Form switching
        this.showRegisterForm.addEventListener('click', () => this.showRegistration());
        this.backToLogin.addEventListener('click', () => this.showLogin());
        
        // Password toggles
        this.togglePassword.addEventListener('click', () => this.togglePasswordVisibility('password'));
        this.toggleRegPassword.addEventListener('click', () => this.togglePasswordVisibility('regPassword'));
        
        // Real-time password strength checking
        this.regPasswordField.addEventListener('input', () => this.checkPasswordStrength());
        this.confirmPasswordField.addEventListener('input', () => this.checkPasswordMatch());
        
        // Enter key handling
        this.usernameField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.passwordField.focus();
        });
        
        this.passwordField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleLogin(e);
        });
    }

    setupPasswordStrengthChecker() {
        this.passwordCriteria = {
            length: { regex: /.{8,}/, weight: 25 },
            lowercase: { regex: /[a-z]/, weight: 15 },
            uppercase: { regex: /[A-Z]/, weight: 15 },
            numbers: { regex: /\d/, weight: 15 },
            symbols: { regex: /[!@#$%^&*(),.?":{}|<>]/, weight: 15 },
            noRepeat: { regex: /^(?!.*(.)\1{2,})/, weight: 15 }
        };
    }

    setupFloatingAnimation() {
        // Add dynamic floating elements
        const container = document.querySelector('.floating-elements');
        if (container) {
            // Create additional floating particles
            for (let i = 6; i <= 15; i++) {
                const element = document.createElement('div');
                element.className = `element element-${i}`;
                element.style.width = `${Math.random() * 40 + 20}px`;
                element.style.height = element.style.width;
                element.style.left = `${Math.random() * 100}%`;
                element.style.top = `${Math.random() * 100}%`;
                element.style.animationDelay = `${Math.random() * 6}s`;
                element.style.opacity = Math.random() * 0.1 + 0.05;
                element.style.background = this.getRandomColor();
                container.appendChild(element);
            }
        }
    }

    getRandomColor() {
        const colors = [
            'var(--primary-color)',
            'var(--secondary-color)',
            'var(--accent-color)',
            'var(--primary-light)'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    async handleLogin(e) {
        e.preventDefault();
        
        const username = this.usernameField.value.trim();
        const password = this.passwordField.value;
        
        if (!username || !password) {
            this.showError('Please fill in all fields');
            return;
        }
        
        this.setLoadingState(this.loginBtn, true);
        this.hideMessages();
        
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Store session data
                localStorage.setItem('access_token', data.access_token);
                localStorage.setItem('session_id', data.session_id);
                localStorage.setItem('user_id', data.user_id);
                localStorage.setItem('username', data.username);
                
                this.showSuccess('Login successful! Redirecting...');
                this.showSuccessOverlay(data.redirect);
                
                // Redirect after animation
                setTimeout(() => {
                    window.location.href = data.redirect;
                }, 2000);
                
            } else {
                this.showError(data.error || 'Login failed');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.setLoadingState(this.loginBtn, false);
        }
    }

    async handleRegister(e) {
        e.preventDefault();
        
        const username = document.getElementById('regUsername').value.trim();
        const email = document.getElementById('regEmail').value.trim();
        const password = this.regPasswordField.value;
        const confirmPassword = this.confirmPasswordField.value;
        const agreeTerms = document.getElementById('agreeTerms').checked;
        
        // Validation
        if (!username || !email || !password || !confirmPassword) {
            this.showError('Please fill in all fields');
            return;
        }
        
        if (password !== confirmPassword) {
            this.showError('Passwords do not match');
            return;
        }
        
        if (!agreeTerms) {
            this.showError('Please agree to the terms and conditions');
            return;
        }
        
        const passwordStrength = this.calculatePasswordStrength(password);
        if (passwordStrength < 60) {
            this.showError('Password is too weak. Please choose a stronger password.');
            return;
        }
        
        this.setLoadingState(this.registerBtn, true);
        this.hideMessages();
        
        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, email, password })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess('Registration successful! Please log in.');
                setTimeout(() => {
                    this.showLogin();
                    this.usernameField.value = username;
                    this.usernameField.focus();
                }, 2000);
            } else {
                this.showError(data.error || 'Registration failed');
            }
        } catch (error) {
            console.error('Registration error:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.setLoadingState(this.registerBtn, false);
        }
    }

    showRegistration() {
        this.loginForm.style.display = 'none';
        this.registerForm.style.display = 'block';
        this.registerForm.style.animation = 'fadeIn 0.5s ease-out';
        
        // Focus first field
        document.getElementById('regUsername').focus();
    }

    showLogin() {
        this.registerForm.style.display = 'none';
        this.loginForm.style.display = 'block';
        this.loginForm.style.animation = 'fadeIn 0.5s ease-out';
        
        // Clear registration form
        this.registerForm.reset();
        this.resetPasswordStrength();
        
        // Focus username field
        this.usernameField.focus();
    }

    togglePasswordVisibility(fieldId) {
        const field = document.getElementById(fieldId);
        const toggle = fieldId === 'password' ? this.togglePassword : this.toggleRegPassword;
        const icon = toggle.querySelector('i');
        
        if (field.type === 'password') {
            field.type = 'text';
            icon.classList.remove('fa-eye');
            icon.classList.add('fa-eye-slash');
        } else {
            field.type = 'password';
            icon.classList.remove('fa-eye-slash');
            icon.classList.add('fa-eye');
        }
    }

    checkPasswordStrength() {
        const password = this.regPasswordField.value;
        const strength = this.calculatePasswordStrength(password);
        this.updatePasswordStrengthUI(strength, password);
    }

    calculatePasswordStrength(password) {
        if (!password) return 0;
        
        let score = 0;
        
        for (const [key, criterion] of Object.entries(this.passwordCriteria)) {
            if (criterion.regex.test(password)) {
                score += criterion.weight;
            }
        }
        
        return Math.min(score, 100);
    }

    updatePasswordStrengthUI(strength, password) {
        const fill = this.strengthBar.querySelector('.strength-fill');
        const text = this.strengthText;
        
        // Update progress bar
        fill.style.width = `${strength}%`;
        
        // Update colors and text
        if (strength === 0) {
            fill.style.background = 'transparent';
            text.textContent = 'Password strength';
            text.style.color = 'var(--text-muted)';
        } else if (strength < 30) {
            fill.style.background = 'var(--danger-color)';
            text.textContent = 'Very weak';
            text.style.color = 'var(--danger-color)';
        } else if (strength < 50) {
            fill.style.background = 'var(--warning-color)';
            text.textContent = 'Weak';
            text.style.color = 'var(--warning-color)';
        } else if (strength < 70) {
            fill.style.background = 'var(--accent-color)';
            text.textContent = 'Fair';
            text.style.color = 'var(--accent-color)';
        } else if (strength < 90) {
            fill.style.background = 'var(--secondary-color)';
            text.textContent = 'Good';
            text.style.color = 'var(--secondary-color)';
        } else {
            fill.style.background = 'var(--primary-color)';
            text.textContent = 'Excellent';
            text.style.color = 'var(--primary-color)';
        }
        
        // Add animation
        fill.style.transition = 'all 0.3s ease-out';
    }

    checkPasswordMatch() {
        const password = this.regPasswordField.value;
        const confirmPassword = this.confirmPasswordField.value;
        
        if (confirmPassword && password !== confirmPassword) {
            this.confirmPasswordField.style.borderColor = 'var(--danger-color)';
        } else {
            this.confirmPasswordField.style.borderColor = 'var(--border-color)';
        }
    }

    resetPasswordStrength() {
        const fill = this.strengthBar.querySelector('.strength-fill');
        const text = this.strengthText;
        
        fill.style.width = '0%';
        fill.style.background = 'transparent';
        text.textContent = 'Password strength';
        text.style.color = 'var(--text-muted)';
    }

    setLoadingState(button, isLoading) {
        const btnText = button.querySelector('.btn-text');
        const btnLoader = button.querySelector('.btn-loader');
        
        if (isLoading) {
            btnText.style.opacity = '0';
            btnLoader.style.display = 'block';
            button.disabled = true;
        } else {
            btnText.style.opacity = '1';
            btnLoader.style.display = 'none';
            button.disabled = false;
        }
    }

    showError(message) {
        this.hideMessages();
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
        this.errorMessage.style.animation = 'fadeIn 0.3s ease-out';
        
        // Auto-hide after 5 seconds
        setTimeout(() => this.hideMessages(), 5000);
    }

    showSuccess(message) {
        this.hideMessages();
        this.successMessage.textContent = message;
        this.successMessage.style.display = 'block';
        this.successMessage.style.animation = 'fadeIn 0.3s ease-out';
    }

    hideMessages() {
        this.errorMessage.style.display = 'none';
        this.successMessage.style.display = 'none';
    }

    showSuccessOverlay(redirectPath) {
        const overlay = this.successOverlay;
        const message = document.getElementById('successRedirectMessage');
        
        if (redirectPath.includes('calibration')) {
            message.textContent = 'Redirecting to behavioral calibration...';
        } else if (redirectPath.includes('challenge')) {
            message.textContent = 'Redirecting to secure dashboard...';
        } else {
            message.textContent = 'Redirecting...';
        }
        
        overlay.style.display = 'flex';
        overlay.style.animation = 'fadeIn 0.5s ease-out';
    }

    // Utility method to handle API responses
    async handleApiResponse(response) {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }

    // Method to check if user is already logged in
    checkExistingSession() {
        const sessionId = localStorage.getItem('session_id');
        const accessToken = localStorage.getItem('access_token');
        
        if (sessionId && accessToken) {
            // Verify session is still valid
            fetch(`/api/session/status?session_id=${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.session_active) {
                        // Redirect to appropriate page
                        if (data.calibration_complete) {
                            window.location.href = '/challenge';
                        } else {
                            window.location.href = '/calibration';
                        }
                    } else {
                        // Clear invalid session data
                        this.clearSessionData();
                    }
                })
                .catch(error => {
                    console.error('Session check error:', error);
                    this.clearSessionData();
                });
        }
    }

    clearSessionData() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('session_id');
        localStorage.removeItem('user_id');
        localStorage.removeItem('username');
    }

    // Initialize keyboard event listeners for accessibility
    initializeAccessibility() {
        // Add keyboard navigation for form switching
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'r') {
                e.preventDefault();
                if (this.loginForm.style.display !== 'none') {
                    this.showRegistration();
                } else {
                    this.showLogin();
                }
            }
        });
        
        // Add focus management
        this.loginForm.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideMessages();
            }
        });
        
        this.registerForm.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.showLogin();
            }
        });
    }

    // Add smooth transitions and micro-interactions
    addMicroInteractions() {
        // Button hover effects
        const buttons = document.querySelectorAll('.btn, .action-button');
        buttons.forEach(button => {
            button.addEventListener('mouseenter', () => {
                button.style.transform = 'translateY(-2px)';
            });
            
            button.addEventListener('mouseleave', () => {
                if (!button.disabled) {
                    button.style.transform = 'translateY(0)';
                }
            });
        });
        
        // Input focus animations
        const inputs = document.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.style.transform = 'scale(1.02)';
                input.parentElement.style.transition = 'transform 0.2s ease-out';
            });
            
            input.addEventListener('blur', () => {
                input.parentElement.style.transform = 'scale(1)';
            });
        });
        
        // Feature item animations
        const featureItems = document.querySelectorAll('.feature-item');
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'fadeIn 0.6s ease-out forwards';
                    entry.target.style.opacity = '1';
                }
            });
        }, observerOptions);
        
        featureItems.forEach(item => {
            item.style.opacity = '0';
            observer.observe(item);
        });
    }
}

// Initialize the login manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const loginManager = new LoginManager();
    
    // Check for existing session
    loginManager.checkExistingSession();
    
    // Initialize accessibility features
    loginManager.initializeAccessibility();
    
    // Add micro-interactions
    loginManager.addMicroInteractions();
    
    // Focus username field
    const usernameField = document.getElementById('username');
    if (usernameField) {
        usernameField.focus();
    }
    
    console.log('🔐 Behavioral Authentication System - Login Page Initialized');
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        // Re-check session when page becomes visible
        const sessionId = localStorage.getItem('session_id');
        if (sessionId) {
            fetch(`/api/session/status?session_id=${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.session_active) {
                        // Session is still active, check if we should redirect
                        if (window.location.pathname === '/login' || window.location.pathname === '/') {
                            if (data.calibration_complete) {
                                window.location.href = '/challenge';
                            } else {
                                window.location.href = '/calibration';
                            }
                        }
                    }
                })
                .catch(error => console.error('Session check error:', error));
        }
    }
});

// Export for potential use in other modules
window.LoginManager = LoginManager;