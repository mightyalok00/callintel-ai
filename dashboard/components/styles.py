"""
Design System and Custom CSS for AI Call Intelligence Dashboard.
Implements dark mode glassmorphism, modern typography, and glowing status badges.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #f1f5f9;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* App Background */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(30, 27, 75, 0.4) 0%, rgba(10, 15, 30, 0.95) 90%), #070913;
}

/* Glassmorphic Cards */
.glass-card {
    background: rgba(18, 24, 38, 0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-2px);
}

/* Gradient Header Text */
.gradient-text {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* KPI Stat Cards */
.kpi-container {
    display: flex;
    flex-direction: column;
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}

.kpi-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
}

.kpi-title {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    color: #ffffff;
    margin: 6px 0;
}

.kpi-subtext {
    font-size: 0.82rem;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-urgent {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    animation: pulse-red 2s infinite;
}

.badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-info {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.4);
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    50% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
}

/* Evidence Ledger Quote Box */
.evidence-quote-box {
    background: rgba(30, 41, 59, 0.5);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-style: italic;
    color: #e2e8f0;
}

.evidence-timestamp {
    font-size: 0.75rem;
    font-family: monospace;
    color: #818cf8;
    font-weight: 600;
    text-transform: uppercase;
}

/* Turn Bubble */
.turn-agent {
    background: rgba(30, 41, 59, 0.7);
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
}

.turn-customer {
    background: rgba(49, 46, 129, 0.3);
    border-left: 3px solid #ec4899;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
}
</style>
"""
