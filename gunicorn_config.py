"""
Gunicorn config - ULTRA optimized for FREE TIER (512MB RAM)
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker settings - CRITICAL for free tier
workers = 1  # Only 1 worker
worker_class = "sync"
threads = 1  # No threading

# Timeout - Allow model loading
timeout = 300
graceful_timeout = 300
keepalive = 2

# Memory management - AGGRESSIVE
max_requests = 10  # Restart after just 10 requests
max_requests_jitter = 2
worker_connections = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Performance
preload_app = False
worker_tmp_dir = "/dev/shm"

# Prevent memory leaks
max_requests = 10  # Very aggressive restart

print("🔧 FREE TIER Gunicorn Config:")
print(f"   Workers: {workers}")
print(f"   Timeout: {timeout}s")
print(f"   Max requests: {max_requests}")
print("   ⚠️  ULTRA MEMORY OPTIMIZATION MODE")