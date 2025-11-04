"""
Gunicorn configuration for Segmentation API on Render
Optimized for 150MB model file with limited resources
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker processes
# Use only 1 worker for memory-intensive segmentation model
workers = 1
worker_class = "sync"

# Timeout settings - CRITICAL for model loading on first request
# 150MB model takes time to load
timeout = 300  # 5 minutes for first request
graceful_timeout = 300
keepalive = 5

# Memory management
# Restart worker periodically to prevent memory leaks
max_requests = 50  # Lower than classification API due to larger model
max_requests_jitter = 5

# File upload limits
limit_request_line = 0  # No limit on request line
limit_request_field_size = 0  # No limit on request field size

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr  
loglevel = "info"

# Performance
preload_app = False  # Don't preload - using lazy loading
worker_tmp_dir = "/dev/shm"  # Use memory for heartbeat files

# Prevent worker timeout during initial model loading
worker_connections = 1000

# Graceful shutdown
graceful_timeout = 120

print("🔧 Gunicorn Configuration Loaded:")
print(f"   Bind: {bind}")
print(f"   Workers: {workers}")
print(f"   Timeout: {timeout}s")
print(f"   Max requests per worker: {max_requests}")
print(f"   Worker temp dir: {worker_tmp_dir}")