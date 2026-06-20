"""Central place to control Modal resources for this project."""

# Training job resources
TRAIN_GPU = "L40S"
TRAIN_CPU = 8.0
TRAIN_MEMORY_MB = 65536
TRAIN_TIMEOUT_SECONDS = 60 * 60 * 24

# Cache / preprocessing job resources
CACHE_CPU = 8.0
CACHE_MEMORY_MB = 32768
CACHE_TIMEOUT_SECONDS = 60 * 60 * 8

# Lightweight dataset / summary job resources
OPS_CPU = 2.0
OPS_MEMORY_MB = 4096
OPS_TIMEOUT_SECONDS = 60 * 60 * 6
