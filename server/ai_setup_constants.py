"""
Single source of truth for AI Setup device options (inference device).
Used by REST API (aisetup) and camera/inference code.
"""

# Device options for the AI inference backend: label and value for UI, value is stored in settings.
DEVICE_OPTIONS = [
    {"label": "cpu (optimized for x86)", "value": "cpu (optimized for x86)"},
    {"label": "arm_cpu (optimized for ARM)", "value": "arm_cpu (optimized for ARM)"},
    {"label": "cuda:0 (Nvidia GPU)", "value": "cuda:0 (Nvidia GPU)"},
]

DEFAULT_DEVICE = "cpu (optimized for x86)"
