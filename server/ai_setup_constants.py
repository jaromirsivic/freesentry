"""
Single source of truth for AI Setup device options (inference device).
Used by REST API (aisetup) and camera/inference code.
"""

CPU_DEVICE_VALUE = "cpu (optimized for x86)"
ARM_CPU_DEVICE_VALUE = "arm_cpu (optimized for ARM)"
VULKAN_DEVICE_0_VALUE = "vulkan:0 (AMD, Nvidia, Intel, ...)"
VULKAN_DEVICE_1_VALUE = "vulkan:1 (AMD, Nvidia, Intel, ...)"
CUDA_DEVICE_0_VALUE = "cuda:0 (Nvidia GPU Only)"
CUDA_DEVICE_1_VALUE = "cuda:1 (Nvidia GPU Only)"
VULKAN_DEVICE_VALUES = (
    VULKAN_DEVICE_0_VALUE,
    VULKAN_DEVICE_1_VALUE,
)
CUDA_DEVICE_VALUES = (
    CUDA_DEVICE_0_VALUE,
    CUDA_DEVICE_1_VALUE,
)
# Backward compatible alias for existing imports/defaults.
CUDA_DEVICE_VALUE = CUDA_DEVICE_0_VALUE

# Device options for the AI inference backend: label and value for UI, value is stored in settings.
DEVICE_OPTIONS = [
    {"label": CPU_DEVICE_VALUE, "value": CPU_DEVICE_VALUE},
    {"label": ARM_CPU_DEVICE_VALUE, "value": ARM_CPU_DEVICE_VALUE},
    {"label": VULKAN_DEVICE_0_VALUE, "value": VULKAN_DEVICE_0_VALUE},
    {"label": VULKAN_DEVICE_1_VALUE, "value": VULKAN_DEVICE_1_VALUE},
    {"label": CUDA_DEVICE_0_VALUE, "value": CUDA_DEVICE_0_VALUE},
    {"label": CUDA_DEVICE_1_VALUE, "value": CUDA_DEVICE_1_VALUE},
]

DEFAULT_DEVICE = CPU_DEVICE_VALUE
