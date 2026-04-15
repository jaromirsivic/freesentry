export const DEFAULT_CAMERA_SETTINGS = Object.freeze({
    selectedCamera: 'scope_camera',
    streamQuality: 80,
    scopeCameraMode: 0,
    spotterCamera1Mode: 0,
    spotterCamera2Mode: 0,
    spotterCamera3Mode: 0
});

export const DEFAULT_RETICLE_SETTINGS = Object.freeze({
    x: 0.5,
    y: 0.5,
    color: '#88ff00cc',
    outline: '#000000cc',
    size: 1.0
});

function getCameraMode(settings, cameraCode) {
    switch (cameraCode) {
        case 'scope_camera':
            return settings.scopeCameraMode ?? 0;
        case 'spotter_camera1':
            return settings.spotterCamera1Mode ?? 0;
        case 'spotter_camera2':
            return settings.spotterCamera2Mode ?? 0;
        case 'spotter_camera3':
            return settings.spotterCamera3Mode ?? 0;
        default:
            return 0;
    }
}

export function buildCameraStreamUrl(settings = DEFAULT_CAMERA_SETTINGS) {
    const resolvedSettings = { ...DEFAULT_CAMERA_SETTINGS, ...(settings ?? {}) };
    const cameraCode = resolvedSettings.selectedCamera || DEFAULT_CAMERA_SETTINGS.selectedCamera;
    const quality = resolvedSettings.streamQuality ?? DEFAULT_CAMERA_SETTINGS.streamQuality;
    const mode = getCameraMode(resolvedSettings, cameraCode);
    return `/api/cameras/stream/${cameraCode}?mode=${mode}&quality=${quality}`;
}

export function buildStopCameraUrl({ cameraCode, cameraSettings } = {}) {
    const activeCameraCode = cameraCode
        || cameraSettings?.selectedCamera
        || DEFAULT_CAMERA_SETTINGS.selectedCamera;
    return `/api/cameras/stop/${activeCameraCode}`;
}

export function getReticleSettingsFromCameraConfig(cameraConfig) {
    return {
        x: cameraConfig?.static_reticle_x ?? DEFAULT_RETICLE_SETTINGS.x,
        y: cameraConfig?.static_reticle_y ?? DEFAULT_RETICLE_SETTINGS.y,
        color: cameraConfig?.static_reticle_color ?? DEFAULT_RETICLE_SETTINGS.color,
        outline: cameraConfig?.static_reticle_outline ?? DEFAULT_RETICLE_SETTINGS.outline,
        size: cameraConfig?.static_reticle_size ?? DEFAULT_RETICLE_SETTINGS.size
    };
}

export async function fetchReticleSettings({ cameraCode = DEFAULT_CAMERA_SETTINGS.selectedCamera, signal } = {}) {
    const response = await fetch('/api/cameras/list', { signal });
    if (!response.ok) {
        throw new Error(`GET /api/cameras/list failed with status ${response.status}`);
    }

    const data = await response.json();
    if (!data?.success || !data?.cameras) {
        return { ...DEFAULT_RETICLE_SETTINGS };
    }

    return getReticleSettingsFromCameraConfig(data.cameras[cameraCode]);
}

export async function stopManagedCameraStream({ cameraCode, cameraSettings } = {}) {
    try {
        const response = await fetch(buildStopCameraUrl({ cameraCode, cameraSettings }), {
            method: 'POST'
        });
        const data = await response.json();
        return data.success === true;
    } catch (err) {
        console.error('Failed to stop camera stream:', err);
        return false;
    }
}

export function queueDismissalCameraStop({ cameraCode, cameraSettings } = {}) {
    const stopUrl = buildStopCameraUrl({ cameraCode, cameraSettings });

    if (typeof fetch === 'function' && typeof Request !== 'undefined' && 'keepalive' in Request.prototype) {
        try {
            void fetch(stopUrl, { method: 'POST', keepalive: true }).catch(() => {});
            return;
        } catch {
            // Fall through to sendBeacon for older browsers.
        }
    }

    if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
        try {
            navigator.sendBeacon(stopUrl, new Blob());
        } catch {
            // Best-effort only during page dismissal.
        }
    }
}
