import React, { useState, useCallback, useEffect, useRef } from 'react';
import Panel from './components/Panel';
import Button from './components/Button';
import ComboBox from './components/ComboBox';
import Switch from './components/Switch';
import ColumnLayout from './components/ColumnLayout';
import HorizontalSeparator from './components/HorizontalSeparator';
import ModalWindow from './components/ModalWindow';
import StaticText from './components/StaticText';
import Slider from './components/Slider';
import NumericInput from './components/NumericInput';
import Polygon from './components/Polygon';
import ColorPicker from './components/ColorPicker';
import MultiSwitch from './components/MultiSwitch';
import editIcon from './assets/icons/edit.svg';
import reloadIcon from './assets/icons/reload.svg';
import cameraOffIcon from './assets/icons/cameraOff.svg';
import { queueDismissalCameraStop } from './lib/cameraStream';

const boldTextStyle = { fontWeight: 'bold' };

/**
 * Helper to render static text field.
 */
const RenderStaticField = ({ label, value }) => (
    <StaticText text={<>{label}: <span style={boldTextStyle}>{value}</span></>} />
);

/**
 * Camera code to display name mapping.
 */
const CAMERA_NAMES = {
    scope_camera: 'Scope Camera',
    spotter_camera1: 'Spotter Camera 1',
    spotter_camera2: 'Spotter Camera 2',
    spotter_camera3: 'Spotter Camera 3'
};

/**
 * Camera codes in display order.
 */
const CAMERA_CODES = ['scope_camera', 'spotter_camera1', 'spotter_camera2', 'spotter_camera3'];

/**
 * Rotation options for the dropdown.
 */
const ROTATE_OPTIONS = [
    { label: '0', value: '0' },
    { label: '90', value: '90' },
    { label: '180', value: '180' },
    { label: '270', value: '270' }
];

/**
 * Keys in modal state that do NOT trigger stream stop / orange outline.
 * Static Reticle: all; Preview: only HTTP Stream Quality and Display Mode trigger.
 */
const NON_TRIGGER_KEYS = new Set([
    'reticleX', 'reticleY', 'reticleColor', 'reticleOutline', 'reticleSize',
    'maskPolygons'
]);

/** Red button color for Apply/Save when there are relevant unsaved changes. */
const RELEVANT_CHANGE_BUTTON_COLOR = '#ef4444';

/**
 * Mapping from slider property names to capability keys.
 */
const CAPABILITY_MAP = {
    brightness: 'CAP_PROP_BRIGHTNESS',
    contrast: 'CAP_PROP_CONTRAST',
    hue: 'CAP_PROP_HUE',
    saturation: 'CAP_PROP_SATURATION',
    sharpness: 'CAP_PROP_SHARPNESS',
    gamma: 'CAP_PROP_GAMMA',
    gain: 'CAP_PROP_GAIN',
    backlight: 'CAP_PROP_BACKLIGHT',
    whiteBalanceTemperature: 'CAP_PROP_WB_TEMPERATURE',
    focus: 'CAP_PROP_FOCUS',
    exposure: 'CAP_PROP_EXPOSURE'
};

/**
 * Default slider props when no capabilities are available.
 */
const DEFAULT_SLIDER_PROPS = {
    min: -1000000,
    max: 1000000,
    minSlider: -255,
    maxSlider: 255,
    step: 1,
    disabled: false
};

/**
 * Get slider props from capabilities for a given property.
 * @param {object} capabilities - Device capabilities object.
 * @param {string} propName - Property name (e.g., 'brightness').
 * @returns {object} Slider props (min, max, minSlider, maxSlider, step, disabled).
 */
const getSliderPropsFromCapabilities = (capabilities, propName) => {
    const capKey = CAPABILITY_MAP[propName];
    if (!capabilities || !capKey || !capabilities[capKey]) {
        return DEFAULT_SLIDER_PROPS;
    }
    const cap = capabilities[capKey];
    return {
        min: cap.min ?? DEFAULT_SLIDER_PROPS.min,
        max: cap.max ?? DEFAULT_SLIDER_PROPS.max,
        minSlider: cap.minSlider ?? cap.min ?? DEFAULT_SLIDER_PROPS.minSlider,
        maxSlider: cap.maxSlider ?? cap.max ?? DEFAULT_SLIDER_PROPS.maxSlider,
        step: cap.step ?? DEFAULT_SLIDER_PROPS.step,
        disabled: cap.enabled === false
    };
};

/**
 * Parse resolution string to get dimensions.
 */
const getResolutionDimensions = (resString) => {
    if (!resString) return { width: 0, height: 0 };
    const parts = resString.split(' x ');
    return {
        width: parseInt(parts[0]) || 0,
        height: parseInt(parts[1]) || 0
    };
};

const getInputDeviceIndex = (device) => {
    const parsedIndex = Number(device?.index);
    return Number.isInteger(parsedIndex) ? parsedIndex : null;
};

const normalizeInputDevices = (devices) => {
    const normalizedDevices = new Map();

    (Array.isArray(devices) ? devices : []).forEach((device) => {
        const index = getInputDeviceIndex(device);
        if (index === null || normalizedDevices.has(index)) {
            return;
        }

        const label = typeof device?.name === 'string' && device.name.trim()
            ? device.name.trim()
            : `Device ${index}`;

        normalizedDevices.set(index, {
            ...device,
            index,
            name: label
        });
    });

    return Array.from(normalizedDevices.values());
};

/**
 * Cameras settings page with three panels:
 * - Scope Camera - AI
 * - Scope Camera - Manual Control
 * - Spotter Camera
 */
const Cameras = () => {
    // ========================
    // State: Cameras data from API
    // ========================
    const [cameras, setCameras] = useState({});
    const [inputDevices, setInputDevices] = useState([]);
    const [isLoading, setIsLoading] = useState(true);

    // ========================
    // State: Modal
    // ========================
    const [activeModal, setActiveModal] = useState(null); // camera_code or null
    const [tempState, setTempState] = useState({});
    const [originalModalState, setOriginalModalState] = useState({});
    const [showCancelConfirmModal, setShowCancelConfirmModal] = useState(false);
    const [showResetConfirmModal, setShowResetConfirmModal] = useState(false);
    const [showResetAllConfirmModal, setShowResetAllConfirmModal] = useState(false);
    const [isResettingAll, setIsResettingAll] = useState(false);
    const [streamVersion, setStreamVersion] = useState(0);
    const [previewEnabled, setPreviewEnabled] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [isApplying, setIsApplying] = useState(false);
    const [isResetting, setIsResetting] = useState(false);
    const [isClosing, setIsClosing] = useState(false);

    // Preview controls (not persisted in settings.json)
    const [maskedAreaColor, setMaskedAreaColor] = useState('#ff00000f');
    const [streamQuality, setStreamQuality] = useState(80);
    const [displayMode, setDisplayMode] = useState(0);
    // Snapshot of stream quality/display mode when modal opened (for relevant-change detection)
    const [originalStreamQuality, setOriginalStreamQuality] = useState(80);
    const [originalDisplayMode, setOriginalDisplayMode] = useState(0);
    const activeModalRef = useRef(null);
    const previewEnabledRef = useRef(false);
    const dismissalStopQueuedRef = useRef(false);

    const setActiveModalState = useCallback((cameraCode) => {
        activeModalRef.current = cameraCode;
        setActiveModal(cameraCode);
    }, []);

    const setPreviewEnabledState = useCallback((enabled) => {
        previewEnabledRef.current = enabled;
        setPreviewEnabled(enabled);
    }, []);

    const queueDismissalStop = useCallback((cameraCode) => {
        const resolvedCameraCode = cameraCode ?? activeModalRef.current;
        if (!resolvedCameraCode || dismissalStopQueuedRef.current) {
            return;
        }

        dismissalStopQueuedRef.current = true;
        queueDismissalCameraStop({ cameraCode: resolvedCameraCode });
    }, []);

    // ========================
    // Fetch cameras list from API
    // ========================
    const fetchCameras = useCallback(async () => {
        try {
            const response = await fetch('/api/cameras/list');
            const data = await response.json();
            if (data.success) {
                setCameras(data.cameras || {});
            }
        } catch (err) {
            console.error('Failed to load cameras:', err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // ========================
    // Fetch input devices from API
    // ========================
    const fetchInputDevices = useCallback(async () => {
        try {
            const response = await fetch('/api/cameras/input_devices');
            const data = await response.json();
            if (data.success) {
                setInputDevices(data.input_devices || []);
            }
        } catch (err) {
            console.error('Failed to load input devices:', err);
        }
    }, []);

    // Initial data load
    useEffect(() => {
        fetchCameras();
        fetchInputDevices();
    }, [fetchCameras, fetchInputDevices]);

    // ========================
    // Helper: Get camera state from config
    // ========================
    const getCameraStateFromConfig = useCallback((config) => {
        if (!config) return {};
        return {
            index: config.index ?? 0,
            name: config.name ?? '',
            preferredResolution: config.width && config.height ? `${config.width} x ${config.height}` : '1920 x 1080',
            fps: config.fps ?? 30,
            flipHorizontally: config.flip_horizontal ?? false,
            flipVertically: config.flip_vertical ?? false,
            rotateDegrees: String(config.rotate ?? 0),
            brightness: config.brightness ?? 0,
            contrast: config.contrast ?? 0,
            hue: config.hue ?? 0,
            saturation: config.saturation ?? 0,
            sharpness: config.sharpness ?? 0,
            gamma: config.gamma ?? 0,
            whiteBalanceTemperature: config.white_balance_temperature ?? 0,
            backlight: config.backlight ?? 0,
            gain: config.gain ?? 0,
            focus: config.focus ?? 0,
            exposure: config.exposure ?? 0,
            autoWhiteBalance: config.auto_white_balance_temperature ?? false,
            autoFocus: config.auto_focus ?? false,
            autoExposure: config.auto_exposure ?? false,
            cropTop: (config.crop_top ?? 0) * 100,
            cropLeft: (config.crop_left ?? 0) * 100,
            cropBottom: (config.crop_bottom ?? 0) * 100,
            cropRight: (config.crop_right ?? 0) * 100,
            stretchEnabled: config.stretch_enabled ?? false,
            stretchWidth: config.stretch_width ?? 0,
            stretchHeight: config.stretch_height ?? 0,
            reticleX: config.static_reticle_x ?? 0,
            reticleY: config.static_reticle_y ?? 0,
            reticleColor: config.static_reticle_color ?? '#ff0000cc',
            reticleOutline: config.static_reticle_outline ?? '#000000cc',
            reticleSize: config.static_reticle_size ?? 0,
            maskPolygons: config.mask_polygons ?? []
        };
    }, []);

    // ========================
    // Modal: Open
    // ========================
    const openModal = useCallback((cameraCode) => {
        const config = cameras[cameraCode] || {};
        const state = getCameraStateFromConfig(config);
        setTempState(state);
        setOriginalModalState(JSON.parse(JSON.stringify(state)));
        setActiveModalState(cameraCode);
        setOriginalStreamQuality(streamQuality);
        setOriginalDisplayMode(displayMode);
        // Eagerly warm up the camera worker so the preview <img> below
        // doesn't have to pay for process spawn + device open + first
        // frame encode.  We fire a tiny stream request and abort it after
        // ~50 ms — just long enough for the server to run
        // Camera._ensure_active (which spawns the worker and opens the
        // device).  Failures are silent on purpose: this is a best-effort
        // latency hint, the real stream below still works without it.
        try {
            const controller = new AbortController();
            const warmupUrl = `/api/cameras/stream/${encodeURIComponent(cameraCode)}?mode=0&quality=10&_warmup=${Date.now()}`;
            fetch(warmupUrl, { signal: controller.signal }).catch(() => {});
            setTimeout(() => { try { controller.abort(); } catch { /* noop */ } }, 50);
        } catch { /* noop */ }
        // Enable preview and increment stream version
        setPreviewEnabledState(true);
        setStreamVersion(v => v + 1);
    }, [cameras, getCameraStateFromConfig, streamQuality, displayMode, setActiveModalState, setPreviewEnabledState]);

    // ========================
    // Modal: Close
    // ========================
    const closeModal = useCallback(() => {
        setPreviewEnabledState(false);
        setActiveModalState(null);
        setTempState({});
        setOriginalModalState({});
        setShowCancelConfirmModal(false);
        setIsSaving(false);
        setIsApplying(false);
        setIsClosing(false);
    }, [setActiveModalState, setPreviewEnabledState]);

    // ========================
    // Modal: Check for changes
    // ========================
    const hasModalChanges = useCallback(() => {
        return JSON.stringify(tempState) !== JSON.stringify(originalModalState);
    }, [tempState, originalModalState]);

    /**
     * True when user has modified any field that triggers stream stop (all sections
     * except Static Reticle; in Preview only HTTP Stream Quality and Display Mode).
     */
    const hasRelevantChanges = useCallback(() => {
        for (const key of Object.keys(tempState)) {
            if (NON_TRIGGER_KEYS.has(key)) continue;
            if (JSON.stringify(tempState[key]) !== JSON.stringify(originalModalState[key])) {
                return true;
            }
        }
        if (streamQuality !== originalStreamQuality || displayMode !== originalDisplayMode) {
            return true;
        }
        return false;
    }, [tempState, originalModalState, streamQuality, originalStreamQuality, displayMode, originalDisplayMode]);

    // ========================
    // Helper: Stop camera stream via API
    // ========================
    const stopCameraStream = useCallback(async ({ cameraCode, keepalive = false } = {}) => {
        const resolvedCameraCode = cameraCode ?? activeModalRef.current;
        if (!resolvedCameraCode) {
            return false;
        }

        try {
            const requestOptions = { method: 'POST' };
            if (keepalive && typeof Request !== 'undefined' && 'keepalive' in Request.prototype) {
                requestOptions.keepalive = true;
            }

            const response = await fetch(`/api/cameras/stop/${resolvedCameraCode}`, requestOptions);
            if (!response.ok) {
                console.error(`Failed to stop camera stream for ${resolvedCameraCode}: ${response.status} ${response.statusText}`);
                return false;
            }

            const data = await response.json();
            if (data.success !== true) {
                console.error(`Camera stop did not return success for ${resolvedCameraCode}:`, data);
                return false;
            }

            return true;
        } catch (err) {
            console.error(`Failed to stop camera stream for ${resolvedCameraCode}:`, err);
            return false;
        }
    }, []);

    const stopPreviewStream = useCallback(async ({ cameraCode, updateUi = true, keepalive = false } = {}) => {
        if (updateUi) {
            setPreviewEnabledState(false);
        }

        return stopCameraStream({ cameraCode, keepalive });
    }, [setPreviewEnabledState, stopCameraStream]);

    useEffect(() => {
        dismissalStopQueuedRef.current = false;
    }, [activeModal]);

    // Route unmount must still release the backend preview stream.
    useEffect(() => {
        return () => {
            if (!activeModalRef.current || dismissalStopQueuedRef.current) {
                return;
            }

            void stopPreviewStream({
                cameraCode: activeModalRef.current,
                updateUi: false,
                keepalive: true
            });
        };
    }, [stopPreviewStream]);

    useEffect(() => {
        const handlePageHide = (event) => {
            if (!event.persisted) {
                queueDismissalStop();
            }
        };

        const handleBeforeUnload = () => {
            queueDismissalStop();
        };

        window.addEventListener('pagehide', handlePageHide);
        window.addEventListener('beforeunload', handleBeforeUnload);

        return () => {
            window.removeEventListener('pagehide', handlePageHide);
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, [queueDismissalStop]);

    // ========================
    // Modal: Close request
    // ========================
    const handleCloseRequest = useCallback(async () => {
        // Stop the preview stream immediately and show "Closing..."
        setPreviewEnabledState(false);
        setIsClosing(true);
        
        // Stop the camera stream via API immediately
        await stopPreviewStream({ updateUi: false });
        
        // Wait 1 second to allow the UI to update
        await new Promise(resolve => setTimeout(resolve, 100));
        
        if (hasModalChanges()) {
            setShowCancelConfirmModal(true);
            setIsClosing(false);
        } else {
            closeModal();
        }
    }, [hasModalChanges, closeModal, setPreviewEnabledState, stopPreviewStream]);

    // ========================
    // Modal: Cancel confirm
    // ========================
    const handleCancelConfirm = useCallback(() => {
        closeModal();
    }, [closeModal]);

    // ========================
    // Modal: Update temp state
    // ========================
    const updateTempState = useCallback((key, value) => {
        setTempState(prev => ({ ...prev, [key]: value }));
    }, []);

    /**
     * Update temp state and, for trigger fields, stop stream and show stream-unavailable icon.
     * Do not use for Static Reticle or for Preview except HTTP Stream Quality / Display Mode.
     */
    const handleRelevantFieldChange = useCallback((key, value) => {
        setTempState(prev => ({ ...prev, [key]: value }));
        void stopPreviewStream();
    }, [stopPreviewStream]);

    // ========================
    // Helper: Build payload from tempState
    // ========================
    const buildPayload = useCallback(() => {
        const [width, height] = tempState.preferredResolution.split(' x ').map(Number);
        return {
            index: Number(tempState.index),
            name: tempState.name,
            width: width || 1920,
            height: height || 1080,
            fps: Number(tempState.fps),
            flip_horizontal: tempState.flipHorizontally,
            flip_vertical: tempState.flipVertically,
            rotate: Number(tempState.rotateDegrees),
            brightness: Number(tempState.brightness),
            contrast: Number(tempState.contrast),
            hue: Number(tempState.hue),
            saturation: Number(tempState.saturation),
            sharpness: Number(tempState.sharpness),
            gamma: Number(tempState.gamma),
            white_balance_temperature: Number(tempState.whiteBalanceTemperature),
            backlight: Number(tempState.backlight),
            gain: Number(tempState.gain),
            focus: Number(tempState.focus),
            exposure: Number(tempState.exposure),
            auto_white_balance_temperature: tempState.autoWhiteBalance,
            auto_focus: tempState.autoFocus,
            auto_exposure: tempState.autoExposure,
            crop_top: Number(tempState.cropTop) / 100.0,
            crop_left: Number(tempState.cropLeft) / 100.0,
            crop_bottom: Number(tempState.cropBottom) / 100.0,
            crop_right: Number(tempState.cropRight) / 100.0,
            stretch_enabled: tempState.stretchEnabled,
            stretch_width: Number(tempState.stretchWidth),
            stretch_height: Number(tempState.stretchHeight),
            static_reticle_x: Number(tempState.reticleX),
            static_reticle_y: Number(tempState.reticleY),
            static_reticle_color: tempState.reticleColor,
            static_reticle_outline: tempState.reticleOutline,
            static_reticle_size: Number(tempState.reticleSize),
            mask_polygons: tempState.maskPolygons || []
        };
    }, [tempState]);

    // ========================
    // Modal: Save (save to disk and close)
    // ========================
    const saveModal = useCallback(async () => {
        if (!activeModal) return;

        try {
            // Stop preview and show saving state
            setPreviewEnabledState(false);
            setIsSaving(true);

            // Stop the camera stream via API immediately
            await stopPreviewStream({ updateUi: false });

            // Wait 1 second to allow the UI to update
            await new Promise(resolve => setTimeout(resolve, 100));

            const payload = buildPayload();

            const response = await fetch(`/api/cameras/update/${activeModal}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.success) {
                // Refresh cameras list and close modal
                await fetchCameras();
                await fetchInputDevices();
                closeModal();
            }
        } catch (err) {
            console.error('Failed to save camera settings:', err);
            // Re-enable preview on error
            setPreviewEnabledState(true);
        } finally {
            setIsSaving(false);
        }
    }, [activeModal, buildPayload, fetchCameras, fetchInputDevices, closeModal, setPreviewEnabledState, stopPreviewStream]);

    // ========================
    // Modal: Apply (apply changes and restart preview)
    // ========================
    const applyModal = useCallback(async () => {
        if (!activeModal) return;

        try {
            // Stop preview and show applying state
            setPreviewEnabledState(false);
            setIsApplying(true);

            // Stop the camera stream via API immediately
            await stopPreviewStream({ updateUi: false });

            // Wait 1 second to allow the UI to update
            await new Promise(resolve => setTimeout(resolve, 100));

            const payload = buildPayload();

            const response = await fetch(`/api/cameras/update/${activeModal}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.success) {
                await fetchCameras();
                await fetchInputDevices();
                setOriginalModalState(JSON.parse(JSON.stringify(tempState)));
                setOriginalStreamQuality(streamQuality);
                setOriginalDisplayMode(displayMode);
            }
        } catch (err) {
            console.error('Failed to apply camera settings:', err);
        } finally {
            setIsApplying(false);
            setPreviewEnabledState(true);
            setStreamVersion(v => v + 1);
        }
    }, [activeModal, buildPayload, fetchCameras, fetchInputDevices, tempState, streamQuality, displayMode, setPreviewEnabledState, stopPreviewStream]);

    // ========================
    // Modal: Reset Camera - show confirmation
    // ========================
    const handleResetRequest = useCallback(() => {
        setShowResetConfirmModal(true);
    }, []);

    // ========================
    // Modal: Reset Camera - cancel
    // ========================
    const handleResetCancel = useCallback(() => {
        setShowResetConfirmModal(false);
    }, []);

    // ========================
    // Modal: Reset Camera - confirm (reset to defaults and refresh)
    // ========================
    const handleResetConfirm = useCallback(async () => {
        if (!activeModal) return;

        // Hide the confirmation modal
        setShowResetConfirmModal(false);

        try {
            // Stop preview and show resetting state
            setPreviewEnabledState(false);
            setIsResetting(true);

            // Call reset API
            const response = await fetch(`/api/cameras/reset/${activeModal}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (data.success) {
                // Refresh cameras list and input devices
                await fetchCameras();
                await fetchInputDevices();
                
                // Update tempState with new camera config from refreshed data
                // Need to get updated config from API response
                const camerasResponse = await fetch('/api/cameras/list');
                const camerasData = await camerasResponse.json();
                if (camerasData.success) {
                    const updatedConfig = camerasData.cameras[activeModal] || {};
                    const updatedState = getCameraStateFromConfig(updatedConfig);
                    setTempState(updatedState);
                    setOriginalModalState(JSON.parse(JSON.stringify(updatedState)));
                }
            }
        } catch (err) {
            console.error('Failed to reset camera settings:', err);
        } finally {
            setIsResetting(false);
            // Restart the preview
            setPreviewEnabledState(true);
            setStreamVersion(v => v + 1);
        }
    }, [activeModal, fetchCameras, fetchInputDevices, getCameraStateFromConfig, setPreviewEnabledState]);

    // ========================
    // Reset All Cameras: Show confirmation modal
    // ========================
    const handleResetAllRequest = useCallback(() => {
        setShowResetAllConfirmModal(true);
    }, []);

    // ========================
    // Reset All Cameras: Cancel
    // ========================
    const handleResetAllCancel = useCallback(() => {
        setShowResetAllConfirmModal(false);
    }, []);

    // ========================
    // Reset All Cameras: Confirm (call API and refresh)
    // ========================
    const handleResetAllConfirm = useCallback(async () => {
        setIsResettingAll(true);

        try {
            const response = await fetch('/api/cameras/resetall', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            if (data.success) {
                // Refresh cameras list and input devices
                await fetchCameras();
                await fetchInputDevices();
                // Close the modal after success
                setShowResetAllConfirmModal(false);
            }
        } catch (err) {
            console.error('Failed to reset all cameras:', err);
        } finally {
            setIsResettingAll(false);
        }
    }, [fetchCameras, fetchInputDevices]);

    // ========================
    // Modal: Input device change handler
    // ========================
    const onInputDeviceChange = useCallback((val) => {
        const device = normalizeInputDevices(inputDevices).find(d => String(d.index) === String(val));
        if (device) {
            const deviceState = getCameraStateFromConfig(device);
            setTempState(deviceState);
            void stopPreviewStream();
        }
    }, [inputDevices, getCameraStateFromConfig, stopPreviewStream]);

    // ========================
    // Modal: Resolution change handler
    // ========================
    const onResolutionChange = useCallback((val) => {
        updateTempState('preferredResolution', val);
        void stopPreviewStream();
    }, [updateTempState, stopPreviewStream]);

    // ========================
    // Modal: FPS change handler
    // ========================
    const onFpsChange = useCallback((val) => {
        updateTempState('fps', val);
        void stopPreviewStream();
    }, [updateTempState, stopPreviewStream]);

    // ========================
    // Modal: Stream quality change handler
    // ========================
    const onStreamQualityChange = useCallback((val) => {
        setStreamQuality(val);
        void stopPreviewStream();
    }, [stopPreviewStream]);

    // ========================
    // Modal: Display mode change handler
    // ========================
    const onDisplayModeChange = useCallback((val) => {
        setDisplayMode(val);
        void stopPreviewStream();
    }, [stopPreviewStream]);

    // ========================
    // Compute derived values
    // ========================
    const availableInputDevices = normalizeInputDevices(inputDevices);

    const inputDeviceOptions = availableInputDevices.map(d => ({
        label: d.name,
        value: String(d.index)
    }));

    // Resolution options based on selected device in modal
    const selectedDevice = availableInputDevices.find(d => String(d.index) === String(tempState.index));
    let resolutionOptions = [];
    if (selectedDevice?.supported_resolutions?.length > 0) {
        resolutionOptions = selectedDevice.supported_resolutions.map(r => ({
            label: r.label,
            value: r.label
        }));
    } else {
        resolutionOptions = [{ label: '1920 x 1080', value: '1920 x 1080' }];
    }

    // Resolution dimensions for crop pixel display
    const { width: resWidth, height: resHeight } = getResolutionDimensions(tempState.preferredResolution);

    // Get capabilities from selected device
    const selectedDeviceCapabilities = selectedDevice?.capabilities || {};

    // ========================
    // Validation for Camera Modal
    // ========================
    const getCameraValidationErrors = () => {
        if (!activeModal) return [];
        const errors = [];
        const top = Number(tempState.cropTop || 0);
        const left = Number(tempState.cropLeft || 0);
        const bottom = Number(tempState.cropBottom || 0);
        const right = Number(tempState.cropRight || 0);

        if (left + right >= 99) {
            errors.push('Total horizontal crop (Left + Right) cannot exceed 99%');
        }
        if (top + bottom >= 99) {
            errors.push('Total vertical crop (Top + Bottom) cannot exceed 99%');
        }

        // Validate device index uniqueness across all cameras
        const otherCameraCodes = CAMERA_CODES.filter(code => code !== activeModal);
        for (const otherCode of otherCameraCodes) {
            const otherCameraConfig = cameras[otherCode] || {};
            const otherDeviceIndex = otherCameraConfig.index;
            if (otherDeviceIndex !== undefined && Number(tempState.index) === Number(otherDeviceIndex)) {
                errors.push(`${CAMERA_NAMES[activeModal]} and ${CAMERA_NAMES[otherCode]} must use different Device Indexes`);
                break;
            }
        }

        return errors;
    };

    const validationErrors = getCameraValidationErrors();
    const isValid = validationErrors.length === 0;

    // ========================
    // Render: Camera Panel
    // ========================
    const renderCameraPanel = (cameraCode) => {
        const config = cameras[cameraCode] || {};
        const panelName = CAMERA_NAMES[cameraCode];
        const resolution = config.width && config.height 
            ? `${config.width} x ${config.height}` 
            : 'Not configured';

        return (
            <div key={cameraCode} style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex' }}>
                <Panel
                    style={{ flex: 1 }}
                    title={panelName}
                    headerAction={
                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                            <Button
                                label={<img src={reloadIcon} alt="Reset all" width="24" height="24" />}
                                onClick={handleResetAllRequest}
                                disabled={isResettingAll}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            />
                            <Button
                                label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                onClick={() => openModal(cameraCode)}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            />
                        </div>
                    }
                >
                    <ColumnLayout gap="0.75rem">
                        <HorizontalSeparator label="Camera Settings" fullWidth={true} />
                        <RenderStaticField label="Device Index" value={config.index ?? 'N/A'} />
                        <RenderStaticField label="Resolution" value={resolution} />
                        <RenderStaticField label="FPS" value={config.fps ?? 'N/A'} />
                    </ColumnLayout>
                </Panel>
            </div>
        );
    };

    // ========================
    // Render: Edit Modal
    // ========================
    const renderEditModal = () => {
        if (!activeModal) return null;

        const panelName = CAMERA_NAMES[activeModal];

        const isProcessing = isSaving || isApplying || isResetting || isClosing;
        const showRelevantOutline = hasRelevantChanges();

        return (
            <ModalWindow
                isOpen={true}
                title={`Edit ${panelName}`}
                onOk={saveModal}
                onCancel={handleCloseRequest}
                okLabel={isSaving ? "Saving..." : "Save"}
                cancelLabel={isClosing ? "Closing..." : "Close"}
                validationErrors={validationErrors}
                okDisabled={!isValid || isProcessing}
                okButtonColor={showRelevantOutline ? RELEVANT_CHANGE_BUTTON_COLOR : null}
                customFooterButtons={[
                    <Button
                        key="apply"
                        label={isApplying ? "Applying..." : "Apply"}
                        onClick={applyModal}
                        color={(!isValid || isProcessing) ? '#94a3b8' : (showRelevantOutline ? RELEVANT_CHANGE_BUTTON_COLOR : '#3b82f6')}
                        disabled={!isValid || isProcessing}
                        style={{ height: '40px', display: 'flex', alignItems: 'center' }}
                    />
                ]}
            >
                <ColumnLayout gap="0.75rem">
                    {/* General Setup */}
                    <HorizontalSeparator label="General Setup" fullWidth={true} bleed="1rem" />
                    <ComboBox
                        label="Input Device"
                        items={inputDeviceOptions}
                        value={String(tempState.index)}
                        onChange={onInputDeviceChange}
                        labelWidth="160px"
                    />
                    <ComboBox
                        label="Resolution"
                        items={resolutionOptions}
                        value={tempState.preferredResolution}
                        onChange={onResolutionChange}
                        labelWidth="160px"
                    />
                    <Slider
                        label="FPS"
                        value={tempState.fps}
                        onChange={onFpsChange}
                        min={1}
                        max={240}
                        minSlider={1}
                        maxSlider={240}
                        step={1}
                        decimalPlaces={0}
                        allowManualInput={true}
                        labelWidth="160px"
                    />
                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                        <Button
                            label={isResetting ? "Resetting..." : "Reset Camera"}
                            onClick={handleResetRequest}
                            color={isProcessing ? '#94a3b8' : '#ef4444'}
                            disabled={isProcessing}
                            style={{ height: '40px', display: 'flex', alignItems: 'center' }}
                        />
                    </div>

                    {/* Flip and Rotate */}
                    <HorizontalSeparator label="Flip and Rotate" fullWidth={true} bleed="1rem" />
                    <Switch
                        label="Flip Horizontally"
                        value={tempState.flipHorizontally}
                        onChange={(val) => handleRelevantFieldChange('flipHorizontally', val)}
                        labelWidth="160px"
                    />
                    <Switch
                        label="Flip Vertically"
                        value={tempState.flipVertically}
                        onChange={(val) => handleRelevantFieldChange('flipVertically', val)}
                        labelWidth="160px"
                    />
                    <ComboBox
                        label="Rotate (degrees)"
                        items={ROTATE_OPTIONS}
                        value={tempState.rotateDegrees}
                        onChange={(val) => handleRelevantFieldChange('rotateDegrees', val)}
                        labelWidth="160px"
                    />

                    {/* Crop */}
                    <HorizontalSeparator label="Crop" fullWidth={true} bleed="1rem" />
                    <Slider
                        label={`Top % (${Math.round((tempState.cropTop || 0) * resHeight / 100)} px)`}
                        value={tempState.cropTop}
                        onChange={(val) => handleRelevantFieldChange('cropTop', val)}
                        min={0}
                        max={100}
                        step={0.01}
                        decimalPlaces={2}
                        allowManualInput={true}
                        labelWidth="150px"
                    />
                    <Slider
                        label={`Left % (${Math.round((tempState.cropLeft || 0) * resWidth / 100)} px)`}
                        value={tempState.cropLeft}
                        onChange={(val) => handleRelevantFieldChange('cropLeft', val)}
                        min={0}
                        max={100}
                        step={0.01}
                        decimalPlaces={2}
                        allowManualInput={true}
                        labelWidth="150px"
                    />
                    <Slider
                        label={`Bottom % (${Math.round((tempState.cropBottom || 0) * resHeight / 100)} px)`}
                        value={tempState.cropBottom}
                        onChange={(val) => handleRelevantFieldChange('cropBottom', val)}
                        min={0}
                        max={100}
                        step={0.01}
                        decimalPlaces={2}
                        allowManualInput={true}
                        labelWidth="150px"
                    />
                    <Slider
                        label={`Right % (${Math.round((tempState.cropRight || 0) * resWidth / 100)} px)`}
                        value={tempState.cropRight}
                        onChange={(val) => handleRelevantFieldChange('cropRight', val)}
                        min={0}
                        max={100}
                        step={0.01}
                        decimalPlaces={2}
                        allowManualInput={true}
                        labelWidth="150px"
                    />

                    {/* Stretch */}
                    <HorizontalSeparator label="Stretch after Crop" fullWidth={true} bleed="1rem" />
                    <Switch
                        label="Stretch Enabled"
                        value={tempState.stretchEnabled}
                        onChange={(val) => handleRelevantFieldChange('stretchEnabled', val)}
                        labelWidth="150px"
                    />
                    <NumericInput
                        label="Width"
                        value={tempState.stretchWidth}
                        onChange={(val) => handleRelevantFieldChange('stretchWidth', val)}
                        min={64}
                        max={8192}
                        step={1}
                        decimalPlaces={0}
                        disabled={!tempState.stretchEnabled}
                        labelWidth="150px"
                        labelPosition="left"
                    />
                    <NumericInput
                        label="Height"
                        value={tempState.stretchHeight}
                        onChange={(val) => handleRelevantFieldChange('stretchHeight', val)}
                        min={64}
                        max={4608}
                        step={1}
                        decimalPlaces={0}
                        disabled={!tempState.stretchEnabled}
                        labelWidth="150px"
                        labelPosition="left"
                    />

                    {/* Image Settings */}
                    <HorizontalSeparator label="Image Settings" fullWidth={true} bleed="1rem" />
                    <Slider label="Brightness" value={tempState.brightness} onChange={(val) => handleRelevantFieldChange('brightness', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'brightness')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />
                    <Slider label="Contrast" value={tempState.contrast} onChange={(val) => handleRelevantFieldChange('contrast', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'contrast')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />
                    <Slider label="Hue" value={tempState.hue} onChange={(val) => handleRelevantFieldChange('hue', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'hue')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />
                    <Slider label="Saturation" value={tempState.saturation} onChange={(val) => handleRelevantFieldChange('saturation', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'saturation')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />
                    <Slider label="Sharpness" value={tempState.sharpness} onChange={(val) => handleRelevantFieldChange('sharpness', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'sharpness')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />
                    <Slider label="Gamma" value={tempState.gamma} onChange={(val) => handleRelevantFieldChange('gamma', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'gamma')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />
                    <Slider label="Gain" value={tempState.gain} onChange={(val) => handleRelevantFieldChange('gain', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'gain')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />
                    <Slider label="Backlight" value={tempState.backlight} onChange={(val) => handleRelevantFieldChange('backlight', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'backlight')} decimalPlaces={2} allowManualInput={true} labelWidth="100px" />

                    {/* Controls */}
                    <HorizontalSeparator label="Controls" fullWidth={true} bleed="1rem" />
                    <Switch label="Auto White Balance" value={tempState.autoWhiteBalance} onChange={(val) => handleRelevantFieldChange('autoWhiteBalance', val)} labelWidth="160px" />
                    <Slider label="White Balance Temp" value={tempState.whiteBalanceTemperature} onChange={(val) => handleRelevantFieldChange('whiteBalanceTemperature', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'whiteBalanceTemperature')} decimalPlaces={2} allowManualInput={true} labelWidth="160px" />
                    <Switch label="Auto Focus" value={tempState.autoFocus} onChange={(val) => handleRelevantFieldChange('autoFocus', val)} labelWidth="160px" />
                    <Slider label="Focus" value={tempState.focus} onChange={(val) => handleRelevantFieldChange('focus', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'focus')} decimalPlaces={2} allowManualInput={true} labelWidth="160px" />
                    <Switch label="Auto Exposure" value={tempState.autoExposure} onChange={(val) => handleRelevantFieldChange('autoExposure', val)} labelWidth="160px" />
                    <Slider label="Exposure" value={tempState.exposure} onChange={(val) => handleRelevantFieldChange('exposure', val)} {...getSliderPropsFromCapabilities(selectedDeviceCapabilities, 'exposure')} decimalPlaces={2} allowManualInput={true} labelWidth="160px" />

                    {/* Static Reticle */}
                    <HorizontalSeparator label="Static Reticle" fullWidth={true} bleed="1rem" />
                    <Slider
                        label="X coord"
                        value={tempState.reticleX}
                        onChange={(val) => updateTempState('reticleX', val)}
                        min={0}
                        max={1}
                        step={0.0001}
                        decimalPlaces={4}
                        allowManualInput={true}
                        labelWidth="150px"
                    />
                    <Slider
                        label="Y coord"
                        value={tempState.reticleY}
                        onChange={(val) => updateTempState('reticleY', val)}
                        min={0}
                        max={1}
                        step={0.0001}
                        decimalPlaces={4}
                        allowManualInput={true}
                        labelWidth="150px"
                    />
                    <ColorPicker
                        label="Color"
                        color={tempState.reticleColor}
                        onChange={(val) => updateTempState('reticleColor', val)}
                        showAlpha={true}
                        labelWidth="150px"
                    />
                    <ColorPicker
                        label="Outline"
                        color={tempState.reticleOutline}
                        onChange={(val) => updateTempState('reticleOutline', val)}
                        showAlpha={true}
                        labelWidth="150px"
                    />
                    <Slider
                        label="Size"
                        value={tempState.reticleSize}
                        onChange={(val) => updateTempState('reticleSize', val)}
                        min={0}
                        max={10}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="150px"
                    />

                    {/* Preview */}
                    <HorizontalSeparator label="Preview" fullWidth={true} bleed="1rem" />
                    <ColorPicker
                        label="Masked Area Color"
                        color={maskedAreaColor}
                        onChange={setMaskedAreaColor}
                        showAlpha={true}
                        labelWidth="150px"
                    />
                    <Slider
                        label="HTTP Stream Quality"
                        value={streamQuality}
                        onChange={onStreamQualityChange}
                        min={15}
                        max={100}
                        step={1}
                        decimalPlaces={0}
                        allowManualInput={true}
                        labelWidth="150px"
                    />
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <span style={{ width: '150px', flexShrink: 0 }}>Display Mode</span>
                        <MultiSwitch
                            options={[
                                { label: 'Raw', value: 0 },
                                { label: 'Mask', value: 1 },
                                { label: 'AI', value: 3 }
                            ]}
                            value={displayMode}
                            onChange={onDisplayModeChange}
                            style={{ flex: 1 }}
                        />
                    </div>
                    <Polygon
                        key={`modal-preview-${streamVersion}`}
                        style={{ width: '100%', height: 'auto', aspectRatio: '4/3', border: '1px solid #444' }}
                        src={previewEnabled ? `/api/cameras/stream/${activeModal}?mode=${displayMode}&quality=${streamQuality}&v=${streamVersion}` : cameraOffIcon}
                        stretchMode="fit"
                        mode="designer"
                        background="#000099"
                        zoomPanEnabled={true}
                        showReticle={tempState.reticleSize > 0}
                        reticleX={tempState.reticleX}
                        reticleY={tempState.reticleY}
                        reticleColor={tempState.reticleColor}
                        reticleOutlineColor={tempState.reticleOutline}
                        reticleSize={tempState.reticleSize}
                        fillColor={maskedAreaColor}
                        borderColor={maskedAreaColor.length >= 7 ? maskedAreaColor.slice(0, 7) + 'ff' : maskedAreaColor}
                        polygons={tempState.maskPolygons || []}
                        onChange={(newPolygons) => updateTempState('maskPolygons', newPolygons)}
                    />
                </ColumnLayout>
            </ModalWindow>
        );
    };

    // ========================
    // Render: Cancel Confirmation Modal
    // ========================
    const renderCancelConfirmModal = () => {
        if (!showCancelConfirmModal) return null;

        return (
            <ModalWindow
                isOpen={true}
                title="Unsaved Changes"
                onOk={handleCancelConfirm}
                onCancel={() => setShowCancelConfirmModal(false)}
                okLabel="Yes"
                cancelLabel="No"
            >
                <div style={{ padding: '1rem' }}>
                    Modifications you have made will be lost. Do you want to continue?
                </div>
            </ModalWindow>
        );
    };

    // ========================
    // Render: Reset Confirmation Modal
    // ========================
    const renderResetConfirmModal = () => {
        if (!showResetConfirmModal) return null;

        return (
            <ModalWindow
                isOpen={true}
                title="Reset Camera"
                onOk={handleResetConfirm}
                onCancel={handleResetCancel}
                okLabel="Yes"
                cancelLabel="No"
            >
                <div style={{ padding: '1rem' }}>
                    Are you sure you want to reset the camera to default settings?
                    This operation may take a minute.
                </div>
            </ModalWindow>
        );
    };

    // ========================
    // Render: Reset All Confirmation Modal
    // ========================
    const renderResetAllConfirmModal = () => {
        if (!showResetAllConfirmModal) return null;

        return (
            <ModalWindow
                isOpen={true}
                title="Reset All Cameras"
                onOk={handleResetAllConfirm}
                onCancel={handleResetAllCancel}
                okLabel={isResettingAll ? "Reseting ..." : "Yes"}
                cancelLabel="No"
                okDisabled={isResettingAll}
                cancelDisabled={isResettingAll}
            >
                <div style={{ padding: '1rem' }}>
                    Are you sure you want to reset all cameras?
                    This operation may take a minute.
                </div>
            </ModalWindow>
        );
    };

    // ========================
    // Main Render
    // ========================
    if (isLoading) {
        return (
            <div className="page-container">
                <StaticText text="Loading cameras..." />
            </div>
        );
    }

    return (
        <div className="page-container">
            <div style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '1rem', alignItems: 'stretch' }}>
                {CAMERA_CODES.map(code => renderCameraPanel(code))}
            </div>

            {renderEditModal()}
            {renderCancelConfirmModal()}
            {renderResetConfirmModal()}
            {renderResetAllConfirmModal()}
        </div>
    );
};

export default Cameras;

