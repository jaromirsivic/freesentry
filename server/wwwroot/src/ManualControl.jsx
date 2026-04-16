import React, { useState, useEffect, useCallback, useRef } from 'react';
import CameraViewport from './components/CameraViewport';
import ModalWindow from './components/ModalWindow';
import Switch from './components/Switch';
import ComboBox from './components/ComboBox';
import StaticText from './components/StaticText';
import HorizontalSeparator from './components/HorizontalSeparator';
import Joystick1D from './components/Joystick1D';
import Timer from './components/Timer';
import MultiSwitch from './components/MultiSwitch';
import Slider from './components/Slider';
import useDocumentFullscreen from './hooks/useDocumentFullscreen';
import settingsIcon from './assets/icons/settings.svg';
import {
    DEFAULT_CAMERA_SETTINGS,
    DEFAULT_RETICLE_SETTINGS,
    buildCameraStreamUrl,
    buildStopCameraUrl,
    fetchReticleSettings,
    queueDismissalCameraStop,
    stopManagedCameraStream
} from './lib/cameraStream';

/**
 * ManualControl page - Full-screen camera view with joystick control.
 * 
 * Features:
 * - Full-screen Polygon component filling entire content area
 * - Live video feed from primary camera (image_cropped_resized)
 * - Zoom and pan support
 * - Joystick control overlay (up to 4 joysticks based on enabled motors)
 * - Reticle display based on camera settings
 * - Setup button for settings access
 * - Fullscreen mode toggle
 */
const ManualControl = () => {
    // Reticle settings from camera configuration
    const [reticleSettings, setReticleSettings] = useState(() => ({
        ...DEFAULT_RETICLE_SETTINGS
    }));
    const [streamUrl, setStreamUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const { isFullscreen, isFullscreenRef, toggleFullscreen } = useDocumentFullscreen();
    
    // Motor settings from backend (max 4 motors)
    const [motors, setMotors] = useState([]);
    // Temporary motor state for modal (to allow cancel)
    const [tempMotors, setTempMotors] = useState([]);
    
    // Camera settings for manual control (persisted)
    const [cameraSettings, setCameraSettings] = useState(() => ({
        ...DEFAULT_CAMERA_SETTINGS
    }));
    // Temporary camera settings for modal
    const [tempCameraSettings, setTempCameraSettings] = useState(() => ({
        ...DEFAULT_CAMERA_SETTINGS
    }));
    // Saving state
    const [isSaving, setIsSaving] = useState(false);
    const [isResetting, setIsResetting] = useState(false);
    
    // Ref to track if component is mounted (for cleanup)
    const isMountedRef = useRef(true);
    // Ref to store current camera settings for cleanup (avoids stale closure)
    const cameraSettingsRef = useRef(cameraSettings);
    const motorsRef = useRef(motors);
    const initialLoadGenerationRef = useRef(0);
    const dismissalStopQueuedRef = useRef(false);
    
    // Keep cameraSettingsRef in sync with cameraSettings state
    useEffect(() => {
        cameraSettingsRef.current = cameraSettings;
    }, [cameraSettings]);

    useEffect(() => {
        motorsRef.current = motors;
    }, [motors]);
    
    // Window dimensions for responsive joystick positioning
    const [windowWidth, setWindowWidth] = useState(window.innerWidth);
    
    // Polygon joystick position (for motors 0 and 1)
    const polygonJoystickRef = useRef({ x: 0, y: 0 });
    
    // Motor Joystick1D values (indexed by motor index)
    const motorJoystickValuesRef = useRef({});
    
    // Timer enabled state - starts when component is loaded
    const [timerEnabled, setTimerEnabled] = useState(false);
    
    // Request-in-flight tracking for sendManualControlAction
    const requestInFlightRef = useRef(false);
    const lastActionResultRef = useRef({ success: true, motors: [] });

    /**
     * Build stream URL from camera settings.
     */
    const buildStreamUrl = useCallback((settings) => {
        return `${buildCameraStreamUrl(settings)}&_t=${Date.now()}`;
    }, []);

    const queueDismissalStop = useCallback((cameraCode) => {
        if (dismissalStopQueuedRef.current) {
            return;
        }

        dismissalStopQueuedRef.current = true;
        queueDismissalCameraStop({
            cameraCode,
            cameraSettings: cameraSettingsRef.current
        });
    }, []);

    /**
     * Stop camera stream via API.
     * Waits for success response before resolving.
     * @param {string} cameraCode - Camera code to stop.
     * @returns {Promise<boolean>} - True if successfully stopped.
     */
    const stopCameraStream = useCallback(async ({ cameraCode } = {}) => {
        return stopManagedCameraStream({
            cameraCode,
            cameraSettings: cameraSettingsRef.current
        });
    }, []);

    // Fetch camera and motor settings on mount
    useEffect(() => {
        isMountedRef.current = true;
        const controller = new AbortController();
        const loadGeneration = initialLoadGenerationRef.current + 1;
        initialLoadGenerationRef.current = loadGeneration;

        const canApplyLoad = () => (
            isMountedRef.current
            && !controller.signal.aborted
            && initialLoadGenerationRef.current === loadGeneration
        );

        const loadInitialState = async () => {
            const defaultCameraSettings = {
                ...DEFAULT_CAMERA_SETTINGS
            };

            let nextMotors = null;
            let nextCameraSettings = defaultCameraSettings;
            let nextReticleSettings = null;

            try {
                const motorResponse = await fetch('/api/manualcontrol/motors', {
                    signal: controller.signal
                });
                const motorData = await motorResponse.json();

                if (motorData.success && Array.isArray(motorData.motors)) {
                    nextMotors = motorData.motors.slice(0, 4);
                }

                if (motorData.success && motorData.camera) {
                    nextCameraSettings = motorData.camera;
                }

                try {
                    nextReticleSettings = await fetchReticleSettings({
                        cameraCode: nextCameraSettings.selectedCamera,
                        signal: controller.signal
                    });
                } catch (cameraListError) {
                    if (cameraListError?.name === 'AbortError') {
                        throw cameraListError;
                    }

                    console.error('Failed to fetch reticle settings:', cameraListError);
                }

                if (!canApplyLoad()) {
                    return;
                }

                if (nextMotors) {
                    motorsRef.current = nextMotors;
                    setMotors(nextMotors);
                }

                cameraSettingsRef.current = nextCameraSettings;
                setCameraSettings(nextCameraSettings);

                if (nextReticleSettings) {
                    setReticleSettings(nextReticleSettings);
                }

                setStreamUrl(buildStreamUrl(nextCameraSettings));
                setIsLoading(false);
                setTimerEnabled(true);
            } catch (error) {
                if (error?.name === 'AbortError') {
                    return;
                }

                console.error('Failed to load manual control settings:', error);

                if (!canApplyLoad()) {
                    return;
                }

                setStreamUrl(buildStreamUrl(defaultCameraSettings));
                setIsLoading(false);
                setTimerEnabled(true);
            }
        };

        void loadInitialState();
        
        // Cleanup when component unmounts (navigation away)
        return () => {
            isMountedRef.current = false;
            initialLoadGenerationRef.current += 1;
            controller.abort();
            // Stop the timer
            setTimerEnabled(false);
            // Show camera off image
            setStreamUrl(null);
            // SPA route changes use the normal stop endpoint; page dismissal has its own keepalive fallback.
            if (!dismissalStopQueuedRef.current) {
                void fetch(buildStopCameraUrl({ cameraSettings: cameraSettingsRef.current }), {
                    method: 'POST'
                }).catch(() => {});
            }
        };
    }, [buildStreamUrl]);

    // Handle browser dismissal without forcing an unload confirmation prompt.
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

    // Handle main menu open/close - pause streaming when menu opens, resume when it closes
    useEffect(() => {
        const handleMenuStateChange = async (event) => {
            const { isOpen, currentPath, isNavigating } = event.detail;
            
            // Only respond if we're on the manual-control page
            if (currentPath !== '/manual-control') return;
            
            if (isOpen) {
                // Menu opened - stop streaming and show camera off
                setTimerEnabled(false);
                setStreamUrl(null);
                await stopCameraStream({ cameraCode: cameraSettings.selectedCamera });
            } else if (!isNavigating) {
                // Menu closed AND user is NOT navigating away - restart streaming
                // (If user is navigating away, the component will unmount and cleanup will handle stopping the stream)
                setStreamUrl(buildStreamUrl(cameraSettings));
                setTimerEnabled(true);
            }
        };

        window.addEventListener('menuStateChange', handleMenuStateChange);
        
        return () => {
            window.removeEventListener('menuStateChange', handleMenuStateChange);
        };
    }, [cameraSettings, stopCameraStream, buildStreamUrl]);

    // Handle page visibility change to pause/resume stream
    useEffect(() => {
        const handleVisibilityChange = () => {
            if (document.hidden) {
                // Page is hidden, could pause stream here if needed
                console.log('Manual Control: Page hidden');
            } else {
                // Page is visible again
                console.log('Manual Control: Page visible');
            }
        };

        document.addEventListener('visibilitychange', handleVisibilityChange);
        
        return () => {
            document.removeEventListener('visibilitychange', handleVisibilityChange);
        };
    }, [isFullscreenRef]);

    // Track window resize for responsive joystick positioning
    useEffect(() => {
        const handleResize = () => {
            setWindowWidth(window.innerWidth);
        };

        window.addEventListener('resize', handleResize);
        
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, [isFullscreenRef]);

    /**
     * Handle Setup button click - open modal settings.
     */
    const handleSetupClick = useCallback(() => {
        // Create a copy of motors for temp editing
        setTempMotors(motors.map(m => ({ ...m })));
        // Create a copy of camera settings for temp editing
        setTempCameraSettings({ ...cameraSettings });
        setIsModalOpen(true);
    }, [motors, cameraSettings]);

    /**
     * Handle modal cancel - discard changes and restart stream.
     */
    const handleCloseModal = useCallback(() => {
        setIsModalOpen(false);
        setTempMotors([]);
        setTempCameraSettings({ ...cameraSettings });
        // Restart the stream with current settings
        setStreamUrl(buildStreamUrl(cameraSettings));
    }, [cameraSettings, buildStreamUrl]);

    /**
     * Handle motor switch toggle in modal.
     */
    const handleMotorToggle = useCallback((motorIndex) => {
        setTempMotors(prev => prev.map(m => 
            m.index === motorIndex ? { ...m, enabled: !m.enabled } : m
        ));
    }, []);

    /**
     * Handle motor mode change in modal.
     */
    const handleMotorModeChange = useCallback((motorIndex, newMode) => {
        setTempMotors(prev => prev.map(m => 
            m.index === motorIndex ? { ...m, mode: newMode } : m
        ));
    }, []);

    /**
     * Handle camera setting change - disables main stream.
     */
    const handleCameraSettingChange = useCallback((key, value) => {
        setTempCameraSettings(prev => ({ ...prev, [key]: value }));
        // Stop the main stream (show cameraOff.svg behind modal)
        setStreamUrl(null);
    }, []);

    /**
     * Save motor and camera settings to backend.
     * Sequence: stop camera → wait for success → wait 100ms → save → restart stream.
     */
    const handleSaveMotors = useCallback(async () => {
        try {
            setIsSaving(true);
            
            // 1. Call api/cameras/stop and wait for success
            const stopSuccess = await stopCameraStream({ cameraCode: tempCameraSettings.selectedCamera });
            if (!stopSuccess) {
                console.warn('Camera stop did not return success, continuing anyway');
            }
            
            // 2. Wait additional 100ms
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // 3. Save settings to backend
            const response = await fetch('/api/manualcontrol/motors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    motors: tempMotors.map(m => ({
                        index: m.index,
                        enabled: m.enabled,
                        mode: m.mode || 'joystick'
                    })),
                    camera: {
                        selectedCamera: tempCameraSettings.selectedCamera,
                        streamQuality: tempCameraSettings.streamQuality,
                        scopeCameraMode: tempCameraSettings.scopeCameraMode,
                        spotterCamera1Mode: tempCameraSettings.spotterCamera1Mode,
                        spotterCamera2Mode: tempCameraSettings.spotterCamera2Mode,
                        spotterCamera3Mode: tempCameraSettings.spotterCamera3Mode
                    }
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update main state with saved values
                motorsRef.current = tempMotors;
                setMotors(tempMotors);
                setCameraSettings(tempCameraSettings);
                setIsModalOpen(false);
                setTempMotors([]);
                
                // 4. Update reticle settings from the newly selected camera
                try {
                    setReticleSettings(await fetchReticleSettings({
                        cameraCode: tempCameraSettings.selectedCamera
                    }));
                } catch (camErr) {
                    console.error('Failed to update reticle settings:', camErr);
                }
                
                // 5. Restart video stream with new settings
                setStreamUrl(buildStreamUrl(tempCameraSettings));
            } else {
                console.error('Failed to save motor settings');
            }
        } catch (error) {
            console.error('Error saving motor settings:', error);
        } finally {
            setIsSaving(false);
        }
    }, [tempMotors, tempCameraSettings, stopCameraStream, buildStreamUrl]);

    /**
     * Reset AI engagement counters and runtime state immediately.
     */
    const handleResetAiEngagements = useCallback(async () => {
        try {
            setIsResetting(true);

            const response = await fetch('/api/manualcontrol/reset-ai-engagements', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.detail || 'Failed to reset AI engagements');
            }
        } catch (error) {
            console.error('Error resetting AI engagements:', error);
        } finally {
            setIsResetting(false);
        }
    }, []);

    /**
     * Send manual control action to backend.
     * Called by the Timer loop, which serializes requests and
     * always sends the latest state snapshot from refs.
     * 
     * The request-in-flight guard stays as a defensive fallback.
     * Includes a 5-second timeout for the request.
     */
    const sendManualControlAction = useCallback(async () => {
        // If a request is already in flight, return last known result immediately
        if (requestInFlightRef.current) {
            return lastActionResultRef.current;
        }
        
        try {
            // Mark request as in flight
            requestInFlightRef.current = true;
            
            // Build motors array from all motors displayed in modal
            const motorsPayload = motorsRef.current.slice(0, 4).map(motor => ({
                index: motor.index,
                value: motorJoystickValuesRef.current[motor.index] ?? 0.0
            }));
            
            const payload = {
                fullscreen: isFullscreenRef.current,
                joystick: {
                    x: polygonJoystickRef.current.x,
                    y: polygonJoystickRef.current.y
                },
                motors: motorsPayload
            };
            
            // Create abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            try {
                const response = await fetch('/api/manualcontrol/action', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload),
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                const data = await response.json();
                
                if (!data.success) {
                    console.error('Manual control action failed:', data);
                }
                
                // Store last known result
                lastActionResultRef.current = data;
                return data;
                
            } catch (fetchError) {
                clearTimeout(timeoutId);
                if (fetchError.name === 'AbortError') {
                    console.warn('Manual control action request timed out after 5 seconds');
                } else {
                    throw fetchError;
                }
                return lastActionResultRef.current;
            }
            
        } catch (error) {
            console.error('Error sending manual control action:', error);
            return lastActionResultRef.current;
        } finally {
            // Mark request as complete
            requestInFlightRef.current = false;
        }
    }, [isFullscreenRef]);

    // Handle Polygon joystick move events (controls motors 0 and 1)
    const handlePolygonJoystickMove = useCallback((coords) => {
        // coords: { x: -1 to 1, y: -1 to 1 }
        polygonJoystickRef.current = { x: coords.x, y: coords.y };
    }, []);

    // Handle Polygon joystick start
    const handlePolygonJoystickStart = useCallback(() => {
        // Nothing specific needed
    }, []);

    // Handle Polygon joystick end
    const handlePolygonJoystickEnd = useCallback(() => {
        // Reset polygon joystick position
        polygonJoystickRef.current = { x: 0, y: 0 };
    }, []);

    /**
     * Create handler for Joystick1D onChange event.
     * @param {number} motorIndex - The motor index this joystick controls.
     */
    const createMotorJoystickHandler = useCallback((motorIndex) => {
        return (data) => {
            // data.value contains the joystick value (-1 to 1)
            motorJoystickValuesRef.current[motorIndex] = data.value ?? 0;
        };
    }, []);

    /**
     * Create handler for Joystick1D onEnd event.
     * @param {number} motorIndex - The motor index this joystick controls.
     */
    const createMotorJoystickEndHandler = useCallback((motorIndex) => {
        return () => {
            const motor = motorsRef.current.find(m => m.index === motorIndex);
            if (!motor || motor.mode !== 'slider') {
                motorJoystickValuesRef.current[motorIndex] = 0;
            }
        };
    }, []);

    // Button style - 20% transparent (opacity 0.8), z-index below menu (48-50)
    // Background color matches menu button (#887700)
    const buttonStyle = {
        position: 'absolute',
        bottom: '16px',
        zIndex: 10,
        width: '48px',
        height: '48px',
        padding: '8px',
        backgroundColor: '#887700',
        border: 'none',
        borderRadius: '8px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        opacity: 0.8,
        transition: 'opacity 0.2s, background-color 0.2s'
    };

    /**
     * Calculate first joystick (Blue) position based on window width.
     * 
     * Logic:
     * - Default: bottom-center, same line as fullscreen/setup buttons (bottom: 16px)
     * - If fullscreen button would collide, move joystick left
     * - If joystick would touch left edge, jump up and center horizontally
     */
    const getFirstJoystickPosition = useCallback(() => {
        const joystickWidth = 200;
        const buttonWidth = 48;
        const buttonMargin = 16;
        const margin = 16;
        
        // Fullscreen button left edge position from left of screen
        // Right: 80px means button right edge is 80px from right
        // Fullscreen button: right edge at windowWidth - 80, left edge at windowWidth - 80 - 48
        const fullscreenLeftEdge = windowWidth - 80 - buttonWidth;
        
        // Joystick centered: center at windowWidth/2, right edge at windowWidth/2 + joystickWidth/2
        const joystickCenterX = windowWidth / 2;
        const joystickRightEdge = joystickCenterX + joystickWidth / 2;
        
        // Default position: bottom-center, same line as buttons
        const defaultPosition = {
            bottom: `${buttonMargin}px`,
            left: '50%',
            transform: 'translateX(-50%)'
        };
        
        // Check if there's enough space (joystick right edge + margin < fullscreen left edge)
        if (joystickRightEdge + margin < fullscreenLeftEdge) {
            // No collision, use default centered position
            return defaultPosition;
        }
        
        // Collision detected - calculate how much to retreat left
        // Move joystick so its right edge is at fullscreenLeftEdge - margin
        const requiredRightEdge = fullscreenLeftEdge - margin;
        const newCenterX = requiredRightEdge - joystickWidth / 2;
        const newLeftEdge = newCenterX - joystickWidth / 2;
        
        // Check if joystick would go off the left side
        if (newLeftEdge <= margin) {
            // Jump up and center horizontally
            // Position at the same y as the original design (vertical middle area)
            // Using bottom: 50% - half joystick height for vertical center-ish
            return {
                bottom: '80px',  // Original elevated position
                left: '50%',
                transform: 'translateX(-50%)'
            };
        }
        
        // Retreat left while staying at bottom
        return {
            bottom: `${buttonMargin}px`,
            left: `${newCenterX}px`,
            transform: 'translateX(-50%)'
        };
    }, [windowWidth]);

    // Joystick configurations based on motor position
    // Motor 1 (index 0): Blue, horizontal, bottom-center (responsive - handled by getFirstJoystickPosition)
    // Motor 2 (index 1): Purple, vertical, middle-right
    // Motor 3 (index 2): Yellow, horizontal, top-center
    // Motor 4 (index 3): Brown, vertical, middle-left
    const joystickConfigs = [
        {
            // Default fallback position (actual position computed by getFirstJoystickPosition)
            position: { bottom: '16px', left: '50%', transform: 'translateX(-50%)' },
            orientation: 'horizontal',
            colors: { ruler: '#3b82f6', button: '#3b82f6', outline: '#2563eb' }
        },
        {
            position: { top: '50%', right: '16px', transform: 'translateY(-50%)' },
            orientation: 'vertical',
            colors: { ruler: '#7134ed', button: '#8b5cf6', outline: '#7c3aed' }
        },
        {
            position: { top: '16px', left: '50%', transform: 'translateX(-50%)' },
            orientation: 'horizontal',
            colors: { ruler: '#eab308', button: '#eab308', outline: '#ca8a04' }
        },
        {
            position: { top: '50%', left: '16px', transform: 'translateY(-50%)' },
            orientation: 'vertical',
            colors: { ruler: '#92400e', button: '#92400e', outline: '#78350f' }
        }
    ];

    if (isLoading) {
        return <CameraViewport isLoading={true} />;
    }

    return (
        <>
            <CameraViewport
                streamUrl={streamUrl}
                reticleSettings={reticleSettings}
                polygonMode="joystick"
                isFullscreen={isFullscreen}
                onToggleFullscreen={toggleFullscreen}
                fullscreenButtonRight="80px"
                onJoystickMove={handlePolygonJoystickMove}
                onJoystickStart={handlePolygonJoystickStart}
                onJoystickEnd={handlePolygonJoystickEnd}
            >
                {/* Render Joystick1D components for enabled motors */}
                {motors.map((motor, idx) => {
                    if (!motor.enabled || idx >= 4) return null;
                    
                    const config = joystickConfigs[idx];
                    const isHorizontal = config.orientation === 'horizontal';
                    
                    // Use dynamic position for first joystick, static for others
                    const position = idx === 0 ? getFirstJoystickPosition() : config.position;
                    
                    // Use motor color from settings, or fallback to config colors
                    const motorColor = motor.color || config.colors.ruler;
                    
                    // Derive outline color (slightly darker)
                    const deriveOutlineColor = (color) => {
                        // Simple darkening: reduce RGB values by 20%
                        if (color.startsWith('#') && color.length === 7) {
                            const r = Math.max(0, Math.floor(parseInt(color.slice(1, 3), 16) * 0.8));
                            const g = Math.max(0, Math.floor(parseInt(color.slice(3, 5), 16) * 0.8));
                            const b = Math.max(0, Math.floor(parseInt(color.slice(5, 7), 16) * 0.8));
                            return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
                        }
                        return color;
                    };
                    
                    return (
                        <div
                            key={motor.index}
                            style={{
                                position: 'absolute',
                                zIndex: 5,
                                ...position
                            }}
                        >
                            <Joystick1D
                                orientation={config.orientation}
                                mode={motor.mode || 'joystick'}
                                width={isHorizontal ? 200 : 60}
                                height={isHorizontal ? 60 : 200}
                                rulerColor={motorColor}
                                buttonColor={motorColor}
                                buttonOutline={deriveOutlineColor(motorColor)}
                                backgroundColor="rgba(0, 0, 0, 0.2)"
                                rulerShowText={true}
                                rulerLineDistance={0.2}
                                valueOrigin={0}
                                minValue={-1}
                                maxValue={1}
                                snapAnimationDuration={0.1}
                                onChange={createMotorJoystickHandler(motor.index)}
                                onEnd={createMotorJoystickEndHandler(motor.index)}
                            />
                        </div>
                    );
                })}

                {/* Setup button - right bottom corner */}
                <button
                    onClick={handleSetupClick}
                    style={{
                        ...buttonStyle,
                        right: '16px'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.opacity = '1';
                        e.currentTarget.style.backgroundColor = '#885500';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.opacity = '0.8';
                        e.currentTarget.style.backgroundColor = '#887700';
                    }}
                    title="Setup"
                >
                    <img src={settingsIcon} alt="Setup" width="24" height="24" />
                </button>
            </CameraViewport>

            <ModalWindow
                isOpen={isModalOpen}
                title="Manual Control Settings"
                onCancel={handleCloseModal}
                okLabel={isSaving ? "Saving..." : "Save"}
                onOk={handleSaveMotors}
                okDisabled={isSaving || isResetting}
            >
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '0.5rem' }}>
                    {/* Cameras Section */}
                    <HorizontalSeparator label="Cameras" fullWidth={true} bleed="1.5rem" />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
                        <ComboBox
                            label="Camera"
                            items={[
                                { label: 'Scope', value: 'scope_camera' },
                                { label: 'Spotter Camera 1', value: 'spotter_camera1' },
                                { label: 'Spotter Camera 2', value: 'spotter_camera2' },
                                { label: 'Spotter Camera 3', value: 'spotter_camera3' }
                            ]}
                            value={tempCameraSettings.selectedCamera}
                            onChange={(val) => handleCameraSettingChange('selectedCamera', val)}
                            labelWidth="150px"
                        />
                        <Slider
                            label="HTTP Stream Quality"
                            value={tempCameraSettings.streamQuality}
                            onChange={(val) => handleCameraSettingChange('streamQuality', val)}
                            min={15}
                            max={100}
                            step={1}
                            decimalPlaces={0}
                            allowManualInput={true}
                            labelWidth="150px"
                        />
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <span style={{ width: '150px', flexShrink: 0 }}>Scope Camera</span>
                            <MultiSwitch
                                options={[
                                    { label: 'Raw', value: 0 },
                                    { label: 'Mask', value: 1 },
                                    { label: 'AI', value: 3 }
                                ]}
                                value={tempCameraSettings.scopeCameraMode}
                                onChange={(val) => handleCameraSettingChange('scopeCameraMode', val)}
                                disabled={tempCameraSettings.selectedCamera !== 'scope_camera'}
                                style={{ flex: 1 }}
                            />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <span style={{ width: '150px', flexShrink: 0 }}>Spotter Camera 1</span>
                            <MultiSwitch
                                options={[
                                    { label: 'Raw', value: 0 },
                                    { label: 'Mask', value: 1 },
                                    { label: 'AI', value: 3 }
                                ]}
                                value={tempCameraSettings.spotterCamera1Mode}
                                onChange={(val) => handleCameraSettingChange('spotterCamera1Mode', val)}
                                disabled={tempCameraSettings.selectedCamera !== 'spotter_camera1'}
                                style={{ flex: 1 }}
                            />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <span style={{ width: '150px', flexShrink: 0 }}>Spotter Camera 2</span>
                            <MultiSwitch
                                options={[
                                    { label: 'Raw', value: 0 },
                                    { label: 'Mask', value: 1 },
                                    { label: 'AI', value: 3 }
                                ]}
                                value={tempCameraSettings.spotterCamera2Mode}
                                onChange={(val) => handleCameraSettingChange('spotterCamera2Mode', val)}
                                disabled={tempCameraSettings.selectedCamera !== 'spotter_camera2'}
                                style={{ flex: 1 }}
                            />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <span style={{ width: '150px', flexShrink: 0 }}>Spotter Camera 3</span>
                            <MultiSwitch
                                options={[
                                    { label: 'Raw', value: 0 },
                                    { label: 'Mask', value: 1 },
                                    { label: 'AI', value: 3 }
                                ]}
                                value={tempCameraSettings.spotterCamera3Mode}
                                onChange={(val) => handleCameraSettingChange('spotterCamera3Mode', val)}
                                disabled={tempCameraSettings.selectedCamera !== 'spotter_camera3'}
                                style={{ flex: 1 }}
                            />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <span style={{ width: '150px', flexShrink: 0 }} />
                            <button
                                type="button"
                                onClick={handleResetAiEngagements}
                                disabled={isResetting || isSaving}
                                style={{
                                    padding: '0.625rem 1rem',
                                    backgroundColor: '#dc2626',
                                    border: '1px solid #b91c1c',
                                    borderRadius: '0.375rem',
                                    color: '#ffffff',
                                    fontWeight: 600,
                                    cursor: isResetting || isSaving ? 'not-allowed' : 'pointer',
                                    opacity: isResetting || isSaving ? 0.7 : 1,
                                    transition: 'opacity 0.2s ease, background-color 0.2s ease'
                                }}
                            >
                                {isResetting ? 'Resetting...' : 'Reset AI Engagements'}
                            </button>
                        </div>
                    </div>

                    {/* Motor / Device Section */}
                    {tempMotors.length === 0 && (
                        <div style={{ color: '#6b7280', fontStyle: 'italic' }}>
                            No motors configured
                        </div>
                    )}
                    
                    {tempMotors.slice(0, 4).map((motor, idx) => (
                        <div key={motor.index}>
                            <HorizontalSeparator label="Motor / Device" fullWidth={true} bleed="1.5rem" />
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{
                                        width: '16px',
                                        height: '16px',
                                        borderRadius: '50%',
                                        backgroundColor: motor.color || '#888888',
                                        flexShrink: 0
                                    }} />
                                    <StaticText text={<><span style={{ fontWeight: 500 }}>Name:</span> {motor.name || `Motor ${idx + 1}`}</>} />
                                </div>
                                <Switch
                                    label="Enabled"
                                    value={motor.enabled}
                                    onChange={() => handleMotorToggle(motor.index)}
                                    labelWidth="80px"
                                />
                                <ComboBox
                                    label="Mode"
                                    items={[
                                        { label: 'Joystick', value: 'joystick' },
                                        { label: 'Slider', value: 'slider' }
                                    ]}
                                    value={motor.mode || 'joystick'}
                                    onChange={(val) => handleMotorModeChange(motor.index, val)}
                                    disabled={!motor.enabled}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </ModalWindow>

            {/* Timer for periodic action updates (20 ms cadence) */}
            <Timer
                enabled={timerEnabled}
                interval={0.02}
                onInterval={sendManualControlAction}
            />
        </>
    );
};

export default ManualControl;
