import React, { useState, useEffect, useCallback, useRef } from 'react';
import Polygon from './components/Polygon';
import ModalWindow from './components/ModalWindow';
import Switch from './components/Switch';
import ComboBox from './components/ComboBox';
import StaticText from './components/StaticText';
import HorizontalSeparator from './components/HorizontalSeparator';
import Joystick1D from './components/Joystick1D';
import Timer from './components/Timer';
import MultiSwitch from './components/MultiSwitch';
import Slider from './components/Slider';
import settingsIcon from './assets/icons/settings.svg';
import fullscreenIcon from './assets/icons/fullscreen.svg';
import fullscreenExitIcon from './assets/icons/fullscreenExit.svg';
import cameraOffIcon from './assets/icons/cameraOff.svg';

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
    const [reticleSettings, setReticleSettings] = useState({
        x: 0.5,
        y: 0.5,
        color: '#ff0000cc',
        outline: '#000000cc',
        size: 1.0
    });
    const [streamUrl, setStreamUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    
    // Motor settings from backend (max 4 motors)
    const [motors, setMotors] = useState([]);
    // Temporary motor state for modal (to allow cancel)
    const [tempMotors, setTempMotors] = useState([]);
    
    // Camera settings for manual control (persisted)
    const [cameraSettings, setCameraSettings] = useState({
        selectedCamera: 'scope_camera',
        streamQuality: 80,
        scopeCameraMode: 0,
        spotterCamera1Mode: 0,
        spotterCamera2Mode: 0,
        spotterCamera3Mode: 0
    });
    // Temporary camera settings for modal
    const [tempCameraSettings, setTempCameraSettings] = useState({
        selectedCamera: 'scope_camera',
        streamQuality: 80,
        scopeCameraMode: 0,
        spotterCamera1Mode: 0,
        spotterCamera2Mode: 0,
        spotterCamera3Mode: 0
    });
    // Saving state
    const [isSaving, setIsSaving] = useState(false);
    
    // Ref to track if component is mounted (for cleanup)
    const isMountedRef = useRef(true);
    // Ref to the Polygon component for stream termination
    const polygonRef = useRef(null);
    // Ref to store current camera settings for cleanup (avoids stale closure)
    const cameraSettingsRef = useRef(cameraSettings);
    
    // Keep cameraSettingsRef in sync with cameraSettings state
    useEffect(() => {
        cameraSettingsRef.current = cameraSettings;
    }, [cameraSettings]);
    
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
     * Fetch motor and camera settings from backend.
     */
    const fetchMotorSettings = useCallback(async () => {
        try {
            const response = await fetch('/api/manualcontrol/motors');
            const data = await response.json();
            
            if (data.success && data.motors) {
                // Limit to 4 motors
                const limitedMotors = data.motors.slice(0, 4);
                setMotors(limitedMotors);
            }
            
            // Also get camera settings
            if (data.success && data.camera) {
                setCameraSettings(data.camera);
            }
        } catch (error) {
            console.error('Failed to fetch motor settings:', error);
        }
    }, []);

    /**
     * Build stream URL from camera settings.
     */
    const buildStreamUrl = useCallback((settings) => {
        const camera = settings.selectedCamera || 'scope_camera';
        const quality = settings.streamQuality || 80;
        let mode = 0;
        switch (camera) {
            case 'scope_camera':
                mode = settings.scopeCameraMode || 0;
                break;
            case 'spotter_camera1':
                mode = settings.spotterCamera1Mode || 0;
                break;
            case 'spotter_camera2':
                mode = settings.spotterCamera2Mode || 0;
                break;
            case 'spotter_camera3':
                mode = settings.spotterCamera3Mode || 0;
                break;
            default:
                mode = 0;
        }
        return `/api/cameras/stream/${camera}?mode=${mode}&quality=${quality}`;
    }, []);

    /**
     * Fetch camera settings from new API.
     * Uses camera settings from manualControl in settings.json.
     */
    const fetchCameraSettings = useCallback(async () => {
        try {
            // First get manual control settings (includes camera config)
            const motorResponse = await fetch('/api/manualcontrol/motors');
            const motorData = await motorResponse.json();
            
            let camSettings = {
                selectedCamera: 'scope_camera',
                streamQuality: 80,
                scopeCameraMode: 0,
                spotterCamera1Mode: 0,
                spotterCamera2Mode: 0,
                spotterCamera3Mode: 0
            };
            
            if (motorData.success && motorData.camera) {
                camSettings = motorData.camera;
                setCameraSettings(camSettings);
            }
            
            // Get camera list for reticle settings
            const response = await fetch('/api/cameras/list');
            const data = await response.json();
            
            if (data.success && data.cameras) {
                // Use the selected camera for reticle settings
                const cameraConfig = data.cameras[camSettings.selectedCamera];
                
                if (cameraConfig) {
                    setReticleSettings({
                        x: cameraConfig.static_reticle_x ?? 0.5,
                        y: cameraConfig.static_reticle_y ?? 0.5,
                        color: cameraConfig.static_reticle_color ?? '#88ff00cc',
                        outline: cameraConfig.static_reticle_outline ?? '#000000cc',
                        size: cameraConfig.static_reticle_size ?? 1.0
                    });
                }
            }
            
            // Set the stream URL and enable timer
            if (isMountedRef.current) {
                setStreamUrl(buildStreamUrl(camSettings));
                setIsLoading(false);
                setTimerEnabled(true);
            }
            
        } catch (error) {
            console.error('Failed to fetch camera settings:', error);
            // Use defaults on error
            if (isMountedRef.current) {
                setStreamUrl('/api/cameras/stream/scope_camera?mode=0&quality=80');
                setIsLoading(false);
                setTimerEnabled(true);
            }
        }
    }, [buildStreamUrl]);

    /**
     * Stop camera stream via API.
     * Waits for success response before resolving.
     * @param {string} cameraCode - Camera code to stop.
     * @returns {Promise<boolean>} - True if successfully stopped.
     */
    const stopCameraStream = useCallback(async ({ cameraCode } = {}) => {
        try {
            const response = await fetch(`/api/cameras/stop/${cameraCode || cameraSettings.selectedCamera}`, { method: 'POST' });
            const data = await response.json();
            return data.success === true;
        } catch (err) {
            console.error('Failed to stop camera stream:', err);
            return false;
        }
    }, [cameraSettings.selectedCamera]);

    // Fetch camera and motor settings on mount
    useEffect(() => {
        isMountedRef.current = true;
        fetchCameraSettings();
        fetchMotorSettings();
        
        // Cleanup when component unmounts (navigation away)
        return () => {
            isMountedRef.current = false;
            // Stop the timer
            setTimerEnabled(false);
            // Show camera off image
            setStreamUrl(null);
            // Stop camera stream via API (fire-and-forget, use ref for current value)
            const selectedCamera = cameraSettingsRef.current?.selectedCamera || 'scope_camera';
            fetch(`/api/cameras/stop/${selectedCamera}`, { method: 'POST' }).catch(() => {});
        };
    }, [fetchCameraSettings, fetchMotorSettings]);

    // Handle browser close/refresh (beforeunload)
    useEffect(() => {
        const handleBeforeUnload = (event) => {
            // Show camera off image
            setStreamUrl(null);
            
            // Use sendBeacon for reliable delivery during page unload
            // Note: We can't wait for response during beforeunload
            // Stop all cameras just in case
            const allCameras = ['scope_camera', 'spotter_camera1', 'spotter_camera2', 'spotter_camera3'];
            for (const camera of allCameras) {
                navigator.sendBeacon(`/api/cameras/stop/${camera}`, '');
            }
            
            // For older browsers, return a message (though most modern browsers ignore it)
            event.preventDefault();
            event.returnValue = '';
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        
        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, [cameraSettings.selectedCamera]);

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

    // Listen for fullscreen changes (handles Escape key and other exit methods)
    useEffect(() => {
        const handleFullscreenChange = () => {
            const isNowFullscreen = !!(
                document.fullscreenElement ||
                document.webkitFullscreenElement ||
                document.mozFullScreenElement ||
                document.msFullscreenElement
            );
            setIsFullscreen(isNowFullscreen);
            
            // Add/remove class on body for reliable CSS targeting
            if (isNowFullscreen) {
                document.body.classList.add('is-fullscreen');
            } else {
                document.body.classList.remove('is-fullscreen');
            }
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
        document.addEventListener('mozfullscreenchange', handleFullscreenChange);
        document.addEventListener('MSFullscreenChange', handleFullscreenChange);

        return () => {
            document.removeEventListener('fullscreenchange', handleFullscreenChange);
            document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
            document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
            document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
            // Clean up class on unmount
            document.body.classList.remove('is-fullscreen');
        };
    }, []);

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
    }, []);

    // Track window resize for responsive joystick positioning
    useEffect(() => {
        const handleResize = () => {
            setWindowWidth(window.innerWidth);
        };

        window.addEventListener('resize', handleResize);
        
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    /**
     * Toggle fullscreen mode.
     */
    const toggleFullscreen = useCallback(async () => {
        try {
            if (!isFullscreen) {
                // Enter fullscreen
                const element = document.documentElement;
                if (element.requestFullscreen) {
                    await element.requestFullscreen();
                } else if (element.webkitRequestFullscreen) {
                    await element.webkitRequestFullscreen();
                } else if (element.mozRequestFullScreen) {
                    await element.mozRequestFullScreen();
                } else if (element.msRequestFullscreen) {
                    await element.msRequestFullscreen();
                }
            } else {
                // Exit fullscreen
                if (document.exitFullscreen) {
                    await document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    await document.webkitExitFullscreen();
                } else if (document.mozCancelFullScreen) {
                    await document.mozCancelFullScreen();
                } else if (document.msExitFullscreen) {
                    await document.msExitFullscreen();
                }
            }
        } catch (error) {
            console.error('Fullscreen toggle failed:', error);
        }
    }, [isFullscreen]);

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
                setMotors(tempMotors);
                setCameraSettings(tempCameraSettings);
                setIsModalOpen(false);
                setTempMotors([]);
                
                // 4. Update reticle settings from the newly selected camera
                try {
                    const camListResponse = await fetch('/api/cameras/list');
                    const camListData = await camListResponse.json();
                    if (camListData.success && camListData.cameras) {
                        const cameraConfig = camListData.cameras[tempCameraSettings.selectedCamera];
                        if (cameraConfig) {
                            setReticleSettings({
                                x: cameraConfig.static_reticle_x ?? 0.5,
                                y: cameraConfig.static_reticle_y ?? 0.5,
                                color: cameraConfig.static_reticle_color ?? '#88ff00cc',
                                outline: cameraConfig.static_reticle_outline ?? '#000000cc',
                                size: cameraConfig.static_reticle_size ?? 1.0
                            });
                        }
                    }
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
     * Send manual control action to backend.
     * Called periodically by Timer and on joystick movement.
     * 
     * This function ensures only one request is in flight at a time.
     * If a request is already pending, returns the last known result immediately.
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
            const motorsPayload = motors.slice(0, 4).map(motor => ({
                index: motor.index,
                value: motorJoystickValuesRef.current[motor.index] ?? 0.0
            }));
            
            const payload = {
                fullscreen: isFullscreen,
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
    }, [motors, isFullscreen]);

    // Handle Polygon joystick move events (controls motors 0 and 1)
    const handlePolygonJoystickMove = useCallback((coords) => {
        // coords: { x: -1 to 1, y: -1 to 1 }
        polygonJoystickRef.current = { x: coords.x, y: coords.y };
        // Send action immediately on joystick move
        sendManualControlAction();
    }, [sendManualControlAction]);

    // Handle Polygon joystick start
    const handlePolygonJoystickStart = useCallback(() => {
        // Nothing specific needed
    }, []);

    // Handle Polygon joystick end
    const handlePolygonJoystickEnd = useCallback(() => {
        // Reset polygon joystick position
        polygonJoystickRef.current = { x: 0, y: 0 };
        sendManualControlAction();
    }, [sendManualControlAction]);

    /**
     * Create handler for Joystick1D onChange event.
     * @param {number} motorIndex - The motor index this joystick controls.
     */
    const createMotorJoystickHandler = useCallback((motorIndex) => {
        return (data) => {
            // data.value contains the joystick value (-1 to 1)
            motorJoystickValuesRef.current[motorIndex] = data.value ?? 0;
            // Send action immediately on joystick move
            sendManualControlAction();
        };
    }, [sendManualControlAction]);

    /**
     * Create handler for Joystick1D onEnd event.
     * @param {number} motorIndex - The motor index this joystick controls.
     */
    const createMotorJoystickEndHandler = useCallback((motorIndex) => {
        return () => {
            const motor = motors.find(m => m.index === motorIndex);
            if (!motor || motor.mode !== 'slider') {
                motorJoystickValuesRef.current[motorIndex] = 0;
            }
            sendManualControlAction();
        };
    }, [sendManualControlAction, motors]);

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
        const joystickHeight = 60;
        const buttonWidth = 48;
        const buttonMargin = 16;
        const buttonGap = 16;
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
        return (
            <div style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#000'
            }}>
                <span style={{ color: '#fff' }}>Loading...</span>
            </div>
        );
    }

    return (
        <div style={{
                width: '100%',
                height: '100%',
                margin: 0,
                padding: 0,
                overflow: 'hidden',
                position: 'relative'
            }}
        >
            <Polygon
                ref={polygonRef}
                src={streamUrl || cameraOffIcon}
                stretchMode="fit"
                background="#000000"
                mode="joystick"
                zoomPanEnabled={true}
                showReticle={true}
                reticleX={reticleSettings.x}
                reticleY={reticleSettings.y}
                reticleColor={reticleSettings.color}
                reticleOutlineColor={reticleSettings.outline}
                reticleSize={reticleSettings.size}
                joystickLineMaxLength={0.33}
                onJoystickMove={handlePolygonJoystickMove}
                onJoystickStart={handlePolygonJoystickStart}
                onJoystickEnd={handlePolygonJoystickEnd}
                style={{
                    width: '100%',
                    height: '100%'
                }}
            />

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

            {/* Fullscreen button - left of Setup button */}
            <button
                onClick={toggleFullscreen}
                style={{
                    ...buttonStyle,
                    right: '80px' // 16px margin + 48px button width + 16px gap
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.opacity = '1';
                    e.currentTarget.style.backgroundColor = '#885500';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.opacity = '0.8';
                    e.currentTarget.style.backgroundColor = '#887700';
                }}
                title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
                <img 
                    src={isFullscreen ? fullscreenExitIcon : fullscreenIcon} 
                    alt={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'} 
                    width="24" 
                    height="24" 
                />
            </button>

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

            <ModalWindow
                isOpen={isModalOpen}
                title="Manual Control Settings"
                onCancel={handleCloseModal}
                okLabel={isSaving ? "Saving..." : "Save"}
                onOk={handleSaveMotors}
                okDisabled={isSaving}
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

            {/* Timer for periodic action updates (0.25 second interval) */}
            <Timer
                enabled={timerEnabled}
                interval={0.1}
                onInterval={sendManualControlAction}
            />
        </div>
    );
};

export default ManualControl;
