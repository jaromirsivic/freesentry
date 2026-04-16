import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CameraViewport from './components/CameraViewport';
import ModalWindow from './components/ModalWindow';
import { useAIAgentActivation } from './contexts/AIAgentActivationContext.jsx';
import useDocumentFullscreen from './hooks/useDocumentFullscreen';
import {
    DEFAULT_CAMERA_SETTINGS,
    DEFAULT_RETICLE_SETTINGS,
    buildCameraStreamUrl,
    fetchReticleSettings,
    stopManagedCameraStream
} from './lib/cameraStream';

const AI_AGENT_CAMERA_SETTINGS = Object.freeze({
    ...DEFAULT_CAMERA_SETTINGS,
    selectedCamera: 'scope_camera',
    scopeCameraMode: 3
});

const AIAgent = () => {
    const {
        aiagentFullyActivated,
        activationLoaded,
        isActivationBusy,
        activateAIAgent
    } = useAIAgentActivation();
    const { isFullscreen, toggleFullscreen } = useDocumentFullscreen();
    const navigate = useNavigate();
    const [streamUrl, setStreamUrl] = useState(null);
    const [reticleSettings, setReticleSettings] = useState(() => ({
        ...DEFAULT_RETICLE_SETTINGS
    }));
    const [isViewerLoading, setIsViewerLoading] = useState(false);
    const [isActivationModalOpen, setIsActivationModalOpen] = useState(true);
    const wasActivatedRef = useRef(false);

    const buildStreamUrl = useCallback(() => {
        return `${buildCameraStreamUrl(AI_AGENT_CAMERA_SETTINGS)}&_t=${Date.now()}`;
    }, []);

    const stopVisibleStream = useCallback(async () => {
        return stopManagedCameraStream({
            cameraSettings: AI_AGENT_CAMERA_SETTINGS
        });
    }, []);

    useEffect(() => {
        let cancelled = false;
        const controller = new AbortController();

        if (!activationLoaded) {
            setIsViewerLoading(true);
            return () => {
                controller.abort();
            };
        }

        if (!aiagentFullyActivated) {
            setStreamUrl(null);
            setIsViewerLoading(false);
            return () => {
                controller.abort();
            };
        }

        setIsViewerLoading(true);

        const loadViewerState = async () => {
            try {
                const nextReticleSettings = await fetchReticleSettings({
                    cameraCode: AI_AGENT_CAMERA_SETTINGS.selectedCamera,
                    signal: controller.signal
                });

                if (cancelled || controller.signal.aborted) {
                    return;
                }

                setReticleSettings(nextReticleSettings);
                setStreamUrl(buildStreamUrl());
            } catch (error) {
                if (error?.name === 'AbortError') {
                    return;
                }

                console.error('Failed to load AI agent viewer:', error);

                if (cancelled || controller.signal.aborted) {
                    return;
                }

                setStreamUrl(buildStreamUrl());
            } finally {
                if (!cancelled && !controller.signal.aborted) {
                    setIsViewerLoading(false);
                }
            }
        };

        void loadViewerState();

        return () => {
            cancelled = true;
            controller.abort();
        };
    }, [activationLoaded, aiagentFullyActivated, buildStreamUrl]);

    useEffect(() => {
        if (wasActivatedRef.current && !aiagentFullyActivated) {
            setStreamUrl(null);
            void stopVisibleStream();
            setIsActivationModalOpen(true);
        }

        wasActivatedRef.current = aiagentFullyActivated;
    }, [aiagentFullyActivated, stopVisibleStream]);

    useEffect(() => {
        const handleMenuStateChange = async (event) => {
            const { isOpen, currentPath, isNavigating } = event.detail;

            if (currentPath !== '/ai-agent' || !aiagentFullyActivated) {
                return;
            }

            if (isOpen) {
                setStreamUrl(null);
                await stopVisibleStream();
            } else if (!isNavigating) {
                setStreamUrl(buildStreamUrl());
            }
        };

        window.addEventListener('menuStateChange', handleMenuStateChange);

        return () => {
            window.removeEventListener('menuStateChange', handleMenuStateChange);
        };
    }, [aiagentFullyActivated, buildStreamUrl, stopVisibleStream]);

    const handleCancelActivation = useCallback(() => {
        setIsActivationModalOpen(false);
        navigate('/');
    }, [navigate]);

    const handleStartAIAgent = useCallback(async () => {
        setIsActivationModalOpen(false);
        try {
            await activateAIAgent();
        } catch (error) {
            console.error('Failed to activate AI agent:', error);
            setIsActivationModalOpen(true);
        }
    }, [activateAIAgent]);

    if (!activationLoaded) {
        return <CameraViewport isLoading={true} />;
    }

    if (!aiagentFullyActivated) {
        return (
            <>
                <CameraViewport isLoading={isActivationBusy} />
                <ModalWindow
                    isOpen={isActivationModalOpen}
                    title="AI Agent"
                    okLabel={isActivationBusy ? 'Starting...' : 'Start AI'}
                    cancelLabel="Cancel"
                    onOk={handleStartAIAgent}
                    onCancel={handleCancelActivation}
                    okButtonColor="var(--red_primary)"
                    okDisabled={isActivationBusy}
                    cancelDisabled={isActivationBusy}
                >
                    <p style={{ margin: 0 }}>
                        Pressing the button below activates the AI Agent that will control all motors. This is not a simulation like on the Manual Control page. It starts the real AI Agent, which immediately begins executing the mission according to the configured AI setup.
                    </p>
                </ModalWindow>
            </>
        );
    }

    return (
        <CameraViewport
            isLoading={isViewerLoading}
            streamUrl={streamUrl}
            reticleSettings={reticleSettings}
            polygonMode="viewer"
            isFullscreen={isFullscreen}
            onToggleFullscreen={toggleFullscreen}
        />
    );
};

export default AIAgent;
