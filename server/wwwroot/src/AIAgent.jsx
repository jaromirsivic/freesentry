import React, { useCallback, useEffect, useRef, useState } from 'react';
import Button from './components/Button';
import CameraViewport from './components/CameraViewport';
import ColumnLayout from './components/ColumnLayout';
import HorizontalSeparator from './components/HorizontalSeparator';
import Panel from './components/Panel';
import StaticText from './components/StaticText';
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
    const [streamUrl, setStreamUrl] = useState(null);
    const [reticleSettings, setReticleSettings] = useState(() => ({
        ...DEFAULT_RETICLE_SETTINGS
    }));
    const [isViewerLoading, setIsViewerLoading] = useState(false);
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

    const handleStartAIAgent = useCallback(async () => {
        try {
            await activateAIAgent();
        } catch (error) {
            console.error('Failed to activate AI agent:', error);
        }
    }, [activateAIAgent]);

    if (!activationLoaded) {
        return <CameraViewport isLoading={true} />;
    }

    if (!aiagentFullyActivated) {
        return (
            <div className="page-container">
                <Panel title="AI Agent">
                    <ColumnLayout gap="1rem">
                        <StaticText
                            text="Pressing the button below activates the AI Agent that will control all motors. This is not a simulation like on the Manual Control page. It starts the real AI Agent, which immediately begins executing the mission according to the configured AI setup."
                        />
                        <HorizontalSeparator label="Runtime Activation" fullWidth={true} bleed="1.5rem" />
                        <div className="responsive-input-container" style={{ width: '100%' }}>
                            <span style={{ whiteSpace: 'nowrap' }}>Activate AI Agent:</span>
                            <Button
                                label={isActivationBusy ? 'Starting...' : 'Start AI Agent'}
                                onClick={handleStartAIAgent}
                                disabled={isActivationBusy}
                                color="#dc2626"
                                style={{ width: 'auto' }}
                            />
                        </div>
                    </ColumnLayout>
                </Panel>
            </div>
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
