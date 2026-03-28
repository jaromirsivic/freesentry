import React, { useState, useEffect } from 'react';
import Panel from './components/Panel';
import Button from './components/Button';
import HorizontalSeparator from './components/HorizontalSeparator';
import StaticText from './components/StaticText';
import ColumnLayout from './components/ColumnLayout';
import AISetupEssentialsEditModal from './AISetupEssentialsEditModal';
import AISetupMissionEditModal from './AISetupMissionEditModal';
import AISetupExitStrategyEditModal from './AISetupExitStrategyEditModal';
import editIcon from './assets/icons/edit.svg';
import { getAISetupSettings, saveAISetupSettings } from './lib/api';

const boldTextStyle = { fontWeight: 'bold' };

/**
 * Helper to render static text field.
 */
const RenderStaticField = ({ label, value }) => (
    <StaticText text={<>{label}: <span style={boldTextStyle}>{value}</span></>} />
);

/**
 * Default settings for AI Setup.
 */
/** Normalize device from settings (e.g. "cuda:0") to a value that matches one of deviceOptions. */
const normalizeDevice = (d, deviceOptions, defaultDevice) => {
    if (!d) return defaultDevice ?? '';
    const s = String(d).trim();
    if (!s) return defaultDevice ?? '';
    const opts = deviceOptions ?? [];
    const exact = opts.find((o) => o.value === s);
    if (exact) return exact.value;
    const byPrefix = opts.find((o) => o.value.startsWith(s) || s.startsWith((o.value.split(' ')[0] ?? '')));
    return byPrefix ? byPrefix.value : s;
};

const defaultSettings = {
    // AI Essentials
    activationDateTime: '2199-12-31T23:59:59Z',
    modelName: 'yolo26n-pose',
    device: '',
    minFpsToAllowEngagement: 0,
    organMustBeVisibleSeconds: 0,
    detectionRadiusFromReticle: 50,
    organs: {
        brain: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
        chest: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
        heart: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
        liver: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
        abdomen: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 }
    },
    // Mission
    missions: {
        randomWalk: {
            enabled: true,
            disableDutyCycle: true,
            delayBetweenEngagements: 0.5,
            engagementDuration: 0.5,
            motors: []
        }
    },
    // Exit Strategy
    exitStrategy: {
        maxEngagements: 1000000000,
        timeoutAfterFirstEngagement: 10000000000,
        fixedDateTime: '2199-12-31T23:59:59Z',
        exitStrategyDuration: 0.5,
        motors: []
    }
};

/**
 * AI Setup page with three panels:
 * - AI Essentials
 * - Mission
 * - Exit Strategy
 */
const AISetup = () => {
    const [settings, setSettings] = useState(defaultSettings);
    const [motors, setMotors] = useState([]);
    const [modelNames, setModelNames] = useState([]);
    const [deviceOptions, setDeviceOptions] = useState([]);
    const [defaultDevice, setDefaultDevice] = useState('');
    const [isEssentialsModalOpen, setIsEssentialsModalOpen] = useState(false);
    const [isMissionModalOpen, setIsMissionModalOpen] = useState(false);
    const [isExitStrategyModalOpen, setIsExitStrategyModalOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // Load settings from backend on mount
    useEffect(() => {
        const loadSettings = async () => {
            try {
                const response = await getAISetupSettings();
                if (response.success) {
                    setSettings(response.aiSetup);
                    setMotors(response.motors || []);
                    setModelNames(response.modelNames || []);
                    setDeviceOptions(response.deviceOptions || []);
                    setDefaultDevice(response.defaultDevice ?? '');
                }
            } catch (error) {
                console.error('Failed to load AI Setup settings:', error);
            } finally {
                setIsLoading(false);
            }
        };
        loadSettings();
    }, []);

    /**
     * Save settings to backend.
     */
    const saveSettingsToBackend = async (newSettings) => {
        try {
            await saveAISetupSettings({ aiSetup: newSettings });
        } catch (error) {
            console.error('Failed to save AI Setup settings:', error);
        }
    };

    /**
     * Format date for display.
     */
    const formatDateTime = (date) => {
        if (!date) return 'N/A';
        const d = new Date(date);
        return d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    };

    /**
     * Handle essentials modal save.
     */
    const handleEssentialsSave = async (newSettings) => {
        const updatedSettings = {
            ...settings,
            activationDateTime: newSettings.activationDateTime instanceof Date 
                ? newSettings.activationDateTime.toISOString() 
                : newSettings.activationDateTime,
            modelName: newSettings.modelName ?? settings.modelName,
            device: newSettings.device ?? settings.device ?? defaultDevice,
            minFpsToAllowEngagement: newSettings.minFpsToAllowEngagement,
            organMustBeVisibleSeconds: newSettings.organMustExistForSeconds,
            detectionRadiusFromReticle: newSettings.detectionRadiusFromReticle,
            organs: newSettings.organs
        };
        setSettings(updatedSettings);
        await saveSettingsToBackend(updatedSettings);
    };

    /**
     * Handle mission modal save.
     */
    const handleMissionSave = async (newSettings) => {
        const updatedSettings = {
            ...settings,
            missions: {
                ...settings.missions,
                randomWalk: {
                    ...settings.missions?.randomWalk,
                    enabled: true,
                    disableDutyCycle: newSettings.disableDutyCycle,
                    delayBetweenEngagements: newSettings.delayBetweenEngagements,
                    engagementDuration: newSettings.triggerHoldDuration,
                    motors: newSettings.motors
                }
            }
        };
        setSettings(updatedSettings);
        await saveSettingsToBackend(updatedSettings);
    };

    /**
     * Handle exit strategy modal save.
     */
    const handleExitStrategySave = async (newSettings) => {
        const updatedSettings = {
            ...settings,
            exitStrategy: {
                ...settings.exitStrategy,
                maxEngagements: newSettings.maxEngagements,
                timeoutAfterFirstEngagement: newSettings.timeoutAfterFirstEngagement,
                fixedDateTime: newSettings.predefinedDateTime instanceof Date 
                    ? newSettings.predefinedDateTime.toISOString() 
                    : newSettings.predefinedDateTime,
                exitStrategyDuration: newSettings.exitStrategyDuration,
                motors: newSettings.motors
            }
        };
        setSettings(updatedSettings);
        await saveSettingsToBackend(updatedSettings);
    };

    /**
     * Get mission type display name.
     */
    const getMissionType = () => {
        return 'Random Walk';
    };

    /**
     * Get mission settings from randomWalk.
     */
    const getMissionSettings = () => {
        const randomWalk = settings.missions?.randomWalk || {};
        return {
            missionType: 'Random Walk',
            disableDutyCycle: randomWalk.disableDutyCycle ?? true,
            delayBetweenEngagements: randomWalk.delayBetweenEngagements ?? 0.5,
            triggerHoldDuration: randomWalk.engagementDuration ?? 0.5,
            motors: randomWalk.motors || []
        };
    };

    /**
     * Get exit strategy settings.
     */
    const getExitStrategySettings = () => {
        const exitStrategy = settings.exitStrategy || {};
        return {
            maxEngagements: exitStrategy.maxEngagements ?? 1000000000,
            timeoutAfterFirstEngagement: exitStrategy.timeoutAfterFirstEngagement ?? 10000000000,
            predefinedDateTime: exitStrategy.fixedDateTime,
            exitStrategyDuration: exitStrategy.exitStrategyDuration ?? 0.5,
            motors: exitStrategy.motors || []
        };
    };

    if (isLoading) {
        return (
            <div className="page-container">
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
                    Loading...
                </div>
            </div>
        );
    }

    return (
        <div className="page-container">
            <div style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '1rem', alignItems: 'stretch' }}>
                {/* AI Essentials Panel */}
                <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex' }}>
                    <Panel
                        style={{ flex: 1 }}
                        title="AI Essentials"
                        headerAction={
                            <Button
                                label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                onClick={() => setIsEssentialsModalOpen(true)}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            />
                        }
                    >
                        <ColumnLayout gap="0.75rem">
                            <HorizontalSeparator label="Activation" fullWidth={true} />
                            <RenderStaticField 
                                label="Date and Time" 
                                value={formatDateTime(settings.activationDateTime)} 
                            />
                            <HorizontalSeparator label="Engagement Essentials" fullWidth={true} />
                            <RenderStaticField 
                                label="AI Model" 
                                value={settings.modelName ?? '—'} 
                            />
                            <RenderStaticField 
                                label="Device" 
                                value={normalizeDevice(settings.device, deviceOptions, defaultDevice) || defaultDevice || '—'} 
                            />
                            <RenderStaticField 
                                label="Minimum FPS to Allow AI Engagement" 
                                value={settings.minFpsToAllowEngagement} 
                            />
                            <RenderStaticField 
                                label="Organ must be near the reticle for at least" 
                                value={`${settings.organMustBeVisibleSeconds} s`} 
                            />
                            <RenderStaticField 
                                label="Organ Distance From the Reticle" 
                                value={`${settings.detectionRadiusFromReticle} px`} 
                            />
                        </ColumnLayout>
                    </Panel>
                </div>

                {/* Mission Panel */}
                <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex' }}>
                    <Panel
                        style={{ flex: 1 }}
                        title="Mission"
                        headerAction={
                            <Button
                                label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                onClick={() => setIsMissionModalOpen(true)}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            />
                        }
                    >
                        <ColumnLayout gap="0.75rem">
                            <HorizontalSeparator label="Rules" fullWidth={true} />
                            <RenderStaticField 
                                label="Mission Type" 
                                value={getMissionType()} 
                            />
                            <RenderStaticField 
                                label="Disable Duty Cycle After 1st AI Engagement" 
                                value={getMissionSettings().disableDutyCycle ? 'Yes' : 'No'} 
                            />
                            <RenderStaticField 
                                label="Delay Between Engagements" 
                                value={`${getMissionSettings().delayBetweenEngagements} s`} 
                            />
                            <RenderStaticField 
                                label="Engagement Duration (s)" 
                                value={`${getMissionSettings().triggerHoldDuration} s`} 
                            />
                        </ColumnLayout>
                    </Panel>
                </div>

                {/* Exit Strategy Panel */}
                <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex' }}>
                    <Panel
                        style={{ flex: 1 }}
                        title="Exit Strategy"
                        headerAction={
                            <Button
                                label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                onClick={() => setIsExitStrategyModalOpen(true)}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            />
                        }
                    >
                        <ColumnLayout gap="0.75rem">
                            <HorizontalSeparator label="When to Execute? (logical OR)" fullWidth={true} />
                            <RenderStaticField 
                                label="After a Number of AI Engagements" 
                                value={getExitStrategySettings().maxEngagements} 
                            />
                            <RenderStaticField 
                                label="Timeout after the 1st AI Engagement" 
                                value={`${getExitStrategySettings().timeoutAfterFirstEngagement} s`} 
                            />
                            <RenderStaticField 
                                label="Fixed Date and Time" 
                                value={formatDateTime(getExitStrategySettings().predefinedDateTime)} 
                            />
                            <HorizontalSeparator label="Rules" fullWidth={true} />
                            <RenderStaticField 
                                label="Exit Strategy Duration (s)" 
                                value={`${getExitStrategySettings().exitStrategyDuration} s`} 
                            />
                        </ColumnLayout>
                    </Panel>
                </div>
            </div>

            {/* Modals */}
            <AISetupEssentialsEditModal
                isOpen={isEssentialsModalOpen}
                onClose={() => setIsEssentialsModalOpen(false)}
                onSave={handleEssentialsSave}
                initialSettings={{
                    activationDateTime: settings.activationDateTime,
                    modelName: settings.modelName,
                    device: normalizeDevice(settings.device, deviceOptions, defaultDevice),
                    modelNames,
                    deviceOptions,
                    defaultDevice,
                    minFpsToAllowEngagement: settings.minFpsToAllowEngagement,
                    organMustExistForSeconds: settings.organMustBeVisibleSeconds,
                    detectionRadiusFromReticle: settings.detectionRadiusFromReticle,
                    organs: settings.organs
                }}
            />

            <AISetupMissionEditModal
                isOpen={isMissionModalOpen}
                onClose={() => setIsMissionModalOpen(false)}
                onSave={handleMissionSave}
                initialSettings={getMissionSettings()}
                motors={motors}
            />

            <AISetupExitStrategyEditModal
                isOpen={isExitStrategyModalOpen}
                onClose={() => setIsExitStrategyModalOpen(false)}
                onSave={handleExitStrategySave}
                initialSettings={getExitStrategySettings()}
                motors={motors}
            />
        </div>
    );
};

export default AISetup;
