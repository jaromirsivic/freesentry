import React, { useState, useEffect, useRef } from 'react';
import ModalWindow from './components/ModalWindow';
import HorizontalSeparator from './components/HorizontalSeparator';
import Slider from './components/Slider';
import NumericInput from './components/NumericInput';
import Switch from './components/Switch';
import DateTimePicker from './components/DateTimePicker';
import ComboBox from './components/ComboBox';

/**
 * Organ configuration component with enable switch and settings.
 */
const OrganSection = ({ 
    label, 
    enabled, 
    sizeMultiplier, 
    confidenceThreshold, 
    minimumRadius, 
    onEnabledChange, 
    onSizeMultiplierChange, 
    onConfidenceThresholdChange, 
    onMinimumRadiusChange 
}) => (
    <>
        <HorizontalSeparator label={label} fullWidth={true} bleed="1rem" />
        <Switch
            label="Detection Enabled"
            value={enabled}
            onChange={onEnabledChange}
            labelWidth="180px"
        />
        <Slider
            label="Size Multiplier"
            value={sizeMultiplier}
            onChange={onSizeMultiplierChange}
            min={0.2}
            max={1.0}
            step={0.1}
            decimalPlaces={1}
            allowManualInput={true}
            labelWidth="180px"
            disabled={!enabled}
        />
        <Slider
            label="Confidence Threshold"
            value={confidenceThreshold}
            onChange={onConfidenceThresholdChange}
            min={0.01}
            max={1.0}
            step={0.01}
            decimalPlaces={2}
            allowManualInput={true}
            labelWidth="180px"
            disabled={!enabled}
        />
        <Slider
            label="Organ Minimum Radius"
            value={minimumRadius}
            onChange={onMinimumRadiusChange}
            min={1}
            max={10000}
            minSlider={1}
            maxSlider={200}
            step={1}
            decimalPlaces={0}
            allowManualInput={true}
            labelWidth="180px"
            disabled={!enabled}
        />
    </>
);

/**
 * AI Essentials Edit Modal.
 * Allows editing activation datetime, engagement essentials, and organ settings.
 * deviceOptions and defaultDevice come from the backend via initialSettings.
 */
const AISetupEssentialsEditModal = ({ isOpen, onClose, onSave, initialSettings }) => {
    const [tempSettings, setTempSettings] = useState({
        activationDateTime: new Date('2199-12-31T23:59:59'),
        modelName: 'Fast and Simple',
        device: '',
        minFpsToAllowEngagement: 0,
        organMustExistForSeconds: 0,
        detectionRadiusFromReticle: 50,
        organs: {
            brain: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
            chest: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
            heart: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
            liver: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 },
            abdomen: { enabled: true, sizeMultiplier: 1.0, confidenceThreshold: 0.5, minimumRadius: 1 }
        }
    });

    const prevIsOpenRef = useRef(false);

    // Initialize temp settings only when modal transitions from closed to open
    useEffect(() => {
        if (isOpen && !prevIsOpenRef.current && initialSettings) {
            setTempSettings({
                activationDateTime: initialSettings.activationDateTime 
                    ? new Date(initialSettings.activationDateTime) 
                    : new Date('2199-12-31T23:59:59'),
                modelName: initialSettings.modelName ?? 'Fast and Simple',
                device: initialSettings.device ?? initialSettings.defaultDevice ?? '',
                minFpsToAllowEngagement: initialSettings.minFpsToAllowEngagement ?? 0,
                organMustExistForSeconds: initialSettings.organMustExistForSeconds ?? 0,
                detectionRadiusFromReticle: initialSettings.detectionRadiusFromReticle ?? 50,
                organs: {
                    brain: { ...initialSettings.organs?.brain },
                    chest: { ...initialSettings.organs?.chest },
                    heart: { ...initialSettings.organs?.heart },
                    liver: { ...initialSettings.organs?.liver },
                    abdomen: { ...initialSettings.organs?.abdomen }
                }
            });
        }
        prevIsOpenRef.current = isOpen;
    }, [isOpen, initialSettings]);

    /**
     * Update a root-level setting.
     */
    const updateSetting = (key, value) => {
        setTempSettings(prev => ({ ...prev, [key]: value }));
    };

    /**
     * Update an organ setting.
     */
    const updateOrgan = (organKey, field, value) => {
        setTempSettings(prev => ({
            ...prev,
            organs: {
                ...prev.organs,
                [organKey]: {
                    ...prev.organs[organKey],
                    [field]: value
                }
            }
        }));
    };

    /**
     * Handle save.
     */
    const handleSave = () => {
        if (onSave) {
            onSave(tempSettings);
        }
        onClose();
    };

    /**
     * Handle cancel.
     */
    const handleCancel = () => {
        onClose();
    };

    return (
        <ModalWindow
            isOpen={isOpen}
            title="AI Essentials"
            onOk={handleSave}
            onCancel={handleCancel}
            okLabel="Save"
            cancelLabel="Cancel"
            movable={true}
        >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {/* Activation */}
                <HorizontalSeparator label="Activation" fullWidth={true} bleed="1rem" />
                <DateTimePicker
                    label="Date and Time"
                    value={tempSettings.activationDateTime}
                    onChange={(val) => updateSetting('activationDateTime', val)}
                    minValue={new Date()}
                    maxValue={new Date('2199-12-31T23:59:59')}
                    precision="seconds"
                    labelWidth="150px"
                />

                {/* Engagement Essentials */}
                <HorizontalSeparator label="Engagement Essentials" fullWidth={true} bleed="1rem" />
                <ComboBox
                    label="AI Model"
                    items={(initialSettings?.modelNames ?? []).map((name) => ({ label: name, value: name }))}
                    value={tempSettings.modelName}
                    onChange={(val) => updateSetting('modelName', val)}
                    labelWidth="150px"
                />
                <ComboBox
                    label="Device"
                    items={initialSettings?.deviceOptions ?? []}
                    value={tempSettings.device}
                    onChange={(val) => updateSetting('device', val)}
                    labelWidth="150px"
                />
                <NumericInput
                    label="Minimum FPS to Allow Engagement"
                    labelPosition="left"
                    value={tempSettings.minFpsToAllowEngagement}
                    onChange={(val) => updateSetting('minFpsToAllowEngagement', val)}
                    min={0}
                    max={50}
                    step={1}
                    decimalPlaces={0}
                    labelWidth="250px"
                />
                <NumericInput
                    label="Organ must be detected for at least (sec.)"
                    labelPosition="left"
                    value={tempSettings.organMustExistForSeconds}
                    onChange={(val) => updateSetting('organMustExistForSeconds', val)}
                    min={0}
                    max={2}
                    step={0.05}
                    decimalPlaces={2}
                    labelWidth="250px"
                />
                <NumericInput
                    label="Detection Radius - From Reticle (px)"
                    labelPosition="left"
                    value={tempSettings.detectionRadiusFromReticle}
                    onChange={(val) => updateSetting('detectionRadiusFromReticle', val)}
                    min={0}
                    max={10000}
                    step={10}
                    decimalPlaces={0}
                    labelWidth="250px"
                />

                {/* Organs / Body Parts */}
                <HorizontalSeparator label="Organs / Body Parts" fullWidth={true} bleed="1rem" weight="bold" />

                {/* Brain */}
                <OrganSection
                    label="Brain"
                    enabled={tempSettings.organs.brain.enabled}
                    sizeMultiplier={tempSettings.organs.brain.sizeMultiplier}
                    confidenceThreshold={tempSettings.organs.brain.confidenceThreshold}
                    minimumRadius={tempSettings.organs.brain.minimumRadius}
                    onEnabledChange={(val) => updateOrgan('brain', 'enabled', val)}
                    onSizeMultiplierChange={(val) => updateOrgan('brain', 'sizeMultiplier', val)}
                    onConfidenceThresholdChange={(val) => updateOrgan('brain', 'confidenceThreshold', val)}
                    onMinimumRadiusChange={(val) => updateOrgan('brain', 'minimumRadius', val)}
                />

                {/* Chest */}
                <OrganSection
                    label="Chest"
                    enabled={tempSettings.organs.chest.enabled}
                    sizeMultiplier={tempSettings.organs.chest.sizeMultiplier}
                    confidenceThreshold={tempSettings.organs.chest.confidenceThreshold}
                    minimumRadius={tempSettings.organs.chest.minimumRadius}
                    onEnabledChange={(val) => updateOrgan('chest', 'enabled', val)}
                    onSizeMultiplierChange={(val) => updateOrgan('chest', 'sizeMultiplier', val)}
                    onConfidenceThresholdChange={(val) => updateOrgan('chest', 'confidenceThreshold', val)}
                    onMinimumRadiusChange={(val) => updateOrgan('chest', 'minimumRadius', val)}
                />

                {/* Heart */}
                <OrganSection
                    label="Heart"
                    enabled={tempSettings.organs.heart.enabled}
                    sizeMultiplier={tempSettings.organs.heart.sizeMultiplier}
                    confidenceThreshold={tempSettings.organs.heart.confidenceThreshold}
                    minimumRadius={tempSettings.organs.heart.minimumRadius}
                    onEnabledChange={(val) => updateOrgan('heart', 'enabled', val)}
                    onSizeMultiplierChange={(val) => updateOrgan('heart', 'sizeMultiplier', val)}
                    onConfidenceThresholdChange={(val) => updateOrgan('heart', 'confidenceThreshold', val)}
                    onMinimumRadiusChange={(val) => updateOrgan('heart', 'minimumRadius', val)}
                />

                {/* Liver */}
                <OrganSection
                    label="Liver"
                    enabled={tempSettings.organs.liver.enabled}
                    sizeMultiplier={tempSettings.organs.liver.sizeMultiplier}
                    confidenceThreshold={tempSettings.organs.liver.confidenceThreshold}
                    minimumRadius={tempSettings.organs.liver.minimumRadius}
                    onEnabledChange={(val) => updateOrgan('liver', 'enabled', val)}
                    onSizeMultiplierChange={(val) => updateOrgan('liver', 'sizeMultiplier', val)}
                    onConfidenceThresholdChange={(val) => updateOrgan('liver', 'confidenceThreshold', val)}
                    onMinimumRadiusChange={(val) => updateOrgan('liver', 'minimumRadius', val)}
                />

                {/* Abdomen */}
                <OrganSection
                    label="Abdomen"
                    enabled={tempSettings.organs.abdomen.enabled}
                    sizeMultiplier={tempSettings.organs.abdomen.sizeMultiplier}
                    confidenceThreshold={tempSettings.organs.abdomen.confidenceThreshold}
                    minimumRadius={tempSettings.organs.abdomen.minimumRadius}
                    onEnabledChange={(val) => updateOrgan('abdomen', 'enabled', val)}
                    onSizeMultiplierChange={(val) => updateOrgan('abdomen', 'sizeMultiplier', val)}
                    onConfidenceThresholdChange={(val) => updateOrgan('abdomen', 'confidenceThreshold', val)}
                    onMinimumRadiusChange={(val) => updateOrgan('abdomen', 'minimumRadius', val)}
                />
            </div>
        </ModalWindow>
    );
};

export default AISetupEssentialsEditModal;
