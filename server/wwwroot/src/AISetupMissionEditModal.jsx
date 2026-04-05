import React, { useState, useEffect } from 'react';
import ModalWindow from './components/ModalWindow';
import HorizontalSeparator from './components/HorizontalSeparator';
import Slider from './components/Slider';
import Switch from './components/Switch';
import ComboBox from './components/ComboBox';
import StaticText from './components/StaticText';
import NumericInput from './components/NumericInput';

/**
 * Mission type options.
 */
const MISSION_TYPE_OPTIONS = [
    { label: 'Random Walk', value: 'Random Walk' }
];

/**
 * Mission Edit Modal.
 * Allows editing mission type, duty cycle, delay, trigger hold duration, and motor settings.
 */
const AISetupMissionEditModal = ({ isOpen, onClose, onSave, initialSettings, motors }) => {
    const [tempSettings, setTempSettings] = useState({
        missionType: 'Random Walk',
        disableDutyCycle: true,
        delayBetweenEngagements: 0.5,
        triggerHoldDuration: 0.5,
        motors: []
    });

    // Initialize temp settings when modal opens
    useEffect(() => {
        if (isOpen && initialSettings) {
            // Build motors array with names and colors from motors prop
            const motorSettings = (initialSettings.motors || []).map(m => {
                const motorInfo = motors?.find(mi => mi.index === m.index) || {};
                return {
                    index: m.index,
                    enabled: m.enabled ?? false,
                    speed: m.speed ?? 0.0,
                    name: motorInfo.name || `Motor ${m.index}`,
                    color: motorInfo.color || '#888888'
                };
            });

            setTempSettings({
                missionType: initialSettings.missionType ?? 'Random Walk',
                disableDutyCycle: initialSettings.disableDutyCycle ?? true,
                delayBetweenEngagements: initialSettings.delayBetweenEngagements ?? 0.5,
                triggerHoldDuration: initialSettings.triggerHoldDuration ?? 0.5,
                motors: motorSettings
            });
        }
    }, [isOpen, initialSettings, motors]);

    /**
     * Update a setting.
     */
    const updateSetting = (key, value) => {
        setTempSettings(prev => ({ ...prev, [key]: value }));
    };

    /**
     * Update motor setting.
     */
    const updateMotorSetting = (motorIndex, key, value) => {
        setTempSettings(prev => ({
            ...prev,
            motors: prev.motors.map(m => 
                m.index === motorIndex ? { ...m, [key]: value } : m
            )
        }));
    };

    /**
     * Handle save.
     */
    const handleSave = () => {
        if (onSave) {
            // Strip out name and color from motors before saving
            const settingsToSave = {
                ...tempSettings,
                motors: tempSettings.motors.map(m => ({
                    index: m.index,
                    enabled: m.enabled,
                    speed: m.speed
                }))
            };
            onSave(settingsToSave);
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
            title="Mission"
            onOk={handleSave}
            onCancel={handleCancel}
            okLabel="Save"
            cancelLabel="Cancel"
            movable={true}
        >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {/* Rules */}
                <HorizontalSeparator label="Rules" fullWidth={true} bleed="1rem" />
                <ComboBox
                    label="Mission Type"
                    items={MISSION_TYPE_OPTIONS}
                    value={tempSettings.missionType}
                    onChange={(val) => updateSetting('missionType', val)}
                    labelWidth="220px"
                    disabled={true}
                />
                <Switch
                    label="Disable Duty Cycle After 1st AI Engagement"
                    value={tempSettings.disableDutyCycle}
                    onChange={(val) => updateSetting('disableDutyCycle', val)}
                    labelWidth="220px"
                />
                <NumericInput
                    label="Delay Between Engagements (s)"
                    value={tempSettings.delayBetweenEngagements}
                    onChange={(val) => updateSetting('delayBetweenEngagements', val)}
                    min={0.01}
                    max={1000000000}
                    step={0.01}
                    decimalPlaces={2}
                    labelWidth="220px"
                    labelPosition="left"
                />
                <NumericInput
                    label="Engagement Duration (s)"
                    value={tempSettings.triggerHoldDuration}
                    onChange={(val) => updateSetting('triggerHoldDuration', val)}
                    min={0.01}
                    max={1000000000}
                    step={0.01}
                    decimalPlaces={2}
                    labelWidth="220px"
                    labelPosition="left"
                />

                {/* On Engagement Event */}
                <HorizontalSeparator label="On Engagement Event" fullWidth={true} bleed="1rem" weight="bold" />

                {/* Motor sections */}
                {tempSettings.motors.map((motor) => (
                    <div key={motor.index}>
                        <HorizontalSeparator label="Set Motor Position" fullWidth={true} bleed="1rem" />
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                            {/* Motor Name with color */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <div style={{
                                    width: '16px',
                                    height: '16px',
                                    borderRadius: '50%',
                                    backgroundColor: motor.color || '#888888',
                                    flexShrink: 0
                                }} />
                                <StaticText text={<><span style={{ fontWeight: 500 }}>Name:</span> {motor.name}</>} />
                            </div>
                            {/* Enabled switch */}
                            <Switch
                                label="Enabled"
                                value={motor.enabled}
                                onChange={(val) => updateMotorSetting(motor.index, 'enabled', val)}
                                labelWidth="80px"
                            />
                            {/* Speed slider */}
                            <Slider
                                label="Speed"
                                value={motor.speed}
                                onChange={(val) => updateMotorSetting(motor.index, 'speed', val)}
                                min={-1}
                                max={1}
                                step={0.01}
                                decimalPlaces={3}
                                allowManualInput={true}
                                labelWidth="80px"
                                disabled={!motor.enabled}
                            />
                        </div>
                    </div>
                ))}

                {tempSettings.motors.length === 0 && (
                    <div style={{ color: '#6b7280', fontStyle: 'italic', marginTop: '0.5rem' }}>
                        No motors configured for this mission
                    </div>
                )}
            </div>
        </ModalWindow>
    );
};

export default AISetupMissionEditModal;
