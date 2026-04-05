import React, { useState, useEffect } from 'react';
import ModalWindow from './components/ModalWindow';
import HorizontalSeparator from './components/HorizontalSeparator';
import NumericInput from './components/NumericInput';
import DateTimePicker from './components/DateTimePicker';
import Switch from './components/Switch';
import Slider from './components/Slider';
import StaticText from './components/StaticText';

/**
 * Exit Strategy Edit Modal.
 * Allows editing exit strategy conditions: max engagements, timeout after first engagement, 
 * predefined datetime, exit strategy duration, and motor settings.
 */
const AISetupExitStrategyEditModal = ({ isOpen, onClose, onSave, initialSettings, motors }) => {
    const [tempSettings, setTempSettings] = useState({
        maxEngagements: 1000000,
        timeoutAfterFirstEngagement: 1000000,
        predefinedDateTime: new Date('2199-12-31T23:59:59'),
        exitStrategyDuration: 0.5,
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
                maxEngagements: initialSettings.maxEngagements ?? 1000000,
                timeoutAfterFirstEngagement: initialSettings.timeoutAfterFirstEngagement ?? 1000000,
                predefinedDateTime: initialSettings.predefinedDateTime 
                    ? new Date(initialSettings.predefinedDateTime) 
                    : new Date('2199-12-31T23:59:59'),
                exitStrategyDuration: initialSettings.exitStrategyDuration ?? 0.5,
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
            title="Exit Strategy"
            onOk={handleSave}
            onCancel={handleCancel}
            okLabel="Save"
            cancelLabel="Cancel"
            movable={true}
        >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {/* When to Execute Exit Strategy */}
                <HorizontalSeparator label="When to Execute the Exit Strategy (logical OR)" fullWidth={true} bleed="1rem" />
                <NumericInput
                    label="After a Number of AI Engagements"
                    value={tempSettings.maxEngagements}
                    onChange={(val) => updateSetting('maxEngagements', val)}
                    min={1}
                    max={1000000000}
                    step={1}
                    decimalPlaces={0}
                    labelWidth="230px"
                    labelPosition="left"
                />
                <NumericInput
                    label="Timeout after the 1st AI Engagement (s)"
                    value={tempSettings.timeoutAfterFirstEngagement}
                    onChange={(val) => updateSetting('timeoutAfterFirstEngagement', val)}
                    min={1}
                    max={1000000000}
                    step={1}
                    decimalPlaces={0}
                    labelWidth="230px"
                    labelPosition="left"
                />
                <DateTimePicker
                    label="Fixed Date and Time (UTC)"
                    value={tempSettings.predefinedDateTime}
                    onChange={(val) => updateSetting('predefinedDateTime', val)}
                    minValue={new Date()}
                    maxValue={new Date('2199-12-31T23:59:59')}
                    precision="seconds"
                    labelWidth="230px"
                />
                <HorizontalSeparator label="Rules" fullWidth={true} />
                <NumericInput
                    label="Exit Strategy Duration (s)"
                    value={tempSettings.exitStrategyDuration}
                    onChange={(val) => updateSetting('exitStrategyDuration', val)}
                    min={0.01}
                    max={1000000000}
                    step={0.01}
                    decimalPlaces={2}
                    labelWidth="230px"
                    labelPosition="left"
                />

                {/* On Execute Event */}
                <HorizontalSeparator label="On Execute Event" fullWidth={true} bleed="1rem" weight="bold" />

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
                        No motors configured for exit strategy
                    </div>
                )}
            </div>
        </ModalWindow>
    );
};

export default AISetupExitStrategyEditModal;
