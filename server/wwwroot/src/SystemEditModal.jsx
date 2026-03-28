import React, { useState, useEffect } from 'react';
import ModalWindow from './components/ModalWindow';
import HorizontalSeparator from './components/HorizontalSeparator';
import DateTimePicker from './components/DateTimePicker';
import ComboBox from './components/ComboBox';

/**
 * System Edit Modal.
 * Allows editing server date/time and timezone.
 */
const SystemEditModal = ({ isOpen, onClose, onSave, initialSettings, availableTimezones }) => {
    const [tempSettings, setTempSettings] = useState({
        dateTime: new Date(),
        timezone: 'UTC'
    });
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState('');

    // Initialize temp settings when modal opens
    useEffect(() => {
        if (isOpen && initialSettings) {
            // Parse the datetime string (format: "YYYY-MM-DDTHH:MM:SS")
            const dt = initialSettings.dateTime 
                ? new Date(initialSettings.dateTime) 
                : new Date();
            
            setTempSettings({
                dateTime: dt,
                timezone: initialSettings.timezone || 'UTC'
            });
            setError('');
        }
    }, [isOpen, initialSettings]);

    /**
     * Update a setting.
     */
    const updateSetting = (key, value) => {
        setTempSettings(prev => ({ ...prev, [key]: value }));
    };

    /**
     * Handle save.
     */
    const handleSave = async () => {
        setIsSaving(true);
        setError('');
        
        try {
            const dt = tempSettings.dateTime;
            const settingsToSave = {
                year: dt.getFullYear(),
                month: dt.getMonth() + 1,
                day: dt.getDate(),
                hour: dt.getHours(),
                minute: dt.getMinutes(),
                second: dt.getSeconds(),
                timezone: tempSettings.timezone
            };
            
            await onSave(settingsToSave);
            onClose();
        } catch (err) {
            setError(err.message || 'Failed to save settings');
        } finally {
            setIsSaving(false);
        }
    };

    /**
     * Handle cancel.
     */
    const handleCancel = () => {
        setError('');
        onClose();
    };

    // Convert available timezones to combobox items
    const timezoneItems = (availableTimezones || []).map(tz => ({
        label: tz,
        value: tz
    }));

    return (
        <ModalWindow
            isOpen={isOpen}
            title="System Date and Time"
            onOk={handleSave}
            onCancel={handleCancel}
            okLabel={isSaving ? "Saving..." : "Save"}
            cancelLabel="Cancel"
            okDisabled={isSaving}
            movable={true}
        >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {/* Date and Time */}
                <HorizontalSeparator label="Date and Time" fullWidth={true} bleed="1rem" />
                <DateTimePicker
                    label="Date and Time"
                    value={tempSettings.dateTime}
                    onChange={(val) => updateSetting('dateTime', val)}
                    minValue={new Date('1970-01-01T00:00:00')}
                    maxValue={new Date('2199-12-31T23:59:59')}
                    precision="seconds"
                    labelWidth="120px"
                />
                <ComboBox
                    label="Time Zone"
                    items={timezoneItems}
                    value={tempSettings.timezone}
                    onChange={(val) => updateSetting('timezone', val)}
                    labelWidth="120px"
                />

                {error && (
                    <div style={{ 
                        color: '#ef4444', 
                        padding: '0.5rem', 
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderRadius: '4px',
                        fontSize: '0.875rem'
                    }}>
                        {error}
                    </div>
                )}

                <div style={{ 
                    color: '#6b7280', 
                    fontSize: '0.75rem', 
                    fontStyle: 'italic',
                    marginTop: '0.5rem'
                }}>
                    Note: Changing system date/time requires administrator privileges.
                </div>
            </div>
        </ModalWindow>
    );
};

export default SystemEditModal;
