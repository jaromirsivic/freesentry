import React, { useState, useEffect } from 'react';
import ModalWindow from './components/ModalWindow';
import HorizontalSeparator from './components/HorizontalSeparator';
import ComboBox from './components/ComboBox';
import Textbox from './components/Textbox';
import ColumnLayout from './components/ColumnLayout';
import Button from './components/Button';

const MODE_OPTIONS = [
    { label: 'Disabled', value: 'disabled' },
    { label: 'Access Point', value: 'accessPoint' },
    { label: 'Client', value: 'client' }
];

const AP_SECURITY_OPTIONS = [
    { label: 'wpa-psk', value: 'wpa-psk' }
];

const CLIENT_SECURITY_OPTIONS = [
    { label: 'none', value: 'none' },
    { label: 'wep', value: 'wep' },
    { label: 'wpa-psk', value: 'wpa-psk' },
    { label: 'wpa3', value: 'wpa3' }
];

const WifiEditModal = ({ isOpen, onClose, onSave, wifiSettings }) => {
    const [tempSettings, setTempSettings] = useState(null);
    const [isSaving, setIsSaving] = useState(false);
    const [showRebootInfo, setShowRebootInfo] = useState(false);

    useEffect(() => {
        if (isOpen && wifiSettings) {
            setTempSettings(JSON.parse(JSON.stringify(wifiSettings)));
            setShowRebootInfo(false);
        }
    }, [isOpen, wifiSettings]);

    const updateField = (section, field, value) => {
        setTempSettings(prev => ({
            ...prev,
            [section]: {
                ...prev[section],
                [field]: value
            }
        }));
    };

    const updateMode = (value) => {
        setTempSettings(prev => ({ ...prev, mode: value }));
    };

    const getValidationErrors = () => {
        if (!tempSettings) return [];
        const errors = [];
        if (tempSettings.mode === 'accessPoint' && (tempSettings.accessPoint?.password || '').length < 8) {
            errors.push('Password must be at least 8 characters long.');
        }
        return errors;
    };

    const handleSaveClick = () => {
        if (getValidationErrors().length > 0) return;
        setShowRebootInfo(true);
    };

    const handleConfirmSave = async () => {
        setShowRebootInfo(false);
        setIsSaving(true);
        try {
            await onSave(tempSettings);
            onClose();
        } catch (err) {
            alert(`Failed to save wifi settings: ${err.message}`);
        } finally {
            setIsSaving(false);
        }
    };

    if (!tempSettings) return null;

    const isDisabled = tempSettings.mode === 'disabled';
    const isApDisabled = isDisabled || tempSettings.mode === 'client';
    const isClientDisabled = isDisabled || tempSettings.mode === 'accessPoint';
    const isClientPasswordDisabled = isClientDisabled || tempSettings.client?.security === 'none';
    const hasAccessPointPasswordError = tempSettings.mode === 'accessPoint' && (tempSettings.accessPoint?.password || '').length < 8;

    return (
        <>
        <ModalWindow
            isOpen={isOpen}
            title="Edit Wifi"
            onOk={handleSaveClick}
            onCancel={onClose}
            okLabel={isSaving ? "Saving..." : "Save"}
            cancelLabel="Cancel"
            okDisabled={isSaving}
            movable={true}
            validationErrors={getValidationErrors()}
            validationWarnings={[]}
        >
            <ColumnLayout gap="0.75rem">
                <ComboBox
                    label="Mode"
                    items={MODE_OPTIONS}
                    value={tempSettings.mode}
                    onChange={updateMode}
                    labelWidth="120px"
                />

                <HorizontalSeparator label="Access Point" fullWidth={true} bleed="1rem" />
                <div style={{ opacity: isApDisabled ? 0.5 : 1, pointerEvents: isApDisabled ? 'none' : 'auto' }}>
                    <ColumnLayout gap="0.5rem">
                        <Textbox
                            label="Connection Name"
                            value={tempSettings.accessPoint?.connectionName || ''}
                            onChange={(val) => updateField('accessPoint', 'connectionName', val)}
                            disabled={isApDisabled}
                            maxLength={32}
                            labelWidth="120px"
                        />
                        <Textbox
                            label="SSID"
                            value={tempSettings.accessPoint?.ssid || ''}
                            onChange={(val) => updateField('accessPoint', 'ssid', val)}
                            disabled={isApDisabled}
                            maxLength={32}
                            labelWidth="120px"
                        />
                        <ComboBox
                            label="Security"
                            items={AP_SECURITY_OPTIONS}
                            value={tempSettings.accessPoint?.security || 'wpa-psk'}
                            onChange={(val) => updateField('accessPoint', 'security', val)}
                            disabled={true}
                            labelWidth="120px"
                        />
                        <Textbox
                            label="Password"
                            value={tempSettings.accessPoint?.password || ''}
                            onChange={(val) => updateField('accessPoint', 'password', val)}
                            disabled={isApDisabled}
                            maxLength={32}
                            labelWidth="120px"
                            outline={hasAccessPointPasswordError ? '#ef4444' : null}
                        />
                    </ColumnLayout>
                </div>

                <HorizontalSeparator label="Client" fullWidth={true} bleed="1rem" />
                <div style={{ opacity: isClientDisabled ? 0.5 : 1, pointerEvents: isClientDisabled ? 'none' : 'auto' }}>
                    <ColumnLayout gap="0.5rem">
                        <Textbox
                            label="Connection Name"
                            value={tempSettings.client?.connectionName || ''}
                            onChange={(val) => updateField('client', 'connectionName', val)}
                            disabled={isClientDisabled}
                            maxLength={32}
                            labelWidth="120px"
                        />
                        <Textbox
                            label="SSID"
                            value={tempSettings.client?.ssid || ''}
                            onChange={(val) => updateField('client', 'ssid', val)}
                            disabled={isClientDisabled}
                            maxLength={32}
                            labelWidth="120px"
                        />
                        <ComboBox
                            label="Security"
                            items={CLIENT_SECURITY_OPTIONS}
                            value={tempSettings.client?.security || 'wpa-psk'}
                            onChange={(val) => updateField('client', 'security', val)}
                            disabled={isClientDisabled}
                            labelWidth="120px"
                        />
                        <Textbox
                            label="Password"
                            value={tempSettings.client?.password || ''}
                            onChange={(val) => updateField('client', 'password', val)}
                            disabled={isClientPasswordDisabled}
                            maxLength={32}
                            labelWidth="120px"
                        />
                    </ColumnLayout>
                </div>
            </ColumnLayout>
        </ModalWindow>

        {showRebootInfo && (
            <>
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(255, 255, 255, 0.8)',
                    zIndex: 1100,
                }} />
                <div style={{
                    position: 'fixed',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    backgroundColor: '#3b82f6',
                    padding: '2rem',
                    borderRadius: '0.5rem',
                    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
                    zIndex: 1101,
                    width: '80%',
                    maxWidth: '400px',
                    border: '1px solid #93c5fd',
                    color: '#ffffff'
                }}>
                    <h3 style={{ color: '#ffffff', marginTop: 0, marginBottom: '1rem' }}>
                        Information
                    </h3>
                    <ul style={{ color: '#ffffff', paddingLeft: '1.5rem', marginBottom: '1.5rem' }}>
                        <li>This Wifi setup is effective only on Raspberry Pi.</li>
                        <li>You must reboot the system for the changes to take effect.</li>
                    </ul>
                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                        <Button
                            label="OK"
                            onClick={handleConfirmSave}
                            color="#1e40af"
                        />
                    </div>
                </div>
            </>
        )}
        </>
    );
};

export default WifiEditModal;
