import React, { useState, useEffect } from 'react';
import Panel from './components/Panel';
import Button from './components/Button';
import ModalWindow from './components/ModalWindow';
import Textbox from './components/Textbox';
import NumericInput from './components/NumericInput';
import MultiSwitch from './components/MultiSwitch';
import ColumnLayout from './components/ColumnLayout';
import StaticText from './components/StaticText';
import HorizontalSeparator from './components/HorizontalSeparator';
import SystemEditModal from './SystemEditModal';
import WifiEditModal from './WifiEditModal';
import OperationFeedback from './components/OperationFeedback';

import { getGeneralSettings, saveGeneralSettings, getSystemInfo, getSystemPlatformInfo, setSystemDateTimeAndTimezone, rebootSystem } from './lib/api';
import { createOperationFeedback } from './lib/operationFeedback';
import editIcon from './assets/icons/edit.svg';
import reloadIcon from './assets/icons/reload.svg';
import TextField from './components/TextField';

const GeneralSetup = () => {
    const [settings, setSettings] = useState({
        controllerSetup: {
            controller: 'localhost',
            remoteHost: '192.168.0.1',
            remotePort: 8888
        }
    });
    const [systemInfo, setSystemInfo] = useState({
        dateTime: '',
        timezoneName: '',
        availableTimezones: []
    });
    const [platformInfo, setPlatformInfo] = useState({
        operating_system: '',
        architecture: '',
        user: '',
        cpu: '',
        ram: ''
    });
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isSystemModalOpen, setIsSystemModalOpen] = useState(false);
    const [isWifiModalOpen, setIsWifiModalOpen] = useState(false);
    const [isRebootConfirmOpen, setIsRebootConfirmOpen] = useState(false);
    const [isRebooting, setIsRebooting] = useState(false);
    const [editingSettings, setEditingSettings] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [controllerStatus, setControllerStatus] = useState({ initialized: false, error_message: '' });
    const [controllerSaveFeedback, setControllerSaveFeedback] = useState(null);
    const [rebootFeedback, setRebootFeedback] = useState(null);
    const [isSystemRefreshing, setIsSystemRefreshing] = useState(false);

    const fetchControllerStatus = async () => {
        try {
            const response = await fetch('/api/controller/status');
            const data = await response.json();
            setControllerStatus(data);
        } catch (error) {
            console.error('Failed to fetch controller status:', error);
            setControllerStatus({ initialized: false, error_message: 'Failed to fetch status' });
        }
    };

    const fetchPlatformInfo = async () => {
        try {
            const data = await getSystemPlatformInfo();
            setPlatformInfo({
                operating_system: data.operating_system,
                architecture: data.architecture,
                user: data.user,
                cpu: data.cpu,
                ram: data.ram
            });
        } catch (error) {
            console.error('Failed to fetch platform info:', error);
        }
    };

    const fetchSystemInfo = async () => {
        try {
            const data = await getSystemInfo();
            if (data.success) {
                setSystemInfo({
                    dateTime: data.dateTime,
                    timezoneName: data.timezoneName,
                    availableTimezones: data.availableTimezones || []
                });
            }
        } catch (error) {
            console.error('Failed to fetch system info:', error);
        }
    };

    // Load initial data from API
    useEffect(() => {
        const loadSettings = async () => {
            try {
                setIsLoading(true);
                const data = await getGeneralSettings();
                if (data && data.controllerSetup) {
                    setSettings(data);
                }
                await fetchControllerStatus();
                await fetchSystemInfo();
                await fetchPlatformInfo();
            } catch (error) {
                console.error('Failed to load general settings:', error);
            } finally {
                setIsLoading(false);
            }
        };
        loadSettings();
    }, []);

    const handleEdit = () => {
        const settingsCopy = JSON.parse(JSON.stringify(settings));
        setControllerSaveFeedback(null);
        setEditingSettings(settingsCopy);
        setIsModalOpen(true);
    };

    const handleCloseModal = () => {
        setControllerSaveFeedback(null);
        setIsModalOpen(false);
        setEditingSettings(null);
    };

    const handleSave = async () => {
        if (!editingSettings) {
            return;
        }

        const savedSettings = editingSettings;
        setControllerSaveFeedback(null);

        try {
            setIsSaving(true);
            await saveGeneralSettings(savedSettings, 60000); // 60 seconds timeout
            console.log('General settings saved successfully');
            setSettings(savedSettings);
            handleCloseModal();
            // Refresh status after save (backend triggers reset)
            await fetchControllerStatus();
        } catch (error) {
            console.error('Error saving settings:', error);
            setControllerSaveFeedback(createOperationFeedback(error, 'Saving settings failed.'));
        } finally {
            setIsSaving(false);
        }
    };

    const handleSystemEdit = () => {
        setIsSystemModalOpen(true);
    };

    const handleSystemRefresh = async () => {
        try {
            setIsSystemRefreshing(true);
            await fetchSystemInfo();
            await fetchPlatformInfo();
        } finally {
            setIsSystemRefreshing(false);
        }
    };

    const handleSystemModalClose = () => {
        setIsSystemModalOpen(false);
    };

    const handleSystemSave = async (systemSettings) => {
        await setSystemDateTimeAndTimezone(systemSettings);
        // Refresh system info after save
        await fetchSystemInfo();
    };

    const handleWifiEdit = () => {
        setIsWifiModalOpen(true);
    };

    const handleWifiSave = async (wifiSettings) => {
        const updatedSettings = { ...settings, wifi: wifiSettings };
        await saveGeneralSettings(updatedSettings, 60000);
        setSettings(updatedSettings);
    };

    const handleOpenRebootConfirm = () => {
        setRebootFeedback(null);
        setIsRebootConfirmOpen(true);
    };

    const handleCloseRebootConfirm = () => {
        setRebootFeedback(null);
        setIsRebootConfirmOpen(false);
    };

    const handleReboot = async () => {
        setRebootFeedback(null);

        try {
            setIsRebooting(true);
            await rebootSystem();
            handleCloseRebootConfirm();
        } catch (error) {
            console.error('Failed to reboot system:', error);
            setRebootFeedback(createOperationFeedback(error, 'Reboot could not be confirmed.'));
        } finally {
            setIsRebooting(false);
        }
    };

    const updateControllerSetupField = (field, value) => {
        setEditingSettings(prev => ({
            ...prev,
            controllerSetup: {
                ...prev.controllerSetup,
                [field]: value
            }
        }));
    };

    const getControllerDisplayName = (controller) => {
        return controller === 'localhost' ? 'Localhost Raspberry Pi' : 'Remote pigpio daemon';
    };

    /**
     * Format datetime for display.
     * The datetime from server is already in server timezone.
     */
    const formatDateTime = (dateTimeStr) => {
        if (!dateTimeStr) return 'N/A';
        return dateTimeStr.replace('T', ' ');
    };

    const isRemoteMode = editingSettings?.controllerSetup?.controller === 'remote';
    const isCurrentRemoteMode = settings.controllerSetup.controller === 'remote';

    // Show loading state
    if (isLoading) {
        return (
            <div className="page-container">
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '2rem' }}>
                    <StaticText text="Loading settings..." />
                </div>
            </div>
        );
    }

    return (
        <div className="page-container">
            <div style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '1rem', alignItems: 'stretch' }}>
                {/* System Panel */}
                <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex' }}>
                    <Panel
                        style={{ flex: 1 }}
                        title="System"
                        headerAction={
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                                <Button
                                    label={<img src={reloadIcon} alt="Refresh" width="24" height="24" />}
                                    onClick={handleSystemRefresh}
                                    disabled={isSystemRefreshing}
                                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                                />
                                <Button
                                    label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                    onClick={handleSystemEdit}
                                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                                />
                            </div>
                        }
                    >
                        <ColumnLayout gap="0.5rem">
                            <HorizontalSeparator label="System Info" fullWidth={true} />
                            <StaticText text={<>Operating System: <span style={{ fontWeight: 'bold' }}>{platformInfo.operating_system || 'N/A'}</span></>} />
                            <StaticText text={<>Hardware: <span style={{ fontWeight: 'bold' }}>{platformInfo.architecture || 'N/A'}</span></>} />
                            <StaticText text={<>User: <span style={{ fontWeight: 'bold' }}>{platformInfo.user || 'N/A'}</span></>} />
                            <StaticText text={<>CPU: <span style={{ fontWeight: 'bold' }}>{platformInfo.cpu || 'N/A'}</span></>} />
                            <StaticText text={<>RAM: <span style={{ fontWeight: 'bold' }}>{platformInfo.ram || 'N/A'}</span></>} />
                            <HorizontalSeparator label="Date and Time" fullWidth={true} />
                            <StaticText text={<>Date and Time: <span style={{ fontWeight: 'bold' }}>{formatDateTime(systemInfo.dateTime)}</span></>} />
                            <StaticText text={<>Time Zone: <span style={{ fontWeight: 'bold' }}>{systemInfo.timezoneName || 'N/A'}</span></>} />
                            <HorizontalSeparator label="Reboot" fullWidth={true} />
                            <Button
                                label="Reboot"
                                onClick={handleOpenRebootConfirm}
                                color="#dc2626"
                            />
                        </ColumnLayout>
                    </Panel>
                </div>

                {/* Wifi Panel */}
                <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex' }}>
                    <Panel
                        style={{ flex: 1 }}
                        title="Wifi"
                        headerAction={
                            <Button
                                label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                onClick={handleWifiEdit}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            />
                        }
                    >
                        <ColumnLayout gap="0.5rem">
                            <StaticText text={<>Mode: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.mode || 'N/A'}</span></>} />
                            <HorizontalSeparator label="Access Point" fullWidth={true} />
                            <StaticText
                                text={<>Connection Name: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.accessPoint?.connectionName || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'accessPoint'}
                            />
                            <StaticText
                                text={<>SSID: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.accessPoint?.ssid || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'accessPoint'}
                            />
                            <StaticText
                                text={<>Security: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.accessPoint?.security || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'accessPoint'}
                            />
                            <StaticText
                                text={<>Password: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.accessPoint?.password || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'accessPoint'}
                            />
                            <HorizontalSeparator label="Client" fullWidth={true} />
                            <StaticText
                                text={<>Connection Name: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.client?.connectionName || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'client'}
                            />
                            <StaticText
                                text={<>SSID: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.client?.ssid || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'client'}
                            />
                            <StaticText
                                text={<>Security: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.client?.security || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'client'}
                            />
                            <StaticText
                                text={<>Password: <span style={{ fontWeight: 'bold' }}>{settings.wifi?.client?.password || 'N/A'}</span></>}
                                disabled={settings.wifi?.mode !== 'client'}
                            />
                        </ColumnLayout>
                    </Panel>
                </div>

                {/* Controller Setup Panel */}
                <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex' }}>
                    <Panel
                        style={{ flex: 1 }}
                        title="Controller Setup"
                        headerAction={
                            <Button
                                label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                onClick={handleEdit}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                                disabled={true}
                            />
                        }
                    >
                        <ColumnLayout gap="0.5rem">
                            <StaticText text={<>Controller: <span style={{ fontWeight: 'bold' }}>{getControllerDisplayName(settings.controllerSetup.controller)}</span></>} />
                            <StaticText
                                text={<>Remote Host: <span style={{ fontWeight: 'bold' }}>{settings.controllerSetup.remoteHost}</span></>}
                                disabled={!isCurrentRemoteMode}
                            />
                            <StaticText
                                text={<>Remote Port: <span style={{ fontWeight: 'bold' }}>{settings.controllerSetup.remotePort}</span></>}
                                disabled={!isCurrentRemoteMode}
                            />
                            {controllerStatus.initialized ? (
                                <TextField
                                    text="Initialized: OK"
                                    color="green"
                                    fontWeight="bold"
                                />
                            ) : (
                                <TextField
                                    text={`Error: ${controllerStatus.error_message || 'Unknown error'}`}
                                    color="red"
                                    fontWeight="bold"
                                />
                            )}
                        </ColumnLayout>
                    </Panel>
                </div>
            </div>

            {/* Controller Setup Edit Modal */}
            {editingSettings && (
                <ModalWindow
                    isOpen={isModalOpen}
                    title="Edit Controller Setup"
                    onOk={handleSave}
                    onCancel={handleCloseModal}
                    okLabel={isSaving ? "Saving..." : "Save"}
                    okDisabled={isSaving}
                    cancelDisabled={isSaving}
                >
                    <ColumnLayout gap="1rem">
                        <HorizontalSeparator label="Controller Setup" help="https://www.discussion.com" fullWidth={true} bleed="1rem" />
                        <ColumnLayout gap="0.5rem">
                            <div className="responsive-input-container" style={{ width: '100%' }}>
                                <span style={{ whiteSpace: 'nowrap' }}>Controller:</span>
                                <MultiSwitch
                                    options={[
                                        { label: 'Localhost Raspberry Pi', value: 'localhost' },
                                        { label: 'Remote pigpio daemon', value: 'remote' }
                                    ]}
                                    value={editingSettings.controllerSetup.controller}
                                    onChange={(val) => updateControllerSetupField('controller', val)}
                                    orientation="horizontal"
                                />
                            </div>

                            <Textbox
                                label="Remote Host"
                                value={editingSettings.controllerSetup.remoteHost}
                                onChange={(val) => updateControllerSetupField('remoteHost', val)}
                                disabled={!isRemoteMode}
                            />

                            <NumericInput
                                label="Remote Port"
                                labelPosition="left"
                                value={editingSettings.controllerSetup.remotePort}
                                onChange={(val) => updateControllerSetupField('remotePort', val)}
                                min={0}
                                max={65535}
                                disabled={!isRemoteMode}
                            />
                        </ColumnLayout>
                        <OperationFeedback feedback={controllerSaveFeedback} />
                    </ColumnLayout>
                </ModalWindow>
            )}

            {/* System Edit Modal */}
            <SystemEditModal
                isOpen={isSystemModalOpen}
                onClose={handleSystemModalClose}
                onSave={handleSystemSave}
                initialSettings={{
                    dateTime: systemInfo.dateTime,
                    timezone: systemInfo.timezoneName
                }}
                availableTimezones={systemInfo.availableTimezones}
            />

            {/* Wifi Edit Modal */}
            <WifiEditModal
                isOpen={isWifiModalOpen}
                onClose={() => setIsWifiModalOpen(false)}
                onSave={handleWifiSave}
                wifiSettings={settings.wifi}
            />

            {/* Reboot Confirmation Modal */}
            <ModalWindow
                isOpen={isRebootConfirmOpen}
                title="Reboot"
                onOk={handleReboot}
                onCancel={handleCloseRebootConfirm}
                okLabel={isRebooting ? "Rebooting..." : "OK"}
                okDisabled={isRebooting}
                cancelDisabled={isRebooting}
                movable={false}
            >
                <ColumnLayout gap="0.75rem">
                    <StaticText text="Do you really want to reboot the system (Raspberry Pi)?" />
                    <OperationFeedback feedback={rebootFeedback} />
                </ColumnLayout>
            </ModalWindow>
        </div>
    );
};

export default GeneralSetup;
