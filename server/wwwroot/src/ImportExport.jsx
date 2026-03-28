import React, { useState } from 'react';
import Panel from './components/Panel';
import Button from './components/Button';
import HorizontalSeparator from './components/HorizontalSeparator';
import FileInput from './components/FileInput';
import StaticText from './components/StaticText';
import ColumnLayout from './components/ColumnLayout';
import { getAllSettings, saveAllSettings } from './lib/api';
import defaultSettings from './assets/settings_default.json';

const ImportExport = () => {
    const [isLoading, setIsLoading] = useState(false);

    const handleImport = async (file) => {
        if (!window.confirm('Are you sure you want to import settings? Current settings for Motors, Camera, etc. will be overridden.')) {
            return;
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const settings = JSON.parse(e.target.result);
                setIsLoading(true);
                await saveAllSettings(settings);
                alert('Settings imported successfully!');
            } catch (error) {
                console.error('Error importing settings:', error);
                alert(`Failed to import settings: ${error.message}`);
            } finally {
                setIsLoading(false);
            }
        };
        reader.readAsText(file);
    };

    const handleExport = async () => {
        try {
            setIsLoading(true);
            const settings = await getAllSettings();
            const blob = new Blob([JSON.stringify(settings, null, 4)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'settings.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error exporting settings:', error);
            alert(`Failed to export settings: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFactoryReset = async () => {
        if (!window.confirm('Are you sure you want to restore factory settings? Current settings for Motors, Camera, etc. will be lost.')) {
            return;
        }

        try {
            setIsLoading(true);
            await saveAllSettings(defaultSettings);
            alert('Factory settings restored successfully!');
        } catch (error) {
            console.error('Error restoring factory settings:', error);
            alert(`Failed to restore factory settings: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    if (isLoading) {
        return (
            <div className="page-container">
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '2rem' }}>
                    <StaticText text="Processing..." />
                </div>
            </div>
        );
    }

    return (
        <div className="page-container">
            <Panel title="Import / Export">
                <ColumnLayout gap="1rem">
                    <StaticText text="You can import settings from a file (that will override the current settings) or export settings to a file or restore settings to factory defaults." />

                    <HorizontalSeparator label="Import" fullWidth={true} bleed="1.5rem" />
                    <ColumnLayout gap="0.5rem">
                        <FileInput
                            label="Import from file:"
                            accept=".json"
                            onFileSelect={handleImport}
                            disabled={isLoading}
                        />
                    </ColumnLayout>

                    <HorizontalSeparator label="Export" fullWidth={true} bleed="1.5rem" />
                    <ColumnLayout gap="0.5rem">
                        <div className="responsive-input-container" style={{ width: '100%' }}>
                            <span style={{ whiteSpace: 'nowrap' }}>Export to file:</span>
                            <Button
                                label="Export Settings"
                                onClick={handleExport}
                                disabled={isLoading}
                                style={{ width: 'auto' }}
                            />
                        </div>
                    </ColumnLayout>

                    <HorizontalSeparator label="Factory Defaults" fullWidth={true} bleed="1.5rem" />
                    <ColumnLayout gap="0.5rem">
                        <div className="responsive-input-container" style={{ width: '100%' }}>
                            <span style={{ whiteSpace: 'nowrap' }}>Restore Factory Settings:</span>
                            <Button
                                label="Restore Defaults"
                                onClick={handleFactoryReset}
                                color="#ef4444"
                                disabled={isLoading}
                                style={{ width: 'auto' }}
                            />
                        </div>
                    </ColumnLayout>
                </ColumnLayout>
            </Panel>
        </div>
    );
};

export default ImportExport;
