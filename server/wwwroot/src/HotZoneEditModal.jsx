import React, { useState, useEffect } from 'react';
import ModalWindow from './components/ModalWindow';
import MultiSwitch from './components/MultiSwitch';
import Slider from './components/Slider';
import Switch from './components/Switch';
import HorizontalSeparator from './components/HorizontalSeparator';
import { getHotZoneSettings, saveHotZoneSettings } from './lib/api';
import { generateHotZoneSceneObjects, validateHotZoneSettings } from './lib/HotZoneCloudPointsGenerator';

// Default hot zone settings
const defaultHotZoneSettings = {
    units: 'cm',
    computationQuality: 15,
    centerPole: {
        micStickRadius: 50,
        height: 150,
        yDistanceFromArmPoles: 100
    },
    armPoles: {
        height: 150,
        xDistance: 100
    },
    arms: {
        symmetricArms: false,
        leftArmMinLength: 50,
        leftArmStrokeLen: 100,
        rightArmMinLength: 50,
        rightArmStrokeLen: 100
    }
};

const HotZoneEditModal = ({ isOpen, onClose, onSave, onPreview }) => {
    const [settings, setSettings] = useState(defaultHotZoneSettings);
    const [tempSettings, setTempSettings] = useState(defaultHotZoneSettings);
    const [isLoading, setIsLoading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    // Unit options
    const unitOptions = [
        { label: 'cm', value: 'cm' },
        { label: 'in', value: 'in' }
    ];

    // Load settings when modal opens
    useEffect(() => {
        if (isOpen) {
            loadSettings();
        }
    }, [isOpen]);

    const loadSettings = async () => {
        try {
            setIsLoading(true);
            const data = await getHotZoneSettings();
            // Merge with defaults to ensure all properties exist
            const mergedSettings = {
                ...defaultHotZoneSettings,
                ...data,
                centerPole: { ...defaultHotZoneSettings.centerPole, ...data?.centerPole },
                armPoles: { ...defaultHotZoneSettings.armPoles, ...data?.armPoles },
                arms: { ...defaultHotZoneSettings.arms, ...data?.arms }
            };
            setSettings(mergedSettings);
            setTempSettings(mergedSettings);
            // Trigger initial preview? Maybe not needed if we open with current settings which usually matches scene.
        } catch (error) {
            console.error('Failed to load hot zone settings:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const getValidationErrors = () => {
        const errors = [];

        // Strict business validation
        const isValid = validateHotZoneSettings(tempSettings);
        if (!isValid) {
            errors.push("Invalid Hot Zone Settings. With this setup the motors will not be able to move from fully retracted position to fully extended position. Please adjust the hot zone settings.");
        }

        return errors;
    };

    const getValidationWarnings = () => {
        const warnings = [];
        // Removed the business validation from warnings as it is now an error.
        return warnings;
    };

    const handleSave = async () => {
        const errors = getValidationErrors();
        if (errors.length > 0) {
            return;
        }

        try {
            setIsSaving(true);
            await saveHotZoneSettings(tempSettings);
            setSettings({ ...tempSettings });
            if (onSave) {
                onSave(tempSettings);
            }
            onClose();
        } catch (error) {
            console.error('Failed to save hot zone settings:', error);
            alert(error.message || 'Failed to save settings. Please try again.');
        } finally {
            setIsSaving(false);
        }
    };

    const handleCancel = () => {
        setTempSettings({ ...settings });
        // Revert preview to saved settings
        if (onPreview) {
            onPreview(settings);
        }
        onClose();
    };

    const updateUnits = (value) => {
        const newSettings = { ...tempSettings, units: value };
        setTempSettings(newSettings);
        if (onPreview) onPreview(newSettings);
    };

    // Helpers to update specific nested properties and trigger preview
    // We need to recreate the whole object for preview

    const applyPreview = (newSettings) => {
        if (onPreview) onPreview(newSettings);
    };

    const updateCenterPole = (field, value) => {
        setTempSettings(prev => {
            const next = {
                ...prev,
                centerPole: { ...prev.centerPole, [field]: value }
            };
            return next; // State update is enough for React to re-render, but preview needs immediate data or effect
        });
    };

    // Actually, state setter callback is not good for side-effects.
    // We should compute next state first.

    const handleSliderChange = (section, field, value) => {
        setTempSettings(prev => {
            if (section === 'root') {
                return { ...prev, [field]: value };
            }
            
            let newSettings = {
                ...prev,
                [section]: { ...prev[section], [field]: value }
            };
            
            // Sync right arm values when symmetricArms is enabled and left arm values change
            if (section === 'arms' && prev.arms.symmetricArms) {
                if (field === 'leftArmMinLength') {
                    newSettings.arms.rightArmMinLength = value;
                } else if (field === 'leftArmStrokeLen') {
                    newSettings.arms.rightArmStrokeLen = value;
                }
            }
            
            return newSettings;
        });
    };

    // Handler for symmetricArms switch change
    const handleSymmetricArmsChange = (value) => {
        setTempSettings(prev => {
            const newSettings = {
                ...prev,
                arms: {
                    ...prev.arms,
                    symmetricArms: value
                }
            };
            
            // When enabling symmetric arms, copy left arm values to right arm
            if (value) {
                newSettings.arms.rightArmMinLength = prev.arms.leftArmMinLength;
                newSettings.arms.rightArmStrokeLen = prev.arms.leftArmStrokeLen;
            }
            
            return newSettings;
        });
    };

    const handleSliderAfterChange = (section, field, value) => {
        // Construct the new settings object based on current tempSettings
        const newSettings = { ...tempSettings };
        if (section === 'root') {
            newSettings[field] = value;
        } else {
            newSettings[section] = { ...tempSettings[section], [field]: value };
        }
        
        // Sync right arm values when symmetricArms is enabled and left arm values change
        if (section === 'arms' && tempSettings.arms.symmetricArms) {
            if (field === 'leftArmMinLength') {
                newSettings.arms.rightArmMinLength = value;
            } else if (field === 'leftArmStrokeLen') {
                newSettings.arms.rightArmStrokeLen = value;
            }
        }

        applyPreview(newSettings);
    };
    
    // Handler for symmetricArms switch afterChange (for preview)
    const handleSymmetricArmsAfterChange = (value) => {
        const newSettings = {
            ...tempSettings,
            arms: {
                ...tempSettings.arms,
                symmetricArms: value
            }
        };
        
        // When enabling symmetric arms, copy left arm values to right arm
        if (value) {
            newSettings.arms.rightArmMinLength = tempSettings.arms.leftArmMinLength;
            newSettings.arms.rightArmStrokeLen = tempSettings.arms.leftArmStrokeLen;
        }
        
        applyPreview(newSettings);
    };

    const unit = tempSettings.units;
    const errors = getValidationErrors();
    const hasErrors = errors.length > 0;

    return (
        <ModalWindow
            isOpen={isOpen}
            title="Hot Zone"
            onOk={handleSave}
            onCancel={handleCancel}
            okLabel={isSaving ? "Saving..." : "Save"}
            okDisabled={isSaving || isLoading || hasErrors}
            validationErrors={errors}
            validationWarnings={getValidationWarnings()}
            movable={true}
        >
            {isLoading ? (
                <div style={{ padding: '1rem', textAlign: 'center' }}>
                    Loading...
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span>Units:</span>
                        <MultiSwitch
                            options={unitOptions}
                            value={tempSettings.units}
                            onChange={updateUnits}
                        />
                    </div>

                    <Slider
                        label="Computation Grid"
                        value={tempSettings.computationQuality}
                        onChange={(value) => handleSliderChange('root', 'computationQuality', value)}
                        onAfterChange={(value) => handleSliderAfterChange('root', 'computationQuality', value)}
                        min={5}
                        max={50}
                        step={1}
                        allowManualInput={true}
                        labelWidth="120px"
                    />

                    <HorizontalSeparator label="Center Pole" fullWidth={true} bleed="1rem" />
                    <Slider
                        label={`Const Radius (${unit})`}
                        value={tempSettings.centerPole.micStickRadius}
                        onChange={(value) => handleSliderChange('centerPole', 'micStickRadius', value)}
                        onAfterChange={(value) => handleSliderAfterChange('centerPole', 'micStickRadius', value)}
                        min={10}
                        max={300}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="120px"
                    />
                    <Slider
                        label={`Height (${unit})`}
                        value={tempSettings.centerPole.height}
                        onChange={(value) => handleSliderChange('centerPole', 'height', value)}
                        onAfterChange={(value) => handleSliderAfterChange('centerPole', 'height', value)}
                        min={0}
                        max={300}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="120px"
                    />
                    <Slider
                        label={`Y Distance (${unit})`}
                        value={tempSettings.centerPole.yDistanceFromArmPoles}
                        onChange={(value) => handleSliderChange('centerPole', 'yDistanceFromArmPoles', value)}
                        onAfterChange={(value) => handleSliderAfterChange('centerPole', 'yDistanceFromArmPoles', value)}
                        min={-100}
                        max={300}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="120px"
                    />

                    <HorizontalSeparator label="Arm Poles" fullWidth={true} bleed="1rem" />
                    <Slider
                        label={`Height (${unit})`}
                        value={tempSettings.armPoles.height}
                        onChange={(value) => handleSliderChange('armPoles', 'height', value)}
                        onAfterChange={(value) => handleSliderAfterChange('armPoles', 'height', value)}
                        min={0}
                        max={300}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="100px"
                    />
                    <Slider
                        label={`X Distance (${unit})`}
                        value={tempSettings.armPoles.xDistance}
                        onChange={(value) => handleSliderChange('armPoles', 'xDistance', value)}
                        onAfterChange={(value) => handleSliderAfterChange('armPoles', 'xDistance', value)}
                        min={10}
                        max={300}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="100px"
                    />

                    <HorizontalSeparator label="Arms" fullWidth={true} bleed="1rem" />
                    <Switch
                        label="Symmetric Arms"
                        value={tempSettings.arms.symmetricArms}
                        onChange={(value) => {
                            handleSymmetricArmsChange(value);
                            handleSymmetricArmsAfterChange(value);
                        }}
                        labelWidth="180px"
                    />
                    <Slider
                        label={`Left Arm Min Length (${unit})`}
                        value={tempSettings.arms.leftArmMinLength}
                        onChange={(value) => handleSliderChange('arms', 'leftArmMinLength', value)}
                        onAfterChange={(value) => handleSliderAfterChange('arms', 'leftArmMinLength', value)}
                        min={1}
                        max={300}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="180px"
                    />
                    <Slider
                        label={`Left Arm Stroke Len. (${unit})`}
                        value={tempSettings.arms.leftArmStrokeLen}
                        onChange={(value) => handleSliderChange('arms', 'leftArmStrokeLen', value)}
                        onAfterChange={(value) => handleSliderAfterChange('arms', 'leftArmStrokeLen', value)}
                        min={1}
                        max={100}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="180px"
                    />
                    <Slider
                        label={`Right Arm Min Length (${unit})`}
                        value={tempSettings.arms.rightArmMinLength}
                        onChange={(value) => handleSliderChange('arms', 'rightArmMinLength', value)}
                        onAfterChange={(value) => handleSliderAfterChange('arms', 'rightArmMinLength', value)}
                        min={1}
                        max={300}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="180px"
                        disabled={tempSettings.arms.symmetricArms}
                    />
                    <Slider
                        label={`Right Arm Stroke Len. (${unit})`}
                        value={tempSettings.arms.rightArmStrokeLen}
                        onChange={(value) => handleSliderChange('arms', 'rightArmStrokeLen', value)}
                        onAfterChange={(value) => handleSliderAfterChange('arms', 'rightArmStrokeLen', value)}
                        min={1}
                        max={100}
                        step={0.1}
                        decimalPlaces={1}
                        allowManualInput={true}
                        labelWidth="180px"
                        disabled={tempSettings.arms.symmetricArms}
                    />
                </div>
            )}
        </ModalWindow>
    );
};

export default HotZoneEditModal;
