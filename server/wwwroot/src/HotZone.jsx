import React, { useState, useEffect } from 'react';
import Scene3D from './components/Scene3D';
import Button from './components/Button';
import HotZoneEditModal from './HotZoneEditModal';
import TextField from './components/TextField';
import { generateHotZoneSceneObjects } from './lib/HotZoneCloudPointsGenerator';
import { getHotZoneSettings } from './lib/api';

const HotZone = () => {
    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [sceneObjects, setSceneObjects] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [fov, setFov] = useState({ horizontal: 0, vertical: 0 });

    // Load initial hot zone settings and generate scene
    useEffect(() => {
        loadAndGenerateScene();
    }, []);

    const loadAndGenerateScene = async () => {
        try {
            setIsLoading(true);
            const settings = await getHotZoneSettings();
            if (settings && Object.keys(settings).length > 0) {
                const quality = settings.computationQuality || 15;
                const result = generateHotZoneSceneObjects(settings, quality);

                if (result.fov) {
                    setSceneObjects(result.objects);
                    setFov(result.fov);
                } else {
                    setSceneObjects(result);
                }
            }
        } catch (error) {
            console.error('Failed to load hot zone settings:', error);
        } finally {
            setIsLoading(false);
        }
    };

    // Override main-content padding for full-screen Scene3D
    useEffect(() => {
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            const originalPadding = mainContent.style.padding;
            const originalOverflow = mainContent.style.overflow;
            mainContent.style.padding = '0';
            mainContent.style.overflow = 'hidden';
            return () => {
                mainContent.style.padding = originalPadding;
                mainContent.style.overflow = originalOverflow;
            };
        }
    }, []);

    // Edit button style - positioned above Reset button in Scene3D
    const editButtonStyle = {
        position: 'absolute',
        bottom: '52px',
        right: '10px',
        zIndex: 10
    };

    // Match MultiSwitch default colors (same as Reset button)
    const editButtonColors = {
        backgroundColor: '#cccccc',
        color: '#333333',
        border: '1px solid #999999',
        borderRadius: '0.375rem',
        padding: '0.5rem 1rem',
        fontWeight: 400,
        opacity: 0.5,
        width: '70px',
        textAlign: 'center'
    };

    const handleEditClick = () => {
        setIsEditModalOpen(true);
    };

    const handleEditModalClose = () => {
        setIsEditModalOpen(false);
    };

    const handleSettingsSaved = (newSettings) => {
        // Regenerate the scene with the new settings
        const quality = newSettings.computationQuality || 15;
        const result = generateHotZoneSceneObjects(newSettings, quality);

        if (result.fov) {
            setSceneObjects(result.objects);
            setFov(result.fov);
            console.log('Hot zone settings saved, scene updated with', result.objects.length, 'objects');
        } else {
            setSceneObjects(result);
            console.log('Hot zone settings saved, scene updated with', result.length, 'objects');
        }
    };

    const handlePreview = (previewSettings) => {
        // Just regenerate scene without saving to backend
        const quality = previewSettings.computationQuality || 15;
        const result = generateHotZoneSceneObjects(previewSettings, quality);

        if (result.fov) {
            setSceneObjects(result.objects);
            setFov(result.fov);
        } else {
            setSceneObjects(result);
        }
    };

    return (
        <div style={{
            width: '100%',
            height: '100%',
            position: 'relative'
        }}>
            <div style={{
                position: 'absolute',
                top: '10px',
                left: '10px',
                zIndex: 10,
                display: 'flex',
                flexDirection: 'column',
                gap: '4px'
            }}>
                <TextField
                    text={`Horizontal FOV: ${fov.horizontal.toFixed(2)}°`}
                    fontSize={14}
                    fontWeight="bold"
                    fontColor="#333"
                />
                <TextField
                    text={`Vertical FOV: ${fov.vertical.toFixed(2)}°`}
                    fontSize={14}
                    fontWeight="bold"
                    fontColor="#333"
                />
            </div>

            <div style={editButtonStyle}>
                <Button
                    label="Edit"
                    onClick={handleEditClick}
                    style={editButtonColors}
                />
            </div>
            <Scene3D
                background="#ffffffff"
                gridColor="#eeeeeeff"
                objects={sceneObjects}
            />
            <HotZoneEditModal
                isOpen={isEditModalOpen}
                onClose={handleEditModalClose}
                onSave={handleSettingsSaved}
                onPreview={handlePreview}
            />
        </div>
    );
};

export default HotZone;
