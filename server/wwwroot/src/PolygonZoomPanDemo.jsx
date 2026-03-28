import React, { useState } from 'react';
import Panel from './components/Panel';
import Polygon from './components/Polygon';
import ComboBox from './components/ComboBox';
import Switch from './components/Switch';
import RowLayout from './components/RowLayout';
import ColumnLayout from './components/ColumnLayout';
import StaticText from './components/StaticText';

/**
 * Demo page for testing Polygon component with zoom functionality.
 * Tests different combinations of stretchModes and modes.
 */
const PolygonZoomPanDemo = () => {
    const [stretchMode, setStretchMode] = useState('fit');
    const [mode, setMode] = useState('viewer');
    const [zoomPanEnabled, setZoomPanEnabled] = useState(true);
    const [showReticle, setShowReticle] = useState(false);

    // Sample image for testing
    const sampleImageUrl = 'https://picsum.photos/800/600';

    const stretchModeOptions = [
        { label: 'Fit', value: 'fit' },
        { label: 'Stretch', value: 'stretch' },
        { label: 'Original Size', value: 'originalSize' }
    ];

    const modeOptions = [
        { label: 'Viewer', value: 'viewer' },
        { label: 'Designer', value: 'designer' },
        { label: 'Joystick', value: 'joystick' }
    ];

    return (
        <div className="page-container" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
            <Panel header="Polygon Zoom Demo">
                <ColumnLayout>
                    <RowLayout>
                        <ComboBox
                            label="Stretch Mode"
                            options={stretchModeOptions}
                            value={stretchMode}
                            onChange={(value) => setStretchMode(value)}
                        />
                        <ComboBox
                            label="Mode"
                            options={modeOptions}
                            value={mode}
                            onChange={(value) => setMode(value)}
                        />
                        <Switch
                            label="Zoom Enabled"
                            value={zoomPanEnabled}
                            onChange={(value) => setZoomPanEnabled(value)}
                        />
                        <Switch
                            label="Show Reticle"
                            value={showReticle}
                            onChange={(value) => setShowReticle(value)}
                        />
                    </RowLayout>
                    <StaticText
                        text="Controls: Mouse wheel to zoom at cursor, Pinch gesture on touch devices to zoom"
                        style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem' }}
                    />
                </ColumnLayout>
            </Panel>

            <Panel header="Test Area" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div style={{ flex: 1, minHeight: '400px', border: '1px solid #ccc', backgroundColor: '#f5f5f5' }}>
                    <Polygon
                        src={sampleImageUrl}
                        stretchMode={stretchMode}
                        mode={mode}
                        zoomPanEnabled={zoomPanEnabled}
                        showReticle={showReticle}
                        background="#222"
                        border={0}
                    />
                </div>
            </Panel>
        </div>
    );
};

export default PolygonZoomPanDemo;
