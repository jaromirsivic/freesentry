import React, { useState, useEffect } from 'react';
import Chart2D from './components/Chart2D';
import Panel from './components/Panel';
import Switch from './components/Switch';
import NumericInput from './components/NumericInput';
import ComboBox from './components/ComboBox';
import RowLayout from './components/RowLayout';
import ColumnLayout from './components/ColumnLayout';

const Chart2DDemo = () => {
    // Chart settings
    const [pannable, setPannable] = useState(true);
    const [zoomable, setZoomable] = useState(true);
    const [zoomMode, setZoomMode] = useState('xy');
    const [scrollable, setScrollable] = useState(true);
    const [gridSize, setGridSize] = useState(10);
    const [backgroundColor, setBackgroundColor] = useState('#ffffff');
    const [gridColor, setGridColor] = useState('#e0e0e0');
    const [showLegend, setShowLegend] = useState(true);
    const [legendPosition, setLegendPosition] = useState('top-right');

    // Sample datasets
    const [datasets, setDatasets] = useState([
        {
            label: 'Sin Wave',
            color: '#3b82f6',
            lineWidth: 2,
            lineStyle: 'solid',
            data: []
        },
        {
            label: 'Cos Wave',
            color: '#ef4444',
            lineWidth: 2,
            lineStyle: 'dashed',
            data: []
        },
        {
            label: 'Linear',
            color: '#10b981',
            lineWidth: 3,
            lineStyle: 'dotted',
            data: []
        },
        {
            label: 'Quadratic',
            color: '#f59e0b',
            lineWidth: 2,
            lineStyle: 'dashdot',
            data: []
        }
    ]);

    // Generate sample data
    useEffect(() => {
        const sinData = [];
        const cosData = [];
        const linearData = [];
        const quadraticData = [];

        for (let x = 0; x <= 100; x += 2) {
            sinData.push({ x, y: 50 + 40 * Math.sin(x * Math.PI / 25) });
            cosData.push({ x, y: 50 + 40 * Math.cos(x * Math.PI / 25) });
            linearData.push({ x, y: x });
            quadraticData.push({ x, y: 100 - (x - 50) * (x - 50) / 25 });
        }

        setDatasets([
            { ...datasets[0], data: sinData },
            { ...datasets[1], data: cosData },
            { ...datasets[2], data: linearData },
            { ...datasets[3], data: quadraticData }
        ]);
    }, []);

    // Update dataset property
    const updateDataset = (index, property, value) => {
        setDatasets(prev => {
            const newDatasets = [...prev];
            newDatasets[index] = { ...newDatasets[index], [property]: value };
            return newDatasets;
        });
    };

    const lineStyleOptions = [
        { value: 'solid', label: 'Solid' },
        { value: 'dashed', label: 'Dashed' },
        { value: 'dotted', label: 'Dotted' },
        { value: 'dashdot', label: 'Dash-Dot' }
    ];

    const colorOptions = [
        { value: '#3b82f6', label: 'Blue' },
        { value: '#ef4444', label: 'Red' },
        { value: '#10b981', label: 'Green' },
        { value: '#f59e0b', label: 'Orange' },
        { value: '#8b5cf6', label: 'Purple' },
        { value: '#ec4899', label: 'Pink' },
        { value: '#06b6d4', label: 'Cyan' },
        { value: '#000000', label: 'Black' }
    ];

    const zoomModeOptions = [
        { value: 'xy', label: 'X and Y' },
        { value: 'x', label: 'X Only' }
    ];

    const legendPositionOptions = [
        { value: 'top-right', label: 'Top Right' },
        { value: 'top-left', label: 'Top Left' },
        { value: 'bottom-right', label: 'Bottom Right' },
        { value: 'bottom-left', label: 'Bottom Left' },
        { value: 'bottom-outside', label: 'Bottom Outside' }
    ];

    return (
        <div style={{ padding: '20px' }}>
            <h1>Chart2D Component Demo</h1>
            <p style={{ color: '#666', marginBottom: '20px' }}>
                A versatile 2D charting component with support for multiple datasets, pan, zoom (X-only or X+Y), and scroll functionality.
                Hover near data points to see coordinates with crosshair lines.
            </p>

            <RowLayout gap="20px" style={{ alignItems: 'flex-start' }}>
                {/* Chart */}
                <ColumnLayout gap="20px">
                    <Panel title="Chart2D Example" style={{ width: 'fit-content' }}>
                        <Chart2D
                            title="Multi-Dataset Chart"
                            xLabel="X Axis (units)"
                            yLabel="Y Axis (values)"
                            xMin={0}
                            xMax={100}
                            yMin={0}
                            yMax={100}
                            width={700}
                            height={450}
                            backgroundColor={backgroundColor}
                            gridColor={gridColor}
                            gridSize={gridSize}
                            pannable={pannable}
                            zoomable={zoomable}
                            zoomMode={zoomMode}
                            scrollable={scrollable}
                            showLegend={showLegend}
                            legendPosition={legendPosition}
                            datasets={datasets}
                        />
                    </Panel>

                    {/* Simple Chart Example */}
                    <Panel title="Simple Chart (No Interactions)" style={{ width: 'fit-content' }}>
                        <Chart2D
                            title="Temperature Over Time"
                            xLabel="Time (hours)"
                            yLabel="Temperature (°C)"
                            xMin={0}
                            xMax={24}
                            yMin={10}
                            yMax={35}
                            width={500}
                            height={300}
                            gridSize={6}
                            datasets={[
                                {
                                    label: 'Temperature',
                                    color: '#ef4444',
                                    lineWidth: 2,
                                    lineStyle: 'solid',
                                    data: [
                                        { x: 0, y: 15 },
                                        { x: 4, y: 12 },
                                        { x: 8, y: 18 },
                                        { x: 12, y: 28 },
                                        { x: 16, y: 32 },
                                        { x: 20, y: 25 },
                                        { x: 24, y: 18 }
                                    ]
                                }
                            ]}
                        />
                    </Panel>
                </ColumnLayout>

                {/* Controls */}
                <ColumnLayout gap="20px" style={{ minWidth: '300px' }}>
                    {/* Interaction Controls */}
                    <Panel title="Interaction Settings">
                        <ColumnLayout gap="15px">
                            <Switch
                                label="Pannable"
                                checked={pannable}
                                onChange={setPannable}
                            />
                            <Switch
                                label="Zoomable"
                                checked={zoomable}
                                onChange={setZoomable}
                            />
                            {zoomable && (
                                <ComboBox
                                    label="Zoom Mode"
                                    value={zoomMode}
                                    onChange={setZoomMode}
                                    options={zoomModeOptions}
                                />
                            )}
                            <Switch
                                label="Scrollable"
                                checked={scrollable}
                                onChange={setScrollable}
                            />
                            <NumericInput
                                label="Grid Divisions"
                                value={gridSize}
                                onChange={setGridSize}
                                min={2}
                                max={20}
                                step={1}
                            />
                            <Switch
                                label="Show Legend"
                                checked={showLegend}
                                onChange={setShowLegend}
                            />
                            {showLegend && (
                                <ComboBox
                                    label="Legend Position"
                                    value={legendPosition}
                                    onChange={setLegendPosition}
                                    options={legendPositionOptions}
                                />
                            )}
                        </ColumnLayout>
                    </Panel>

                    {/* Dataset Controls */}
                    <Panel title="Dataset Settings">
                        <ColumnLayout gap="20px">
                            {datasets.map((dataset, index) => (
                                <div key={index} style={{ 
                                    borderLeft: `4px solid ${dataset.color}`,
                                    paddingLeft: '10px'
                                }}>
                                    <h4 style={{ margin: '0 0 10px 0' }}>{dataset.label}</h4>
                                    <ColumnLayout gap="10px">
                                        <ComboBox
                                            label="Color"
                                            value={dataset.color}
                                            onChange={(val) => updateDataset(index, 'color', val)}
                                            options={colorOptions}
                                        />
                                        <ComboBox
                                            label="Line Style"
                                            value={dataset.lineStyle}
                                            onChange={(val) => updateDataset(index, 'lineStyle', val)}
                                            options={lineStyleOptions}
                                        />
                                        <NumericInput
                                            label="Line Width"
                                            value={dataset.lineWidth}
                                            onChange={(val) => updateDataset(index, 'lineWidth', val)}
                                            min={1}
                                            max={10}
                                            step={1}
                                        />
                                    </ColumnLayout>
                                </div>
                            ))}
                        </ColumnLayout>
                    </Panel>
                </ColumnLayout>
            </RowLayout>
        </div>
    );
};

export default Chart2DDemo;
