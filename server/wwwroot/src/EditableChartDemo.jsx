import React, { useState } from 'react';
import EditableChart from './components/EditableChart';
import Panel from './components/Panel';

const EditableChartDemo = () => {
    const [points, setPoints] = useState([
        { x: -0.5, y: 0.3 },
        { x: 0.5, y: 0.7 }
    ]);

    return (
        <div className="page-container">
            <h1>Editable Chart Demo</h1>

            <Panel>
                <h2>Speed vs PWM Multiplier</h2>
                <p style={{ marginBottom: '1rem', color: '#666' }}>
                    This chart shows the relationship between motor speed and PWM multiplier.
                    The start and end points are fixed (green), but you can add and edit intermediate points (red).
                </p>

                <EditableChart
                    xMin={-1}
                    xMax={1}
                    yMin={0}
                    yMax={1}
                    xLabel="Speed"
                    yLabel="PWM Multiplier"
                    maxNumberOfUserPoints={10}
                    startPoint={{ x: -1, y: 0 }}
                    endPoint={{ x: 1, y: 1 }}
                    points={points}
                    onChange={setPoints}
                />

                <div style={{ marginTop: '2rem' }}>
                    <h3>Current Points:</h3>
                    <table style={{ borderCollapse: 'collapse', width: '100%', maxWidth: '400px' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid #333' }}>
                                <th style={{ padding: '0.5rem', textAlign: 'left' }}>Type</th>
                                <th style={{ padding: '0.5rem', textAlign: 'right' }}>Speed</th>
                                <th style={{ padding: '0.5rem', textAlign: 'right' }}>PWM Multiplier</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style={{ backgroundColor: '#f0fdf4' }}>
                                <td style={{ padding: '0.5rem' }}>Start (fixed)</td>
                                <td style={{ padding: '0.5rem', textAlign: 'right' }}>-1.0</td>
                                <td style={{ padding: '0.5rem', textAlign: 'right' }}>0.0</td>
                            </tr>
                            {points
                                .sort((a, b) => a.x - b.x)
                                .map((point, index) => (
                                    <tr key={index} style={{ backgroundColor: index % 2 === 0 ? '#fff' : '#f9fafb' }}>
                                        <td style={{ padding: '0.5rem' }}>Point {index + 1}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'right' }}>{point.x.toFixed(3)}</td>
                                        <td style={{ padding: '0.5rem', textAlign: 'right' }}>{point.y.toFixed(3)}</td>
                                    </tr>
                                ))}
                            <tr style={{ backgroundColor: '#f0fdf4' }}>
                                <td style={{ padding: '0.5rem' }}>End (fixed)</td>
                                <td style={{ padding: '0.5rem', textAlign: 'right' }}>1.0</td>
                                <td style={{ padding: '0.5rem', textAlign: 'right' }}>1.0</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </Panel>
        </div>
    );
};

export default EditableChartDemo;
