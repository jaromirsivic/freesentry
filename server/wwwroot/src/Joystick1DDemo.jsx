import React, { useState } from 'react';
import Panel from './components/Panel';
import ColumnLayout from './components/ColumnLayout';
import RowLayout from './components/RowLayout';
import Joystick1D from './components/Joystick1D';
import StaticText from './components/StaticText';

/**
 * Demo page for Joystick1D component in the Sandbox.
 * Showcases various configurations and use cases.
 */
const Joystick1DDemo = () => {
    // State for tracking events
    const [verticalValue, setVerticalValue] = useState(0);
    const [horizontalValue, setHorizontalValue] = useState(0);
    const [eventLog, setEventLog] = useState([]);
    const [customValue, setCustomValue] = useState(0);
    
    // Add event to log
    const logEvent = (eventName, data = null) => {
        const timestamp = new Date().toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit',
            fractionalSecondDigits: 2
        });
        const entry = data 
            ? `${timestamp} - ${eventName}: ${JSON.stringify(data)}`
            : `${timestamp} - ${eventName}`;
        setEventLog(prev => [entry, ...prev].slice(0, 20));
    };

    return (
        <div className="page-container">
            <ColumnLayout gap="2rem">
                <Panel>
                    <h2>Joystick1D Component</h2>
                    <p style={{ color: '#666', marginBottom: '1rem' }}>
                        A 1D joystick/fader control similar to audio mixer sliders. 
                        The button snaps back to the origin position when released.
                    </p>
                </Panel>

                {/* Vertical Joysticks */}
                <Panel>
                    <h3>Vertical Orientation (Default)</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Drag the button up or down. When released, it snaps back to the center (0).
                    </p>
                    
                    <RowLayout gap="2rem" style={{ flexWrap: 'wrap', justifyContent: 'center' }}>
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                value={verticalValue}
                                onChange={({ value }) => setVerticalValue(value)}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontFamily: 'monospace',
                                fontSize: '0.85rem',
                                color: '#4b5563'
                            }}>
                                Value: {verticalValue.toFixed(3)}
                            </div>
                        </div>
                        
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                buttonColor="#ff0000"
                                buttonOutline="#ff0000"
                                rulerColor="#10b981"
                                socketColor="#995555"
                                rulerVisible={false}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                Green theme
                            </div>
                        </div>
                        
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                buttonColor="#ef4444"
                                buttonOutline="#dc2626"
                                rulerColor="#ef4444"
                                rulerShowText={false}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                No labels
                            </div>
                        </div>
                    </RowLayout>
                </Panel>

                {/* Horizontal Joysticks */}
                <Panel>
                    <h3>Horizontal Orientation</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        The same component in horizontal mode. Drag left or right.
                    </p>
                    
                    <ColumnLayout gap="1.5rem">
                        <div>
                            <Joystick1D
                                orientation="horizontal"
                                value={horizontalValue}
                                onChange={({ value }) => setHorizontalValue(value)}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontFamily: 'monospace',
                                fontSize: '0.85rem',
                                color: '#4b5563'
                            }}>
                                Value: {horizontalValue.toFixed(3)}
                            </div>
                        </div>
                        
                        <div>
                            <Joystick1D
                                orientation="horizontal"
                                buttonColor="#8b5cf6"
                                buttonOutline="#7c3aed"
                                rulerColor="#8b5cf6"
                                width={400}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                Purple theme, wider
                            </div>
                        </div>
                    </ColumnLayout>
                </Panel>

                {/* Custom Range */}
                <Panel>
                    <h3>Custom Value Range</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Configure different min/max values and custom origin points.
                    </p>
                    
                    <RowLayout gap="2rem" style={{ flexWrap: 'wrap', justifyContent: 'center' }}>
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                minValue={0}
                                maxValue={100}
                                valueOrigin={50}
                                rulerLineDistance={10}
                                height={250}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                0 to 100<br/>Origin: 50
                            </div>
                        </div>
                        
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                minValue={-100}
                                maxValue={100}
                                valueOrigin={0}
                                rulerLineDistance={20}
                                height={250}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                -100 to 100<br/>Origin: 0
                            </div>
                        </div>
                        
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                minValue={0}
                                maxValue={1}
                                valueOrigin={0}
                                rulerLineDistance={0.1}
                                height={250}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                0 to 1<br/>Origin: 0 (bottom)
                            </div>
                        </div>
                    </RowLayout>
                </Panel>

                {/* Event Tracking */}
                <Panel>
                    <h3>Event Tracking</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Demonstrates the three events: <code>onStart</code>, <code>onChange</code>, and <code>onEnd</code>.
                    </p>
                    
                    <RowLayout gap="2rem" style={{ flexWrap: 'wrap' }}>
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                value={customValue}
                                onStart={() => logEvent('onStart')}
                                onChange={({ value }) => {
                                    setCustomValue(value);
                                    logEvent('onChange', { value: value.toFixed(6) });
                                }}
                                onEnd={() => logEvent('onEnd')}
                                snapAnimationDuration={0.3}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontFamily: 'monospace',
                                fontSize: '0.85rem',
                                color: '#4b5563'
                            }}>
                                Value: {customValue.toFixed(6)}
                            </div>
                        </div>
                        
                        <div style={{ 
                            flex: 1, 
                            minWidth: '300px',
                            maxHeight: '320px',
                            overflow: 'auto',
                            backgroundColor: '#1f2937',
                            borderRadius: '8px',
                            padding: '12px',
                            fontFamily: 'monospace',
                            fontSize: '0.75rem',
                            color: '#9ca3af'
                        }}>
                            <div style={{ marginBottom: '8px', color: '#60a5fa', fontWeight: 600 }}>
                                Event Log:
                            </div>
                            {eventLog.length === 0 ? (
                                <div style={{ color: '#6b7280', fontStyle: 'italic' }}>
                                    Drag the joystick to see events...
                                </div>
                            ) : (
                                eventLog.map((entry, idx) => (
                                    <div 
                                        key={idx} 
                                        style={{ 
                                            color: entry.includes('onStart') 
                                                ? '#34d399' 
                                                : entry.includes('onEnd') 
                                                    ? '#f87171' 
                                                    : '#fbbf24',
                                            marginBottom: '2px'
                                        }}
                                    >
                                        {entry}
                                    </div>
                                ))
                            )}
                        </div>
                    </RowLayout>
                </Panel>

                {/* Animation Duration */}
                <Panel>
                    <h3>Snap Animation Duration</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Compare different snap-back animation speeds.
                    </p>
                    
                    <RowLayout gap="2rem" style={{ flexWrap: 'wrap', justifyContent: 'center' }}>
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                snapAnimationDuration={0.05}
                                height={200}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                0.05s (fast)
                            </div>
                        </div>
                        
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                snapAnimationDuration={0.1}
                                height={200}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                0.1s (default)
                            </div>
                        </div>
                        
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                snapAnimationDuration={0.3}
                                height={200}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                0.3s (slow)
                            </div>
                        </div>
                        
                        <div style={{ textAlign: 'center' }}>
                            <Joystick1D
                                orientation="vertical"
                                snapAnimationDuration={0.8}
                                height={200}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#6b7280'
                            }}>
                                0.8s (very slow)
                            </div>
                        </div>
                    </RowLayout>
                </Panel>

                {/* With Background */}
                <Panel>
                    <h3>Custom Styling</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Customize background colors and themes.
                    </p>
                    
                    <RowLayout gap="2rem" style={{ flexWrap: 'wrap', justifyContent: 'center' }}>
                        <div style={{ 
                            backgroundColor: '#1f2937', 
                            padding: '1rem', 
                            borderRadius: '8px',
                            textAlign: 'center'
                        }}>
                            <Joystick1D
                                orientation="vertical"
                                backgroundColor="rgba(59, 130, 246, 0.1)"
                                height={200}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#9ca3af'
                            }}>
                                Dark theme
                            </div>
                        </div>
                        
                        <div style={{ 
                            backgroundColor: '#fef3c7', 
                            padding: '1rem', 
                            borderRadius: '8px',
                            textAlign: 'center'
                        }}>
                            <Joystick1D
                                orientation="vertical"
                                backgroundColor="rgba(245, 158, 11, 0.1)"
                                buttonColor="#f59e0b"
                                buttonOutline="#d97706"
                                rulerColor="#b45309"
                                height={200}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#92400e'
                            }}>
                                Amber theme
                            </div>
                        </div>
                        
                        <div style={{ 
                            backgroundColor: '#ecfdf5', 
                            padding: '1rem', 
                            borderRadius: '8px',
                            textAlign: 'center'
                        }}>
                            <Joystick1D
                                orientation="vertical"
                                backgroundColor="rgba(16, 185, 129, 0.1)"
                                buttonColor="#10b981"
                                buttonOutline="#059669"
                                rulerColor="#047857"
                                height={200}
                            />
                            <div style={{ 
                                marginTop: '0.5rem', 
                                fontSize: '0.75rem',
                                color: '#047857'
                            }}>
                                Emerald theme
                            </div>
                        </div>
                    </RowLayout>
                </Panel>

                {/* Features */}
                <Panel>
                    <h3>Features</h3>
                    <ColumnLayout gap="0.75rem">
                        <StaticText text="✓ Vertical and horizontal orientations" />
                        <StaticText text="✓ Smooth snap-back animation when released" />
                        <StaticText text="✓ Drag continues even when cursor leaves the component" />
                        <StaticText text="✓ Touch and mouse support" />
                        <StaticText text="✓ Configurable value range (minValue, maxValue)" />
                        <StaticText text="✓ Configurable origin position (valueOrigin)" />
                        <StaticText text="✓ Customizable colors for button, outline, and ruler" />
                        <StaticText text="✓ Ruler with configurable line distance and text visibility" />
                        <StaticText text="✓ Value precision to 6 decimal places" />
                        <StaticText text="✓ Events: onStart, onChange, onEnd" />
                    </ColumnLayout>
                </Panel>
            </ColumnLayout>
        </div>
    );
};

export default Joystick1DDemo;

