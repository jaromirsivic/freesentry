import React, { useState } from 'react';
import Panel from './components/Panel';
import ColumnLayout from './components/ColumnLayout';
import RowLayout from './components/RowLayout';
import DateTimePicker from './components/DateTimePicker';
import StaticText from './components/StaticText';
import Button from './components/Button';

/**
 * Demo page for DateTimePicker component in the Sandbox.
 * Showcases various configurations, precisions, and use cases.
 */
const DateTimePickerDemo = () => {
    // State for different demo instances
    const [date1, setDate1] = useState(new Date());
    const [date2, setDate2] = useState(new Date(2024, 0, 15));
    const [date3, setDate3] = useState(new Date());
    const [date4, setDate4] = useState(new Date());
    const [date5, setDate5] = useState(new Date(2000, 5, 15));
    
    // State for precision demos
    const [yearOnly, setYearOnly] = useState(new Date(2024, 0, 1));
    const [monthOnly, setMonthOnly] = useState(new Date(2024, 5, 1));
    const [dayOnly, setDayOnly] = useState(new Date(2024, 5, 15));
    const [withHours, setWithHours] = useState(new Date(2024, 5, 15, 14, 0, 0));
    const [withMinutes, setWithMinutes] = useState(new Date(2024, 5, 15, 14, 30, 0));
    const [withSeconds, setWithSeconds] = useState(new Date(2024, 5, 15, 14, 30, 45));

    // Format date for display in demo
    const formatForDisplay = (date, { precision = 'seconds' } = {}) => {
        if (!date) return 'null';
        
        const options = {
            year: 'numeric'
        };
        
        if (['months', 'days', 'hours', 'minutes', 'seconds'].includes(precision)) {
            options.month = 'long';
        }
        if (['days', 'hours', 'minutes', 'seconds'].includes(precision)) {
            options.weekday = 'long';
            options.day = 'numeric';
        }
        if (['hours', 'minutes', 'seconds'].includes(precision)) {
            options.hour = '2-digit';
            options.hour12 = false;
        }
        if (['minutes', 'seconds'].includes(precision)) {
            options.minute = '2-digit';
        }
        if (precision === 'seconds') {
            options.second = '2-digit';
        }
        
        return date.toLocaleString('en-US', options);
    };

    return (
        <div className="page-container">
            <ColumnLayout gap="2rem">
                <Panel>
                    <h2>DateTimePicker Component</h2>
                    <p style={{ color: '#666', marginBottom: '1rem' }}>
                        A flexible date/time picker with calendar popup, keyboard input, and multiple precision levels.
                    </p>
                </Panel>

                {/* Precision Demos */}
                <Panel>
                    <h3>Precision Levels</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        The <code>precision</code> prop controls what level of detail the user can select.
                    </p>
                    
                    <ColumnLayout gap="1.5rem">
                        {/* Years precision */}
                        <div style={{ 
                            padding: '16px', 
                            backgroundColor: '#f8fafc', 
                            borderRadius: '8px',
                            border: '1px solid #e2e8f0'
                        }}>
                            <h4 style={{ marginBottom: '0.75rem', color: '#1f2937' }}>
                                precision="years"
                            </h4>
                            <p style={{ fontSize: '0.85rem', color: '#6b7280', marginBottom: '1rem' }}>
                                User can only select a year. Useful for birth year, graduation year, etc.
                            </p>
                            <RowLayout gap="1rem" style={{ alignItems: 'center', flexWrap: 'wrap' }}>
                                <DateTimePicker
                                    label="Year"
                                    labelWidth="80px"
                                    precision="years"
                                    value={yearOnly}
                                    onChange={setYearOnly}
                                />
                                <span style={{ 
                                    fontFamily: 'monospace', 
                                    fontSize: '0.85rem',
                                    color: '#4b5563'
                                }}>
                                    → {formatForDisplay(yearOnly, { precision: 'years' })}
                                </span>
                            </RowLayout>
                        </div>

                        {/* Months precision */}
                        <div style={{ 
                            padding: '16px', 
                            backgroundColor: '#f8fafc', 
                            borderRadius: '8px',
                            border: '1px solid #e2e8f0'
                        }}>
                            <h4 style={{ marginBottom: '0.75rem', color: '#1f2937' }}>
                                precision="months"
                            </h4>
                            <p style={{ fontSize: '0.85rem', color: '#6b7280', marginBottom: '1rem' }}>
                                User can select a month and year, but not a specific day.
                            </p>
                            <RowLayout gap="1rem" style={{ alignItems: 'center', flexWrap: 'wrap' }}>
                                <DateTimePicker
                                    label="Month"
                                    labelWidth="80px"
                                    precision="months"
                                    value={monthOnly}
                                    onChange={setMonthOnly}
                                />
                                <span style={{ 
                                    fontFamily: 'monospace', 
                                    fontSize: '0.85rem',
                                    color: '#4b5563'
                                }}>
                                    → {formatForDisplay(monthOnly, { precision: 'months' })}
                                </span>
                            </RowLayout>
                        </div>

                        {/* Days precision (default) */}
                        <div style={{ 
                            padding: '16px', 
                            backgroundColor: '#f0f9ff', 
                            borderRadius: '8px',
                            border: '1px solid #bae6fd'
                        }}>
                            <h4 style={{ marginBottom: '0.75rem', color: '#1f2937' }}>
                                precision="days" <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>(default)</span>
                            </h4>
                            <p style={{ fontSize: '0.85rem', color: '#6b7280', marginBottom: '1rem' }}>
                                User can select a specific date. This is the default behavior.
                            </p>
                            <RowLayout gap="1rem" style={{ alignItems: 'center', flexWrap: 'wrap' }}>
                                <DateTimePicker
                                    label="Date"
                                    labelWidth="80px"
                                    precision="days"
                                    value={dayOnly}
                                    onChange={setDayOnly}
                                />
                                <span style={{ 
                                    fontFamily: 'monospace', 
                                    fontSize: '0.85rem',
                                    color: '#4b5563'
                                }}>
                                    → {formatForDisplay(dayOnly, { precision: 'days' })}
                                </span>
                            </RowLayout>
                        </div>

                        {/* Hours precision */}
                        <div style={{ 
                            padding: '16px', 
                            backgroundColor: '#fef3c7', 
                            borderRadius: '8px',
                            border: '1px solid #fcd34d'
                        }}>
                            <h4 style={{ marginBottom: '0.75rem', color: '#1f2937' }}>
                                precision="hours"
                            </h4>
                            <p style={{ fontSize: '0.85rem', color: '#6b7280', marginBottom: '1rem' }}>
                                User can select date and hour. Takes DST transitions into account.
                            </p>
                            <RowLayout gap="1rem" style={{ alignItems: 'center', flexWrap: 'wrap' }}>
                                <DateTimePicker
                                    label="Date/Hour"
                                    labelWidth="80px"
                                    precision="hours"
                                    value={withHours}
                                    onChange={setWithHours}
                                />
                                <span style={{ 
                                    fontFamily: 'monospace', 
                                    fontSize: '0.85rem',
                                    color: '#4b5563'
                                }}>
                                    → {formatForDisplay(withHours, { precision: 'hours' })}
                                </span>
                            </RowLayout>
                        </div>

                        {/* Minutes precision */}
                        <div style={{ 
                            padding: '16px', 
                            backgroundColor: '#f8fafc', 
                            borderRadius: '8px',
                            border: '1px solid #e2e8f0'
                        }}>
                            <h4 style={{ marginBottom: '0.75rem', color: '#1f2937' }}>
                                precision="minutes"
                            </h4>
                            <p style={{ fontSize: '0.85rem', color: '#6b7280', marginBottom: '1rem' }}>
                                User can select date, hour, and minute.
                            </p>
                            <RowLayout gap="1rem" style={{ alignItems: 'center', flexWrap: 'wrap' }}>
                                <DateTimePicker
                                    label="DateTime"
                                    labelWidth="80px"
                                    precision="minutes"
                                    value={withMinutes}
                                    onChange={setWithMinutes}
                                />
                                <span style={{ 
                                    fontFamily: 'monospace', 
                                    fontSize: '0.85rem',
                                    color: '#4b5563'
                                }}>
                                    → {formatForDisplay(withMinutes, { precision: 'minutes' })}
                                </span>
                            </RowLayout>
                        </div>

                        {/* Seconds precision */}
                        <div style={{ 
                            padding: '16px', 
                            backgroundColor: '#f8fafc', 
                            borderRadius: '8px',
                            border: '1px solid #e2e8f0'
                        }}>
                            <h4 style={{ marginBottom: '0.75rem', color: '#1f2937' }}>
                                precision="seconds"
                            </h4>
                            <p style={{ fontSize: '0.85rem', color: '#6b7280', marginBottom: '1rem' }}>
                                User can select date, hour, minute, and second.
                            </p>
                            <RowLayout gap="1rem" style={{ alignItems: 'center', flexWrap: 'wrap' }}>
                                <DateTimePicker
                                    label="Full DateTime"
                                    labelWidth="80px"
                                    precision="seconds"
                                    value={withSeconds}
                                    onChange={setWithSeconds}
                                />
                                <span style={{ 
                                    fontFamily: 'monospace', 
                                    fontSize: '0.85rem',
                                    color: '#4b5563'
                                }}>
                                    → {formatForDisplay(withSeconds, { precision: 'seconds' })}
                                </span>
                            </RowLayout>
                        </div>
                    </ColumnLayout>
                </Panel>

                <Panel>
                    <h3>Basic Usage (Days Precision)</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Default DateTimePicker with <code>yyyy-mm-dd</code> format. Click to open calendar, type to enter date manually.
                    </p>
                    <ColumnLayout gap="1rem" style={{ maxWidth: '400px' }}>
                        <DateTimePicker
                            label="Select Date"
                            value={date1}
                            onChange={setDate1}
                        />
                        <div style={{ 
                            padding: '12px', 
                            backgroundColor: '#f8fafc', 
                            borderRadius: '6px',
                            fontFamily: 'monospace',
                            fontSize: '0.85rem'
                        }}>
                            <strong>Selected:</strong> {formatForDisplay(date1, { precision: 'days' })}
                        </div>
                    </ColumnLayout>
                </Panel>

                <Panel>
                    <h3>With Label and Custom Width</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        DateTimePicker with consistent label width for form alignment.
                    </p>
                    <ColumnLayout gap="1rem" style={{ maxWidth: '500px' }}>
                        <DateTimePicker
                            label="Start Date"
                            labelWidth="100px"
                            value={date2}
                            onChange={setDate2}
                        />
                        <DateTimePicker
                            label="End Date"
                            labelWidth="100px"
                            value={date3}
                            onChange={setDate3}
                        />
                    </ColumnLayout>
                </Panel>

                <Panel>
                    <h3>With Min/Max Constraints</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        DateTimePicker limited to dates between <strong>January 1, 2020</strong> and <strong>December 31, 2025</strong>.
                        Dates outside this range are disabled.
                    </p>
                    <ColumnLayout gap="1rem" style={{ maxWidth: '400px' }}>
                        <DateTimePicker
                            label="Constrained Date"
                            value={date4}
                            onChange={setDate4}
                            minValue={new Date(2020, 0, 1)}
                            maxValue={new Date(2025, 11, 31)}
                        />
                        <div style={{ 
                            padding: '12px', 
                            backgroundColor: '#fef3c7', 
                            borderRadius: '6px',
                            fontSize: '0.85rem'
                        }}>
                            ⚠️ Try navigating to years before 2020 or after 2025 - those dates are disabled.
                        </div>
                    </ColumnLayout>
                </Panel>

                <Panel>
                    <h3>Different Date Formats</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        DateTimePicker supports various date formats through the <code>format</code> prop.
                    </p>
                    <ColumnLayout gap="1.5rem">
                        <RowLayout gap="2rem" style={{ flexWrap: 'wrap' }}>
                            <div style={{ flex: 1, minWidth: '250px' }}>
                                <h4 style={{ marginBottom: '0.5rem' }}>ISO Format (yyyy-mm-dd)</h4>
                                <DateTimePicker
                                    value={date5}
                                    onChange={setDate5}
                                    format="yyyy-mm-dd"
                                />
                            </div>
                            <div style={{ flex: 1, minWidth: '250px' }}>
                                <h4 style={{ marginBottom: '0.5rem' }}>US Format (mm/dd/yyyy)</h4>
                                <DateTimePicker
                                    value={date5}
                                    onChange={setDate5}
                                    format="mm/dd/yyyy"
                                />
                            </div>
                            <div style={{ flex: 1, minWidth: '250px' }}>
                                <h4 style={{ marginBottom: '0.5rem' }}>European Format (dd.mm.yyyy)</h4>
                                <DateTimePicker
                                    value={date5}
                                    onChange={setDate5}
                                    format="dd.mm.yyyy"
                                />
                            </div>
                        </RowLayout>
                        <div style={{ 
                            padding: '12px', 
                            backgroundColor: '#f0f9ff', 
                            borderRadius: '6px',
                            fontSize: '0.85rem'
                        }}>
                            ℹ️ All three pickers above are bound to the same date value. Change one to see them all update.
                        </div>
                    </ColumnLayout>
                </Panel>

                <Panel>
                    <h3>Disabled State</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        DateTimePicker in disabled state - no interaction allowed.
                    </p>
                    <DateTimePicker
                        label="Disabled"
                        value={new Date(2024, 5, 15)}
                        onChange={() => {}}
                        disabled
                    />
                </Panel>

                <Panel>
                    <h3>Features</h3>
                    <ColumnLayout gap="0.75rem">
                        <StaticText text="✓ Six precision levels: years, months, days, hours, minutes, seconds" />
                        <StaticText text="✓ Click month/year to quickly jump to different months and years" />
                        <StaticText text="✓ Type date directly in the input field" />
                        <StaticText text="✓ Copy/paste dates in the specified format" />
                        <StaticText text="✓ Today/Now button for quick selection of current date/time" />
                        <StaticText text="✓ Previous/next month navigation with arrow buttons" />
                        <StaticText text="✓ Visual indicators for today's date and selected date" />
                        <StaticText text="✓ DST (Daylight Saving Time) awareness for time precision levels" />
                        <StaticText text="✓ Min/max date constraints" />
                        <StaticText text="✓ Automatic popup positioning (opens above if no space below)" />
                    </ColumnLayout>
                </Panel>

                <Panel>
                    <h3>Programmatic Control</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Set dates programmatically using buttons.
                    </p>
                    <ColumnLayout gap="1rem" style={{ maxWidth: '500px' }}>
                        <DateTimePicker
                            label="Controlled Date"
                            labelWidth="120px"
                            value={date1}
                            onChange={setDate1}
                        />
                        <RowLayout gap="0.5rem" style={{ flexWrap: 'wrap' }}>
                            <Button
                                label="Today"
                                onClick={() => setDate1(new Date())}
                            />
                            <Button
                                label="New Year 2025"
                                onClick={() => setDate1(new Date(2025, 0, 1))}
                            />
                            <Button
                                label="Christmas 2024"
                                onClick={() => setDate1(new Date(2024, 11, 25))}
                            />
                            <Button
                                label="Random"
                                onClick={() => {
                                    const year = 2000 + Math.floor(Math.random() * 50);
                                    const month = Math.floor(Math.random() * 12);
                                    const day = 1 + Math.floor(Math.random() * 28);
                                    setDate1(new Date(year, month, day));
                                }}
                            />
                        </RowLayout>
                    </ColumnLayout>
                </Panel>

                <Panel>
                    <h3>DST (Daylight Saving Time) Handling</h3>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        When using <code>precision="hours"</code> or finer, the picker accounts for DST transitions.
                        On days with a DST change, skipped or repeated hours are indicated.
                    </p>
                    <ColumnLayout gap="1rem" style={{ maxWidth: '500px' }}>
                        <DateTimePicker
                            label="Select with Time"
                            labelWidth="120px"
                            precision="minutes"
                            value={withMinutes}
                            onChange={setWithMinutes}
                        />
                        <div style={{ 
                            padding: '12px', 
                            backgroundColor: '#f0f9ff', 
                            borderRadius: '6px',
                            fontSize: '0.85rem'
                        }}>
                            ℹ️ Try selecting a date when DST changes in your timezone to see the warning indicator.
                        </div>
                    </ColumnLayout>
                </Panel>
            </ColumnLayout>
        </div>
    );
};

export default DateTimePickerDemo;

