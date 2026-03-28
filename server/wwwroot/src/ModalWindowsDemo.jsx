import React, { useState } from 'react';
import ModalWindow from './components/ModalWindow';
import Button from './components/Button';
import Panel from './components/Panel';

const ModalWindowsDemo = () => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalResult, setModalResult] = useState('');
    const [isValidationModalOpen, setIsValidationModalOpen] = useState(false);
    const [sliderValue1, setSliderValue1] = useState(50);
    const [sliderValue2, setSliderValue2] = useState(60);

    const handleOpenModal = () => {
        setIsModalOpen(true);
        setModalResult('');
    };

    const handleOpenValidationModal = () => {
        setIsValidationModalOpen(true);
        setModalResult('');
    };

    const handleOk = () => {
        setModalResult('OK clicked');
        setIsModalOpen(false);
    };

    const handleCancel = () => {
        setModalResult('Cancel clicked');
        setIsModalOpen(false);
    };

    const handleValidationOk = () => {
        setModalResult('Validation OK clicked');
        setIsValidationModalOpen(false);
    };

    const handleValidationCancel = () => {
        setModalResult('Validation Cancel clicked');
        setIsValidationModalOpen(false);
    };

    const validationErrors = [];
    if (sliderValue1 >= sliderValue2) {
        validationErrors.push('Slider 1 must be less than Slider 2');
    }

    const [isAllComponentsModalOpen, setIsAllComponentsModalOpen] = useState(false);

    // State for All Components Validation
    const [textValue, setTextValue] = useState('');
    const [numericValue, setNumericValue] = useState(0);
    const [switchValue, setSwitchValue] = useState(false);
    const [checkboxValue, setCheckboxValue] = useState(false);
    const [comboValue, setComboValue] = useState('');
    const [sliderVal, setSliderVal] = useState(50);

    const handleOpenAllComponentsModal = () => {
        setIsAllComponentsModalOpen(true);
        setModalResult('');
        // Reset values
        setTextValue('');
        setNumericValue(0);
        setSwitchValue(false);
        setCheckboxValue(false);
        setComboValue('');
        setSliderVal(50);
    };

    const handleAllComponentsOk = () => {
        setModalResult('All Components Validation OK clicked');
        setIsAllComponentsModalOpen(false);
    };

    const handleAllComponentsCancel = () => {
        setModalResult('All Components Validation Cancel clicked');
        setIsAllComponentsModalOpen(false);
    };

    // Validation Logic for All Components
    const allComponentsErrors = [];
    if (textValue.length < 3) allComponentsErrors.push('Textbox: Must be at least 3 characters');
    if (numericValue < 10) allComponentsErrors.push('Numeric Input: Must be at least 10');
    if (!switchValue) allComponentsErrors.push('Switch: Must be turned ON');
    if (checkboxValue) allComponentsErrors.push('Checkbox: Must be UNCHECKED');
    if (comboValue !== 'Option 2') allComponentsErrors.push('Combo Box: Must select "Option 2"');
    if (sliderVal > 80) allComponentsErrors.push('Slider: Must be 80 or less');

    return (
        <div style={{ padding: '20px' }}>
            <h1>Modal Windows Demo</h1>
            <Panel>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'flex-start' }}>
                    <p>Click the button below to open a modal window.</p>
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                        <Button label="Open Simple Modal" onClick={handleOpenModal} />
                        <Button label="Open Validation Modal" onClick={handleOpenValidationModal} />
                        <Button label="Open All Components Validation" onClick={handleOpenAllComponentsModal} />
                    </div>
                    {modalResult && <p>Last result: {modalResult}</p>}
                </div>
            </Panel>

            <ModalWindow
                isOpen={isModalOpen}
                title="Demo Modal"
                onOk={handleOk}
                onCancel={handleCancel}
            >
                <p>This is a demonstration of the ModalWindow component.</p>
                <p>You can put any content here.</p>
                {Array.from({ length: 20 }).map((_, i) => (
                    <p key={i}>Scrollable content line {i + 1}</p>
                ))}
            </ModalWindow>

            <ModalWindow
                isOpen={isValidationModalOpen}
                title="Validation Modal"
                onOk={handleValidationOk}
                onCancel={handleValidationCancel}
                validationErrors={validationErrors}
            >
                <p>Adjust values to trigger validation errors.</p>
                <div style={{ marginBottom: '1rem', padding: '1rem', border: sliderValue1 >= sliderValue2 ? '1px solid red' : '1px solid #ccc', borderRadius: '4px' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem' }}>Slider 1 Value: {sliderValue1}</label>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        value={sliderValue1}
                        onChange={(e) => setSliderValue1(parseInt(e.target.value))}
                        style={{ width: '100%' }}
                    />
                </div>
                <div style={{ marginBottom: '1rem', padding: '1rem', border: sliderValue1 >= sliderValue2 ? '1px solid red' : '1px solid #ccc', borderRadius: '4px' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem' }}>Slider 2 Value: {sliderValue2}</label>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        value={sliderValue2}
                        onChange={(e) => setSliderValue2(parseInt(e.target.value))}
                        style={{ width: '100%' }}
                    />
                </div>
                <p style={{ color: '#666', fontSize: '0.9rem' }}>
                    Rule: Slider 1 must be strictly less than Slider 2.
                </p>
                {Array.from({ length: 15 }).map((_, i) => (
                    <p key={i}>Extra content to force scroll {i + 1}</p>
                ))}
            </ModalWindow>

            <ModalWindow
                isOpen={isAllComponentsModalOpen}
                title="All Components Validation"
                onOk={handleAllComponentsOk}
                onCancel={handleAllComponentsCancel}
                validationErrors={allComponentsErrors}
            >
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <p>Violate the rules to see validation errors.</p>

                    {/* Textbox */}
                    <div style={{ padding: '0.5rem', border: textValue.length < 3 ? '2px solid #ef4444' : '1px solid transparent', boxShadow: textValue.length < 3 ? '0 0 8px rgba(239, 68, 68, 0.6)' : 'none', borderRadius: '4px', transition: 'all 0.2s ease' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Textbox (Min 3 chars):</label>
                        <input
                            type="text"
                            value={textValue}
                            onChange={(e) => setTextValue(e.target.value)}
                            style={{ width: '100%', padding: '0.5rem' }}
                        />
                    </div>

                    {/* Numeric Input */}
                    <div style={{ padding: '0.5rem', border: numericValue < 10 ? '2px solid #ef4444' : '1px solid transparent', boxShadow: numericValue < 10 ? '0 0 8px rgba(239, 68, 68, 0.6)' : 'none', borderRadius: '4px', transition: 'all 0.2s ease' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Numeric Input (Min 10):</label>
                        <input
                            type="number"
                            value={numericValue}
                            onChange={(e) => setNumericValue(parseInt(e.target.value) || 0)}
                            style={{ width: '100%', padding: '0.5rem' }}
                        />
                    </div>

                    {/* Switch */}
                    <div style={{ padding: '0.5rem', border: !switchValue ? '2px solid #ef4444' : '1px solid transparent', boxShadow: !switchValue ? '0 0 8px rgba(239, 68, 68, 0.6)' : 'none', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'all 0.2s ease' }}>
                        <label>Switch (Must be ON):</label>
                        <input
                            type="checkbox"
                            checked={switchValue}
                            onChange={(e) => setSwitchValue(e.target.checked)}
                        />
                        <span>{switchValue ? 'ON' : 'OFF'}</span>
                    </div>

                    {/* Checkbox */}
                    <div style={{ padding: '0.5rem', border: checkboxValue ? '2px solid #ef4444' : '1px solid transparent', boxShadow: checkboxValue ? '0 0 8px rgba(239, 68, 68, 0.6)' : 'none', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'all 0.2s ease' }}>
                        <label>Checkbox (Must be UNCHECKED):</label>
                        <input
                            type="checkbox"
                            checked={checkboxValue}
                            onChange={(e) => setCheckboxValue(e.target.checked)}
                        />
                        <span>{checkboxValue ? 'Checked' : 'Unchecked'}</span>
                    </div>

                    {/* ComboBox */}
                    <div style={{ padding: '0.5rem', border: comboValue !== 'Option 2' ? '2px solid #ef4444' : '1px solid transparent', boxShadow: comboValue !== 'Option 2' ? '0 0 8px rgba(239, 68, 68, 0.6)' : 'none', borderRadius: '4px', transition: 'all 0.2s ease' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Combo Box (Select "Option 2"):</label>
                        <select
                            value={comboValue}
                            onChange={(e) => setComboValue(e.target.value)}
                            style={{ width: '100%', padding: '0.5rem' }}
                        >
                            <option value="">Select an option</option>
                            <option value="Option 1">Option 1</option>
                            <option value="Option 2">Option 2</option>
                            <option value="Option 3">Option 3</option>
                        </select>
                    </div>

                    {/* Slider */}
                    <div style={{ padding: '0.5rem', border: sliderVal > 80 ? '2px solid #ef4444' : '1px solid transparent', boxShadow: sliderVal > 80 ? '0 0 8px rgba(239, 68, 68, 0.6)' : 'none', borderRadius: '4px', transition: 'all 0.2s ease' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Slider (Max 80): {sliderVal}</label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={sliderVal}
                            onChange={(e) => setSliderVal(parseInt(e.target.value))}
                            style={{ width: '100%' }}
                        />
                    </div>
                </div>
            </ModalWindow>
        </div>
    );
};

export default ModalWindowsDemo;
