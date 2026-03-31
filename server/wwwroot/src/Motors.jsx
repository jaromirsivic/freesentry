import React, { useState, useEffect, useRef, useCallback } from 'react';
import Panel from './components/Panel';
import Button from './components/Button';
import ModalWindow from './components/ModalWindow';
import Textbox from './components/Textbox';
import NumericInput from './components/NumericInput';
import Switch from './components/Switch';
import MultiSwitch from './components/MultiSwitch';
import Slider from './components/Slider';
import Table from './components/Table';
import ComboBox from './components/ComboBox';
import ColumnLayout from './components/ColumnLayout';
import StaticText from './components/StaticText';
import HorizontalSeparator from './components/HorizontalSeparator';
import Chart2D from './components/Chart2D';
import Joystick1D from './components/Joystick1D';
import Polygon from './components/Polygon';

import { getMotorsSettings, saveMotorsSettings, startMotorAction, stopMotorAction, getSpeedHistogram, setMotorSpeed } from './lib/api';
import masterData from './assets/masterdata.json';

import editIcon from './assets/icons/edit.svg';
import j8schemaSvg from './assets/j8schema.svg';

const Motors = () => {
    const [motors, setMotors] = useState([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingMotor, setEditingMotor] = useState(null);
    const [editingMotorIndex, setEditingMotorIndex] = useState(-1);

    // Test Motor Modal state
    const [isTestModalOpen, setIsTestModalOpen] = useState(false);
    const [testMotor, setTestMotor] = useState(null);

    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);

    // Speed histogram data for Chart2D
    const [speedHistogramData, setSpeedHistogramData] = useState([]);

    // Histogram Quick Test Timer state
    const [motorActionTimer, setMotorActionTimer] = useState(0); // timer in ms
    const [motorActionActive, setMotorActionActive] = useState(false);
    const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
    const [showPinDiagram, setShowPinDiagram] = useState(false);
    const timerIntervalRef = useRef(null);
    const startTimeRef = useRef(null); // stores the datetime when timer started
    const activeMotorPinRef = useRef(null); // track which pin is active for cleanup

    // Test Motor modal timer refs
    const testMotorTimerRef = useRef(null); // periodic speed API timer (100ms interval)
    const joystickValueRef = useRef(0); // current joystick position for the timer to read

    // Load initial data from API
    useEffect(() => {
        const loadMotors = async () => {
            try {
                setIsLoading(true);
                const data = await getMotorsSettings();
                setMotors(data || []);
            } catch (error) {
                console.error('Failed to load motors settings:', error);
            } finally {
                setIsLoading(false);
            }
        };
        loadMotors();
    }, []);

    // Load speed histogram data
    useEffect(() => {
        const loadSpeedHistogram = async () => {
            try {
                const data = await getSpeedHistogram();
                setSpeedHistogramData(data || []);
            } catch (error) {
                console.error('Failed to load speed histogram:', error);
            }
        };
        loadSpeedHistogram();
    }, [motors]); // Reload when motors change

    // Prepare dropdown options
    const motorTypeOptions = masterData.motors.motorTypes.map(type => ({
        label: type,
        value: type
    }));

    const motorRoleOptions = masterData.motors.motorRoles.map(role => ({
        label: role,
        value: role
    }));

    const pinOptions = masterData.pins.map(pin => ({
        label: `${pin.index} - ${pin.name}`,
        value: pin.index,
        disabled: !pin.isGPIO,
        color: pin.isSPI ? '#ffffe0' : (pin.isPWM ? '#90ee90' : undefined)
    }));

    const handleEdit = (motor, index) => {
        const motorCopy = JSON.parse(JSON.stringify(motor)); // Deep copy

        // Initialize automaticCalibrationEnabled if not present (backward compatibility)
        if (motorCopy.automaticCalibrationEnabled === undefined) {
            motorCopy.automaticCalibrationEnabled = true;
        }

        // Initialize/Sync speedHistogram UI state from histogram data
        if (!motorCopy.speedHistogram) {
            motorCopy.speedHistogram = {
                pwmMultiplier: 1.0,
                defaultState: 'stop',
                histogram: []
            };
        }

        // If histogram data exists, populate the UI table from it
        if (motorCopy.histogram && Array.isArray(motorCopy.histogram)) {
            motorCopy.speedHistogram.histogram = motorCopy.histogram.map(item => [
                { value: item.pwmMultiplier !== undefined ? item.pwmMultiplier : 0 },
                { value: item.forwardSeconds !== undefined ? item.forwardSeconds : 0 },
                { value: item.reverseSeconds !== undefined ? item.reverseSeconds : 0 }
            ]);
        } else if (!motorCopy.speedHistogram.histogram || !Array.isArray(motorCopy.speedHistogram.histogram)) {
            // Default if no data anywhere
            motorCopy.speedHistogram.histogram = [
                [{ value: '0.00' }, { value: '0' }, { value: '0' }],
                [{ value: '1.00' }, { value: '10' }, { value: '10' }]
            ];
        }

        setEditingMotor(motorCopy);
        setEditingMotorIndex(index);
        setShowPinDiagram(false);
        setIsModalOpen(true);
    };

    // Timer management functions
    const startMotorActionTimer = () => {
        // Clear any existing interval
        if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
        }
        // Save current datetime and reset timer to 0
        startTimeRef.current = Date.now();
        setMotorActionTimer(0);
        setMotorActionActive(true);
        // Start new interval - calculate elapsed time from saved datetime
        timerIntervalRef.current = setInterval(() => {
            const elapsed = Date.now() - startTimeRef.current;
            setMotorActionTimer(elapsed);
        }, 100);
    };

    const stopMotorActionTimer = useCallback(({ resetUiState = true, resetTimerValue = true } = {}) => {
        if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
            timerIntervalRef.current = null;
        }
        startTimeRef.current = null;
        if (resetTimerValue) {
            setMotorActionTimer(0);
        }
        if (resetUiState) {
            setMotorActionActive(false);
        }
    }, []);

    const stopActiveMotorAction = useCallback(async ({ resetUiState = true, errorContext = 'cleanup' } = {}) => {
        const activePin = activeMotorPinRef.current;
        activeMotorPinRef.current = null;
        stopMotorActionTimer({
            resetUiState,
            resetTimerValue: resetUiState
        });

        if (activePin === null) {
            return;
        }

        try {
            await stopMotorAction(activePin);
        } catch (error) {
            console.error(`Failed to stop motor action during ${errorContext}:`, error);
        }
    }, [stopMotorActionTimer]);

    const handleCloseModal = async () => {
        await stopActiveMotorAction({ errorContext: 'modal close' });

        setIsModalOpen(false);
        setEditingMotor(null);
        setEditingMotorIndex(-1);
        setShowPinDiagram(false);
    };

    /**
     * Starts the periodic timer that sends joystick position to the speed API every 100ms.
     * @param {string} motorName - Name of the motor to control
     */
    const motorSpeedBeingSetRef = useRef(false);
    const startTestMotorTimer = (motorName) => {
        if (testMotorTimerRef.current) {
            clearInterval(testMotorTimerRef.current);
        }
        joystickValueRef.current = 0;
        motorSpeedBeingSetRef.current = false;

        testMotorTimerRef.current = setInterval(async () => {
            if (motorSpeedBeingSetRef.current) return;
            try {
                motorSpeedBeingSetRef.current = true;
                await setMotorSpeed(motorName, joystickValueRef.current);
            } catch (error) {
                console.error('Failed to set motor speed:', error);
            } finally {
                motorSpeedBeingSetRef.current = false;
            }
        }, 100);
    };

    /**
     * Stops the periodic speed API timer.
     */
    const stopTestMotorTimer = useCallback(() => {
        if (testMotorTimerRef.current) {
            clearInterval(testMotorTimerRef.current);
            testMotorTimerRef.current = null;
        }
        joystickValueRef.current = 0;
        motorSpeedBeingSetRef.current = false;
    }, []);

    // Cleanup active timers and backend motor action on unmount
    useEffect(() => {
        return () => {
            stopTestMotorTimer();
            void stopActiveMotorAction({
                resetUiState: false,
                errorContext: 'page unmount'
            });
        };
    }, [stopActiveMotorAction, stopTestMotorTimer]);

    const handleOpenTestModal = (motor) => {
        setTestMotor(motor);
        setIsTestModalOpen(true);
        startTestMotorTimer(motor.name);
    };

    const handleCloseTestModal = async () => {
        // Stop the periodic timer
        stopTestMotorTimer();
        // Stop motor when closing modal
        if (testMotor) {
            try {
                await setMotorSpeed(testMotor.name, 0);
            } catch (error) {
                console.error('Failed to stop motor on modal close:', error);
            }
        }
        setIsTestModalOpen(false);
        setTestMotor(null);
    };

    /**
     * Called when the joystick interaction starts.
     */
    const handleJoystickStart = () => {
        console.log('Joystick Started');
    };

    /**
     * Called on joystick move. Updates the ref for the periodic timer to read.
     */
    const handleJoystickMove = (data) => {
        joystickValueRef.current = data.value;
    };

    /**
     * Called when the joystick interaction ends. Resets joystick value to 0.
     */
    const handleJoystickEnd = () => {
        joystickValueRef.current = 0;
        console.log('Joystick Released');
    };



    const handleSave = async () => {
        try {
            setIsSaving(true);

            // Sync UI table data back to histogram property
            if (editingMotor.speedHistogram && Array.isArray(editingMotor.speedHistogram.histogram)) {
                editingMotor.histogram = editingMotor.speedHistogram.histogram.map(row => ({
                    pwmMultiplier: Number(row[0]?.value || 0),
                    forwardSeconds: Number(row[1]?.value || 0),
                    reverseSeconds: Number(row[2]?.value || 0)
                }));
            }

            // Update the motors list with the edited motor
            const updatedMotors = motors.map((m, index) =>
                index === editingMotorIndex ? editingMotor : m
            );

            // Save only motors section via API
            await saveMotorsSettings(updatedMotors);
            console.log('Motors settings saved successfully');

            // Update local state
            setMotors(updatedMotors);
            await handleCloseModal();
        } catch (error) {
            console.error('Error saving settings:', error);
            alert(`Failed to save settings: ${error.message}`);
        } finally {
            setIsSaving(false);
        }
    };

    const updateMotorField = (field, value) => {
        let newValue = value;
        if (['forwardPin', 'reversePin', 'forwardEnablePin', 'reverseEnablePin', 'mcp3008ForwardPin', 'mcp3008ReversePin'].includes(field)) {
            newValue = parseInt(value, 10);
        }

        if (field === 'name') {
            // Validation: Max 128 chars, alphanumeric + space + underscore + hyphen
            if (value.length > 128) return;
            if (!/^[a-zA-Z0-9 _-]*$/.test(value)) return;
        }
        setEditingMotor(prev => ({ ...prev, [field]: newValue }));
    };

    const updateDutyCycleField = (field, value) => {
        setEditingMotor(prev => ({
            ...prev,
            dutyCycle: {
                ...prev.dutyCycle,
                [field]: value
            }
        }));
    };

    const updateSpeedHistogramField = (field, value) => {
        setEditingMotor(prev => ({
            ...prev,
            speedHistogram: {
                ...prev.speedHistogram,
                [field]: value
            }
        }));
    };

    const getValidationErrors = () => {
        if (!editingMotor) return [];

        const errors = [];

        // Name validation
        if (!editingMotor.name || editingMotor.name.trim() === '') {
            errors.push('Name cannot be empty');
        }

        if (editingMotor.name && editingMotor.name.length > 128) {
            errors.push('Name cannot exceed 128 characters');
        }

        if (editingMotor.name && editingMotor.name.trim() && !/^[a-zA-Z0-9 _-]+$/.test(editingMotor.name)) {
            errors.push('Name can only contain letters, numbers, spaces, underscores, and hyphens');
        }

        // Name uniqueness validation
        const duplicateMotor = motors.find((m, index) =>
            m.name === editingMotor.name && index !== editingMotorIndex
        );

        if (duplicateMotor) {
            errors.push('A motor with this name already exists');
        }

        // Speed Histogram Validations
        if (editingMotor.speedHistogram && Array.isArray(editingMotor.speedHistogram.histogram)) {
            const histogram = editingMotor.speedHistogram.histogram;

            // 1. Table should contain at least two rows.
            if (histogram.length < 2) {
                errors.push('Speed Histogram must strictly contain at least two rows');
            } else {
                let previousPwm = -1;
                let previousForward = 10001; // higher than max
                let previousReverse = 10001; // higher than max

                for (let i = 0; i < histogram.length; i++) {
                    const row = histogram[i];
                    const pwmStr = String(row[0]?.value || '');
                    const forwardStr = String(row[1]?.value || '');
                    const reverseStr = String(row[2]?.value || '');

                    const pwm = Number(pwmStr);
                    const forward = Number(forwardStr);
                    const reverse = Number(reverseStr);

                    // const isPwmNumber = !isNaN(pwm) && pwmStr.trim() !== '';
                    // const isForwardNumber = !isNaN(forward) && forwardStr.trim() !== '';
                    // const isReverseNumber = !isNaN(reverse) && reverseStr.trim() !== '';

                    // // 2. Each cell in the table must be a number...
                    // if (!isPwmNumber) {
                    //     errors.push(`Row ${i + 1}: "PWM Multiplier" must be a valid number`);
                    // }
                    // if (!isForwardNumber) {
                    //     errors.push(`Row ${i + 1}: "Forward sec." must be a valid number`);
                    // }
                    // if (!isReverseNumber) {
                    //     errors.push(`Row ${i + 1}: "Reverse sec." must be a valid number`);
                    // }

                    // ...with at most 3 decimal places.
                    const decimalPlaces = (numStr) => {
                        if (numStr.includes('.')) {
                            return numStr.split('.')[1].length;
                        }
                        return 0;
                    };
                    if (decimalPlaces(pwmStr) > 3) {
                        errors.push(`Row ${i + 1}: "PWM Multiplier" cannot have more than 3 decimal places`);
                    }
                    if (decimalPlaces(forwardStr) > 3) {
                        errors.push(`Row ${i + 1}: \"Forward sec.\" cannot have more than 3 decimal places`);
                    }
                    if (decimalPlaces(reverseStr) > 3) {
                        errors.push(`Row ${i + 1}: \"Reverse sec.\" cannot have more than 3 decimal places`);
                    }

                    // 3. All values in the column "PWM Multiplier" must be between 0.0 and 1.0.
                    if (pwm < 0.0 || pwm > 1.0) {
                        errors.push(`Row ${i + 1}: \"PWM Multiplier\" must be between 0.0 and 1.0`);
                    }

                    // 6. All values in the column "Forward sec." and "Reverse sec." must be between 0 and 10000.
                    if (forward < 0 || forward > 10000) {
                        errors.push(`Row ${i + 1}: \"Forward sec.\" must be between 0 and 10000`);
                    }
                    if (reverse < 0 || reverse > 10000) {
                        errors.push(`Row ${i + 1}: \"Reverse sec.\" must be between 0 and 10000`);
                    }

                    // 4. The first row (column "PWM Multiplier") should be 0.0.
                    if (i === 0) {
                        if (pwm !== 0.0) {
                            errors.push('PWM Multiplier in the first row must be 0.0');
                        }
                    }

                    // 5. The last row (column "PWM Multiplier") should be 1.0.
                    if (i === histogram.length - 1) {
                        if (pwm !== 1.0) {
                            errors.push('PWM Multiplier in the last row must be 1.0');
                        }
                    }

                    // Check strict ordering (except first row)
                    if (i > 0) {
                        // 7. Each value in the column "PWM Multiplier" at each row except the first one must be greater than the value at the previous row.
                        if (pwm <= previousPwm) {
                            errors.push(`Row ${i + 1}: \"PWM Multiplier\" must be greater than the value in the cell above`);
                        }

                        // 8. Each value in the column "Forward sec." at each row except the first one must be lower than the value at the previous row, once a non-zero value is encountered.
                        if (previousForward !== 0 && forward >= previousForward) {
                            errors.push(`Row ${i + 1}: \"Forward sec.\" must be lower than the value in the cell above`);
                        }

                        // 9. Each value in the column "Reverse sec." at each row except the first one must be lower than the value at the previous row, once a non-zero value is encountered.
                        if (previousReverse !== 0 && reverse >= previousReverse) {
                            errors.push(`Row ${i + 1}: \"Reverse sec.\" must be lower than the value in the cell above`);
                        }
                    }

                    if (i > 0) {
                        // 8. Each value in the column "Forward sec." at each row except the first one must be lower than the value at the previous row, once a non-zero value is encountered.
                        if (previousForward != 0 && forward === 0) {
                            errors.push(`Row ${i + 1}: \"Forward sec.\" must not be 0 if the value in the cell above is non-zero`);
                        }

                        // 9. Each value in the column "Reverse sec." at each row except the first one must be lower than the value at the previous row, once a non-zero value is encountered.
                        if (previousReverse != 0 && reverse === 0) {
                            errors.push(`Row ${i + 1}: \"Reverse sec.\" must not be 0 if the value in the cell above is non-zero`);
                        }
                    }

                    previousPwm = pwm;
                    previousForward = forward;
                    previousReverse = reverse;
                }
            }
        }

        return errors;
    };

    /**
     * Returns validation warnings (non-blocking) for pin conflicts.
     * Checks Forward, Reverse, Forward Enable, and Reverse Enable pins
     * against all pins used by other motors AND within the same motor.
     * Exception: Pin 0 (Dummy GPIO) can be used multiple times.
     */
    const getValidationWarnings = () => {
        if (!editingMotor) return [];
        const warnings = [];

        const forwardPin = editingMotor.forwardPin;
        const reversePin = editingMotor.reversePin;
        const forwardEnablePin = editingMotor.forwardEnablePin ?? 0;
        const reverseEnablePin = editingMotor.reverseEnablePin ?? 0;

        /**
         * Helper to find if a pin is used by any other motor.
         * Returns the conflicting motor and which pin type it conflicts with.
         */
        const findOtherMotorConflict = (pinValue) => {
            if (pinValue === 0) return null; // Pin 0 (Dummy GPIO) is exempt

            for (let i = 0; i < motors.length; i++) {
                if (i === editingMotorIndex) continue;
                const m = motors[i];

                if (m.forwardPin === pinValue) return { motor: m, pinType: 'Forward Pin' };
                if (m.reversePin === pinValue) return { motor: m, pinType: 'Reverse Pin' };
                if ((m.forwardEnablePin ?? 0) === pinValue) return { motor: m, pinType: 'Forward Enable Pin' };
                if ((m.reverseEnablePin ?? 0) === pinValue) return { motor: m, pinType: 'Reverse Enable Pin' };
            }
            return null;
        };

        // Check Forward Pin conflicts
        if (forwardPin !== 0) {
            // Check conflict with other motors
            const otherConflict = findOtherMotorConflict(forwardPin);
            if (otherConflict) {
                warnings.push(`Forward Pin ${forwardPin} is already used as ${otherConflict.pinType} by motor '${otherConflict.motor.name}'`);
            }
            // Check conflict within same motor
            if (forwardPin === reversePin) {
                warnings.push(`Forward Pin ${forwardPin} conflicts with Reverse Pin`);
            }
            if (forwardPin === forwardEnablePin) {
                warnings.push(`Forward Pin ${forwardPin} conflicts with Forward Enable Pin`);
            }
            if (forwardPin === reverseEnablePin) {
                warnings.push(`Forward Pin ${forwardPin} conflicts with Reverse Enable Pin`);
            }
        }

        // Check Reverse Pin conflicts
        if (reversePin !== 0) {
            // Check conflict with other motors
            const otherConflict = findOtherMotorConflict(reversePin);
            if (otherConflict) {
                warnings.push(`Reverse Pin ${reversePin} is already used as ${otherConflict.pinType} by motor '${otherConflict.motor.name}'`);
            }
            // Check conflict within same motor (skip Forward since already checked above)
            if (reversePin === forwardEnablePin) {
                warnings.push(`Reverse Pin ${reversePin} conflicts with Forward Enable Pin`);
            }
            if (reversePin === reverseEnablePin) {
                warnings.push(`Reverse Pin ${reversePin} conflicts with Reverse Enable Pin`);
            }
        }

        // Check Forward Enable Pin conflicts
        if (forwardEnablePin !== 0) {
            // Check conflict with other motors
            const otherConflict = findOtherMotorConflict(forwardEnablePin);
            if (otherConflict) {
                warnings.push(`Forward Enable Pin ${forwardEnablePin} is already used as ${otherConflict.pinType} by motor '${otherConflict.motor.name}'`);
            }
            // Check conflict within same motor (skip already checked above)
            if (forwardEnablePin === reverseEnablePin) {
                warnings.push(`Forward Enable Pin ${forwardEnablePin} conflicts with Reverse Enable Pin`);
            }
        }

        // Check Reverse Enable Pin conflicts with other motors only (same motor already checked above)
        if (reverseEnablePin !== 0) {
            const otherConflict = findOtherMotorConflict(reverseEnablePin);
            if (otherConflict) {
                warnings.push(`Reverse Enable Pin ${reverseEnablePin} is already used as ${otherConflict.pinType} by motor '${otherConflict.motor.name}'`);
            }
        }

        return warnings;
    };

    const getNameValidationError = () => {
        if (!editingMotor) return false;
        if (!editingMotor.name || editingMotor.name.trim() === '') return true;
        if (editingMotor.name.length > 128) return true;
        if (!/^[a-zA-Z0-9 _-]+$/.test(editingMotor.name)) return true;

        const duplicateMotor = motors.find((m, index) =>
            m.name === editingMotor.name && index !== editingMotorIndex
        );

        return !!duplicateMotor;
    };

    const getPinDisplayName = (pinIndex) => {
        const pin = masterData.pins.find(p => p.index === pinIndex);
        return pin ? `${pin.index} - ${pin.name}` : pinIndex;
    };

    /**
     * Helper to check if a pin is used by any other motor.
     * @param {number} pinValue - The pin index to check.
     * @returns {boolean} True if the pin is used by another motor.
     */
    const isPinUsedByOtherMotor = (pinValue) => {
        if (pinValue === 0) return false; // Pin 0 (Dummy GPIO) is exempt

        for (let i = 0; i < motors.length; i++) {
            if (i === editingMotorIndex) continue;
            const m = motors[i];

            if (m.forwardPin === pinValue) return true;
            if (m.reversePin === pinValue) return true;
            if ((m.forwardEnablePin ?? 0) === pinValue) return true;
            if ((m.reverseEnablePin ?? 0) === pinValue) return true;
        }
        return false;
    };

    /**
     * Returns true if Forward Pin has a warning (conflict with any other pin).
     * Checks: other motors AND same motor's other pins.
     */
    const getForwardPinWarning = () => {
        if (!editingMotor) return false;
        const pin = editingMotor.forwardPin;
        if (pin === 0) return false;

        // Check conflict with other motors
        if (isPinUsedByOtherMotor(pin)) return true;

        // Check conflict within same motor
        if (pin === editingMotor.reversePin) return true;
        if (pin === (editingMotor.forwardEnablePin ?? 0)) return true;
        if (pin === (editingMotor.reverseEnablePin ?? 0)) return true;

        return false;
    };

    /**
     * Returns true if Reverse Pin has a warning (conflict with any other pin).
     * Checks: other motors AND same motor's other pins.
     */
    const getReversePinWarning = () => {
        if (!editingMotor) return false;
        const pin = editingMotor.reversePin;
        if (pin === 0) return false;

        // Check conflict with other motors
        if (isPinUsedByOtherMotor(pin)) return true;

        // Check conflict within same motor
        if (pin === editingMotor.forwardPin) return true;
        if (pin === (editingMotor.forwardEnablePin ?? 0)) return true;
        if (pin === (editingMotor.reverseEnablePin ?? 0)) return true;

        return false;
    };

    /**
     * Returns true if Forward Enable Pin has a warning (conflict with any other pin).
     * Checks: other motors AND same motor's other pins.
     */
    const getForwardEnablePinWarning = () => {
        if (!editingMotor) return false;
        const pin = editingMotor.forwardEnablePin ?? 0;
        if (pin === 0) return false;

        // Check conflict with other motors
        if (isPinUsedByOtherMotor(pin)) return true;

        // Check conflict within same motor
        if (pin === editingMotor.forwardPin) return true;
        if (pin === editingMotor.reversePin) return true;
        if (pin === (editingMotor.reverseEnablePin ?? 0)) return true;

        return false;
    };

    /**
     * Returns true if Reverse Enable Pin has a warning (conflict with any other pin).
     * Checks: other motors AND same motor's other pins.
     */
    const getReverseEnablePinWarning = () => {
        if (!editingMotor) return false;
        const pin = editingMotor.reverseEnablePin ?? 0;
        if (pin === 0) return false;

        // Check conflict with other motors
        if (isPinUsedByOtherMotor(pin)) return true;

        // Check conflict within same motor
        if (pin === editingMotor.forwardPin) return true;
        if (pin === editingMotor.reversePin) return true;
        if (pin === (editingMotor.forwardEnablePin ?? 0)) return true;

        return false;
    };

    // Show loading state
    if (isLoading) {
        return (
            <div className="page-container">
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '2rem' }}>
                    <StaticText text="Loading motors settings..." />
                </div>
            </div>
        );
    }

    return (
        <div className="page-container">
            <div style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '1rem', alignItems: 'stretch' }}>
                {motors.map((motor, index) => (
                    <div key={index} style={{ flex: '1 1 400px', minWidth: '300px', display: 'flex' }}>
                        <Panel
                            style={{ flex: 1 }}
                            backgroundColor={motor.enabled ? undefined : '#c8d8e0'}
                            title={
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <div style={{
                                    width: '24px',
                                    height: '24px',
                                    borderRadius: '50%',
                                    backgroundColor: motor.color || '#888888',
                                    flexShrink: 0
                                }} />
                                <span style={{ color: '#333' }}>{motor.name} {motor.enabled ? '' : '(Disabled)'}</span>
                            </div>
                        }
                        headerAction={
                            <Button
                                label={<img src={editIcon} alt="Edit" width="24" height="24" />}
                                onClick={() => handleEdit(motor, index)}
                                style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            />
                        }
                    >

                        <div style={{ opacity: motor.enabled ? 1 : 0.5 }}>
                            <ColumnLayout gap="0.5rem">
                                <StaticText text={<>Inertia: <span style={{ fontWeight: 'bold' }}>{motor.inertia}</span></>} />
                                <StaticText text={<>Stroke Length: <span style={{ fontWeight: 'bold' }}>{motor.strokeLength ?? 0} cm</span></>} />

                                <Button
                                    label="Test Motor"
                                    onClick={() => handleOpenTestModal(motor)}
                                    color="#3b82f6"
                                    style={{ marginTop: '0.5rem', width: '100%' }}
                                />

                                <HorizontalSeparator label="Pins (BTS7960B)" fullWidth={true} bleed="1.5rem" />
                                <StaticText text={<>Forward: <span style={{ fontWeight: 'bold' }}>{getPinDisplayName(motor.forwardPin)}</span></>} />
                                <StaticText text={<>Forward Enable: <span style={{ fontWeight: 'bold' }}>{getPinDisplayName(motor.forwardEnablePin ?? 0)}</span></>} />
                                <StaticText text={<>Reverse: <span style={{ fontWeight: 'bold' }}>{getPinDisplayName(motor.reversePin)}</span></>} />
                                <StaticText text={<>Reverse Enable: <span style={{ fontWeight: 'bold' }}>{getPinDisplayName(motor.reverseEnablePin ?? 0)}</span></>} />
                                <StaticText text={<>PWM Frequency: <span style={{ fontWeight: 'bold' }}>{motor.pwmFrequency ?? 0} Hz</span></>} />
                                
                                <HorizontalSeparator label="Duty Cycle" fullWidth={true} bleed="1.5rem" />
                                <StaticText text={<>Enabled: <span style={{ fontWeight: 'bold' }}>{motor.dutyCycle.enabled ? 'Yes' : 'No'}</span></>} />
                                <StaticText text={<>Max Run: <span style={{ fontWeight: 'bold' }}>{motor.dutyCycle.maxRunningTimeSeconds}s</span></>} />
                                <StaticText text={<>Min Rest: <span style={{ fontWeight: 'bold' }}>{motor.dutyCycle.minRestTimeSeconds}s</span></>} />

                                <HorizontalSeparator label="Histogram" fullWidth={true} bleed="1.5rem" />
                                {(() => {
                                    const histogramData = speedHistogramData.find(h => h.motorName === motor.name);
                                    if (!histogramData || histogramData.error) {
                                        return <StaticText text={<span style={{ color: '#999' }}>{histogramData?.error || 'No histogram data available'}</span>} />;
                                    }
                                    return (
                                        <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
                                            <Chart2D
                                                title="Forward Speed Histogram"
                                                xLabel="Speed %"
                                                yLabel="PWM Multiplier"
                                                xMin={0}
                                                xMax={100}
                                                yMin={0}
                                                yMax={1}
                                                width="100%"
                                                height={180}
                                                gridSize={5}
                                                showLegend={false}
                                                backgroundColor="transparent"
                                                datasets={[
                                                    {
                                                        label: 'Forward',
                                                        color: '#3b82f6',
                                                        lineWidth: 2,
                                                        lineStyle: 'solid',
                                                        data: histogramData.forward.map(point => ({ x: point.x * 100, y: point.y }))
                                                    }
                                                ]}
                                            />
                                            <Chart2D
                                                title="Reverse Speed Histogram"
                                                xLabel="Speed %"
                                                yLabel="PWM Multiplier"
                                                xMin={0}
                                                xMax={100}
                                                yMin={0}
                                                yMax={1}
                                                width="100%"
                                                height={180}
                                                gridSize={5}
                                                showLegend={false}
                                                backgroundColor="transparent"
                                                datasets={[
                                                    {
                                                        label: 'Reverse',
                                                        color: '#ef4444',
                                                        lineWidth: 2,
                                                        lineStyle: 'solid',
                                                        data: histogramData.reverse.map(point => ({ x: point.x * 100, y: point.y }))
                                                    }
                                                ]}
                                            />
                                        </div>
                                    );
                                })()}
                            </ColumnLayout>
                        </div>
                    </Panel>
                    </div>
                ))}
            </div >

            {editingMotor && (
                <ModalWindow
                    isOpen={isModalOpen}
                    title={`Edit ${editingMotor.name}`}
                    onOk={handleSave}
                    onCancel={handleCloseModal}
                    okLabel={isSaving ? "Saving..." : "Save"}
                    okDisabled={isSaving}
                    validationErrors={getValidationErrors()}
                    validationWarnings={getValidationWarnings()}
                >
                    {/* ... Existing Edit Modal Content ... */}
                    <ColumnLayout gap="0.5rem">
                        <Textbox
                            label="Name"
                            value={editingMotor.name}
                            onChange={(val) => updateMotorField('name', val)}
                            disabled={editingMotor.role !== 'general'}
                            outline={getNameValidationError() ? '#ef4444' : null}
                        />

                        <Slider
                            label="Inertia"
                            value={editingMotor.inertia}
                            onChange={(val) => updateMotorField('inertia', Math.round(val * 100) / 100)}
                            min={0}
                            max={2}
                            step={0.01}
                            allowManualInput={true}
                            decimalPlaces={2}
                        />

                        <Slider
                            label="Stroke Length (cm)"
                            value={editingMotor.strokeLength ?? 0}
                            onChange={(val) => updateMotorField('strokeLength', Math.round(val * 10) / 10)}
                            min={0}
                            max={1000}
                            minSlider={1}
                            maxSlider={100}
                            step={0.5}
                            allowManualInput={true}
                            decimalPlaces={1}
                        />

                        <HorizontalSeparator label="Pins (BTS7960B)" fullWidth={true} bleed="1rem" />
                        <ColumnLayout gap="0.5rem">
                            <ComboBox
                                label="Forward"
                                items={pinOptions}
                                value={editingMotor.forwardPin}
                                onChange={(val) => updateMotorField('forwardPin', val)}
                                outline={getForwardPinWarning() ? '#f59e0b' : null}
                            />
                            <ComboBox
                                label="Forward Enable"
                                items={pinOptions}
                                value={editingMotor.forwardEnablePin ?? 0}
                                onChange={(val) => updateMotorField('forwardEnablePin', val)}
                                outline={getForwardEnablePinWarning() ? '#f59e0b' : null}
                            />
                            <ComboBox
                                label="Reverse"
                                items={pinOptions}
                                value={editingMotor.reversePin}
                                onChange={(val) => updateMotorField('reversePin', val)}
                                outline={getReversePinWarning() ? '#f59e0b' : null}
                            />
                            <ComboBox
                                label="Reverse Enable"
                                items={pinOptions}
                                value={editingMotor.reverseEnablePin ?? 0}
                                onChange={(val) => updateMotorField('reverseEnablePin', val)}
                                outline={getReverseEnablePinWarning() ? '#f59e0b' : null}
                            />
                            <Slider
                                label="PWM Frequency (Hz)"
                                value={editingMotor.pwmFrequency ?? 0}
                                onChange={(val) => updateMotorField('pwmFrequency', Math.round(val))}
                                min={0}
                                max={16000}
                                step={1}
                                allowManualInput={true}
                                decimalPlaces={0}
                            />
                            <Switch
                                label="Display Pin Diagram"
                                value={showPinDiagram}
                                onChange={(val) => setShowPinDiagram(val)}
                            />
                            {showPinDiagram && (
                                <div style={{ width: '100%', height: '360px', border: '1px solid #cbd5e1', borderRadius: '0.375rem', overflow: 'hidden' }}>
                                    <Polygon
                                        src={j8schemaSvg}
                                        stretchMode="fit"
                                        mode="viewer"
                                        showReticle={false}
                                        zoomPanEnabled={true}
                                        polygons={[]}
                                        style={{ width: '100%', height: '100%' }}
                                    />
                                </div>
                            )}
                        </ColumnLayout>

                        <HorizontalSeparator label="Advanced Options" fullWidth={true} bleed="1rem" />
                        <Switch
                            label="Display Advanced Options"
                            value={showAdvancedOptions}
                            onChange={(val) => setShowAdvancedOptions(val)}
                        />

                        {showAdvancedOptions && (
                            <>
                                <HorizontalSeparator label="Duty Cycle" fullWidth={true} bleed="1rem" />
                                <ColumnLayout gap="0.5rem">
                                    <Switch
                                        label="Enabled"
                                        value={editingMotor.dutyCycle.enabled}
                                        onChange={(val) => updateDutyCycleField('enabled', val)}
                                    />
                                    <NumericInput
                                        label="Max Run Time (s)"
                                        labelPosition="left"
                                        value={editingMotor.dutyCycle.maxRunningTimeSeconds}
                                        onChange={(val) => updateDutyCycleField('maxRunningTimeSeconds', val)}
                                        min={0}
                                        disabled={!editingMotor.dutyCycle.enabled}
                                    />
                                    <NumericInput
                                        label="Min Rest Time (s)"
                                        labelPosition="left"
                                        value={editingMotor.dutyCycle.minRestTimeSeconds}
                                        onChange={(val) => updateDutyCycleField('minRestTimeSeconds', val)}
                                        min={0}
                                        disabled={!editingMotor.dutyCycle.enabled}
                                    />
                                </ColumnLayout>

                                <HorizontalSeparator label="Histogram Quick Test" fullWidth={true} bleed="1rem" />
                                <ColumnLayout gap="1rem">
                                    <p style={{ fontSize: '0.9rem', color: '#666', margin: 0 }}>
                                        Use "PWM Multiplier" and "Reverse" / "Forward" buttons to test how long it takes to extend the stroke of linear actuator
                                        from fully retracted position to fully extended position and vice versa.
                                    </p>
                                    <Slider
                                        label="PWM Multiplier"
                                        value={editingMotor.speedHistogram.pwmMultiplier}
                                        onChange={(val) => updateSpeedHistogramField('pwmMultiplier', val)}
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        allowManualInput={true}
                                        decimalPlaces={2}
                                    />

                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                            <StaticText text="Motor Action" />
                                            <MultiSwitch
                                                options={[
                                                    { label: 'Reverse', value: 'reverse' },
                                                    { label: 'Stop', value: 'stop' },
                                                    { label: 'Forward', value: 'forward' }
                                                ]}
                                                value={editingMotor.speedHistogram.defaultState}
                                                onChange={async (val) => {
                                                    updateSpeedHistogramField('defaultState', val);

                                                    if (val === 'forward' || val === 'reverse') {
                                                        const pinIndex = val === 'forward'
                                                            ? editingMotor.forwardPin
                                                            : editingMotor.reversePin;
                                                        const pwmMultiplier = editingMotor.speedHistogram.pwmMultiplier;

                                                        try {
                                                            if (activeMotorPinRef.current !== null && activeMotorPinRef.current !== pinIndex) {
                                                                await stopMotorAction(activeMotorPinRef.current);
                                                                activeMotorPinRef.current = null;
                                                            }

                                                            await startMotorAction(pinIndex, pwmMultiplier);
                                                            activeMotorPinRef.current = pinIndex;
                                                            startMotorActionTimer();
                                                        } catch (error) {
                                                            console.error('Failed to start motor action:', error);
                                                        }
                                                    } else if (val === 'stop') {
                                                        await stopActiveMotorAction({ errorContext: 'stop selection' });
                                                    }
                                                }}
                                                orientation="horizontal"
                                            />
                                        </div>
                                        <StaticText
                                            text={`Timer: ${(motorActionTimer / 1000).toFixed(2)}s`}
                                            style={{
                                                fontSize: '0.9rem',
                                                color: motorActionActive ? '#22c55e' : '#666',
                                                fontWeight: motorActionActive ? 'bold' : 'normal'
                                            }}
                                        />
                                    </div>

                                    <HorizontalSeparator label="Speed Histogram" fullWidth={true} bleed="1rem" />
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                        <StaticText text="Histogram" />
                                        <Table
                                            width="100%"
                                            height={300}
                                            columnsHeaders={[
                                                { name: 'PWM Multiplier', width: 120, align: 'center', canResize: true },
                                                { name: 'Forward sec.', width: 120, align: 'center', canResize: true },
                                                { name: 'Reverse sec.', width: 120, align: 'center', canResize: true }
                                            ]}
                                            cells={editingMotor.speedHistogram.histogram}
                                            onCellsChange={(newCells) => updateSpeedHistogramField('histogram', newCells)}
                                            canAddRows={true}
                                            canDeleteRows={true}
                                            canAddColumns={false}
                                            canDeleteColumns={false}
                                            maxRows={20}
                                            minRows={2}
                                        />
                                    </div>
                                </ColumnLayout>
                            </>
                        )}
                    </ColumnLayout>
                </ModalWindow>
            )}

            {testMotor && (
                <ModalWindow
                    isOpen={isTestModalOpen}
                    title={`Test Motor: ${testMotor.name}`}
                    okLabel="Close"
                    onOk={handleCloseTestModal}
                >
                    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
                        <Joystick1D
                            orientation="horizontal"
                            width={350}
                            height={60}
                            valueOrigin={0}
                            minValue={-1}
                            maxValue={1}
                            onChange={handleJoystickMove}
                            onStart={handleJoystickStart}
                            onEnd={handleJoystickEnd}
                        />
                    </div>
                </ModalWindow>
            )}
        </div >
    );
};

export default Motors;
