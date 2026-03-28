/**
 * API Service for handling REST API calls
 * Centralized module for all API communications
 */

// Base URL for API calls - uses current hostname with port 80
const getBaseUrl = () => `http://${window.location.hostname}:80`;

// Default timeout in milliseconds
const DEFAULT_TIMEOUT = 5000;

/**
 * Generic fetch wrapper with timeout support
 * @param {string} endpoint - API endpoint (e.g., '/api/settings/camera')
 * @param {object} options - Fetch options
 * @param {number} timeout - Timeout in milliseconds
 * @returns {Promise<Response>}
 */
const fetchWithTimeout = async (endpoint, options = {}, timeout = DEFAULT_TIMEOUT) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(`${getBaseUrl()}${endpoint}`, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error(`Request timed out after ${timeout}ms`);
        }
        throw error;
    }
};

/**
 * Generic GET request
 * @param {string} endpoint - API endpoint
 * @param {number} timeout - Timeout in milliseconds
 * @returns {Promise<any>}
 */
const get = async (endpoint, timeout = DEFAULT_TIMEOUT) => {
    const response = await fetchWithTimeout(endpoint, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    }, timeout);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `GET ${endpoint} failed with status ${response.status}`);
    }

    return response.json();
};

/**
 * Generic POST request
 * @param {string} endpoint - API endpoint
 * @param {any} data - Data to send
 * @param {number} timeout - Timeout in milliseconds
 * @returns {Promise<any>}
 */
const post = async (endpoint, data, timeout = DEFAULT_TIMEOUT) => {
    const response = await fetchWithTimeout(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    }, timeout);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `POST ${endpoint} failed with status ${response.status}`);
    }

    return response.json();
};



// ============================================
// Motors Settings API
// ============================================

/**
 * Get motors settings from the server
 * @returns {Promise<array>} Array of motor settings
 */
export const getMotorsSettings = async () => {
    return get('/api/settings/motors');
};

/**
 * Save motors settings to the server
 * @param {array} motorsSettings - Array of motor settings to save
 * @returns {Promise<object>} Response from server
 */
export const saveMotorsSettings = async (motorsSettings) => {
    return post('/api/settings/motors', motorsSettings);
};

// ============================================
// General Settings API (if needed in future)
// ============================================

/**
 * Get all settings from the server
 * @returns {Promise<object>} Full settings object
 */
export const getAllSettings = async () => {
    return get('/api/settings');
};

/**
 * Save all settings to the server
 * @param {object} settings - Full settings object to save
 * @returns {Promise<object>} Response from server
 */
export const saveAllSettings = async (settings) => {
    return post('/api/settings', settings);
};

// ============================================
// Motor Action API
// ============================================

/**
 * Start motor action - set J8[pin_index] to pwm_multiplier
 * @param {number} pin_index - Pin index to control
 * @param {number} pwm_multiplier - PWM multiplier value (0-1)
 * @returns {Promise<object>} Response from server
 */
export const startMotorAction = async (pin_index, pwm_multiplier) => {
    return post('/api/motors/action/start', { pin_index, pwm_multiplier });
};

/**
 * Stop motor action - reset J8[pin_index]
 * @param {number} pin_index - Pin index to reset
 * @returns {Promise<object>} Response from server
 */
export const stopMotorAction = async (pin_index) => {
    return post('/api/motors/action/stop', { pin_index });
};

/**
 * Get speed histogram data for Chart2D visualization
 * @returns {Promise<array>} Array of histogram data per motor
 */
export const getSpeedHistogram = async () => {
    return get('/api/motors/speedhistogram');
};

/**
 * Set motor speed
 * @param {string} motorName - Motor name to control
 * @param {number} speed - Speed value (-1 to 1)
 * @returns {Promise<object>} Response from server
 */
export const setMotorSpeed = async (motorName, speed) => {
    return post('/api/motors/speed', { motor_name: motorName, speed });
};

// ============================================
// Hot Zone Settings API
// ============================================

/**
 * Get hot zone settings from the server
 * @returns {Promise<object>} Hot zone settings object
 */
export const getHotZoneSettings = async () => {
    return get('/api/settings/hot-zone');
};

/**
 * Save hot zone settings to the server
 * @param {object} hotZoneSettings - Hot zone settings to save
 * @returns {Promise<object>} Response from server
 */
export const saveHotZoneSettings = async (hotZoneSettings) => {
    return post('/api/settings/hot-zone', hotZoneSettings);
};

// ============================================
// General Settings API
// ============================================

/**
 * Get general settings from the server
 * @returns {Promise<object>} General settings object
 */
export const getGeneralSettings = async () => {
    return get('/api/settings/general');
};

/**
 * Save general settings to the server
 * @param {object} generalSettings - General settings to save
 * @param {number} timeout - Optional timeout in milliseconds
 * @returns {Promise<object>} Response from server
 */
export const saveGeneralSettings = async (generalSettings, timeout) => {
    return post('/api/settings/general', generalSettings, timeout);
};

// ============================================
// AI Setup Settings API
// ============================================

/**
 * Get AI Setup settings from the server
 * @returns {Promise<object>} AI Setup settings object with motors info
 */
export const getAISetupSettings = async () => {
    return get('/api/aisetup');
};

/**
 * Save AI Setup settings to the server
 * @param {object} aiSetupSettings - AI Setup settings to save
 * @returns {Promise<object>} Response from server
 */
export const saveAISetupSettings = async (aiSetupSettings) => {
    return post('/api/aisetup', aiSetupSettings);
};

// ============================================
// System Management API
// ============================================

/**
 * Get system info (date/time, timezone) from the server
 * @returns {Promise<object>} System info object
 */
export const getSystemInfo = async () => {
    return get('/api/system/info');
};

/**
 * Get platform info (OS, hardware, user) from the server
 * @returns {Promise<object>} Platform info object
 */
export const getSystemPlatformInfo = async () => {
    return get('/api/system/platform-info');
};

/**
 * Set system date and time
 * @param {object} dateTime - Object with year, month, day, hour, minute, second
 * @returns {Promise<object>} Response from server
 */
export const setSystemDateTime = async (dateTime) => {
    return post('/api/system/datetime', dateTime, 30000);
};

/**
 * Set system timezone
 * @param {string} timezone - Timezone identifier
 * @returns {Promise<object>} Response from server
 */
export const setSystemTimezone = async (timezone) => {
    return post('/api/system/timezone', { timezone }, 30000);
};

/**
 * Set both system date/time and timezone
 * @param {object} settings - Object with year, month, day, hour, minute, second, timezone
 * @returns {Promise<object>} Response from server
 */
export const setSystemDateTimeAndTimezone = async (settings) => {
    return post('/api/system/datetime-and-timezone', settings, 30000);
};

/**
 * Reboot the system
 * @returns {Promise<object>} Response from server
 */
export const rebootSystem = async () => {
    return post('/api/system/reboot', {}, 30000);
};

// Export default object with all API functions
export default {

    getMotorsSettings,
    saveMotorsSettings,
    startMotorAction,
    stopMotorAction,
    getSpeedHistogram,
    setMotorSpeed,
    getHotZoneSettings,
    saveHotZoneSettings,
    getGeneralSettings,
    saveGeneralSettings,
    getAllSettings,
    saveAllSettings,
    getAISetupSettings,
    saveAISetupSettings,
    getSystemInfo,
    getSystemPlatformInfo,
    setSystemDateTime,
    setSystemTimezone,
    setSystemDateTimeAndTimezone,
    rebootSystem
};
