/**
 * API Service for handling REST API calls
 * Centralized module for all API communications
 */

// Default timeout in milliseconds
const DEFAULT_TIMEOUT = 5000;
const ABSOLUTE_URL_PATTERN = /^[a-zA-Z][a-zA-Z\d+\-.]*:/;
const CONFIGURED_API_BASE_URL = (import.meta.env.VITE_API_BASE_URL?.trim() ?? '').replace(/\/+$/, '');

const isAbsoluteUrl = (url) => typeof url === 'string' && (ABSOLUTE_URL_PATTERN.test(url) || url.startsWith('//'));

/**
 * Resolve API URLs safely across same-origin, reverse-proxy and explicit
 * cross-origin deployments.
 *
 * By default we keep root-relative `/api/...` URLs untouched so the browser
 * resolves them against the current origin. If `VITE_API_BASE_URL` is defined,
 * we prefix relative API paths with that configured base instead.
 *
 * @param {string|URL} url - API endpoint or absolute URL
 * @returns {string|URL}
 */
const resolveApiUrl = (url) => {
    if (typeof url !== 'string' || url.length === 0 || isAbsoluteUrl(url)) {
        return url;
    }

    if (!CONFIGURED_API_BASE_URL) {
        return url;
    }

    if (url.startsWith('/')) {
        return `${CONFIGURED_API_BASE_URL}${url}`;
    }

    return `${CONFIGURED_API_BASE_URL}/${url.replace(/^\/+/, '')}`;
};

const createErrorWithMetadata = (message, metadata = {}) => {
    const error = new Error(message);
    Object.assign(error, metadata);
    return error;
};

const createTimeoutError = (endpoint, timeout) => createErrorWithMetadata(
    `Request timed out after ${timeout}ms`,
    {
        name: 'ApiTimeoutError',
        endpoint,
        timeout
    }
);

const createHttpError = (method, endpoint, status, detail) => createErrorWithMetadata(
    detail || `${method} ${endpoint} failed with status ${status}`,
    {
        name: 'ApiHttpError',
        method,
        endpoint,
        status
    }
);

const isTimeoutError = (error) => error?.name === 'ApiTimeoutError';
const isHttpError = (error) => error?.name === 'ApiHttpError';
const isAbortError = (error) => error?.name === 'AbortError';
const isNetworkError = (error) => error instanceof TypeError;

const isGenericRequestMessage = (message) => {
    if (typeof message !== 'string') {
        return false;
    }

    const trimmedMessage = message.trim();
    return /^Request timed out after \d+ms$/i.test(trimmedMessage)
        || /^(GET|POST)\s+\S+\s+failed with status \d+$/i.test(trimmedMessage)
        || /^Request failed \(\d+\)$/i.test(trimmedMessage);
};

const appendOperationDetail = (baseMessage, detail) => {
    if (typeof detail !== 'string') {
        return baseMessage;
    }

    const trimmedDetail = detail.trim();
    if (!trimmedDetail || isGenericRequestMessage(trimmedDetail)) {
        return baseMessage;
    }

    if (trimmedDetail.toLowerCase().startsWith(baseMessage.toLowerCase())) {
        return trimmedDetail;
    }

    return `${baseMessage} ${trimmedDetail}`;
};

const createLongOperationError = (message, metadata = {}) => createErrorWithMetadata(
    message,
    {
        name: 'ApiLongOperationError',
        userMessage: message,
        ...metadata
    }
);

const normalizeLongOperationError = (error, config) => {
    if (error?.name === 'ApiLongOperationError') {
        return error;
    }

    const {
        endpoint,
        timeoutMessage,
        networkMessage,
        httpBaseMessage,
        defaultMessage,
        abortMessage,
        retryable = {},
        ambiguous = {}
    } = config;

    if (isTimeoutError(error)) {
        return createLongOperationError(timeoutMessage, {
            kind: 'timeout',
            endpoint,
            timeout: error.timeout,
            retryable: retryable.timeout ?? false,
            ambiguous: ambiguous.timeout ?? false,
            cause: error
        });
    }

    if (isAbortError(error)) {
        return createLongOperationError(abortMessage, {
            kind: 'abort',
            endpoint,
            retryable: retryable.abort ?? false,
            ambiguous: false,
            cause: error
        });
    }

    if (isNetworkError(error)) {
        return createLongOperationError(networkMessage, {
            kind: 'network',
            endpoint,
            retryable: retryable.network ?? false,
            ambiguous: ambiguous.network ?? false,
            cause: error
        });
    }

    if (isHttpError(error)) {
        return createLongOperationError(
            appendOperationDetail(httpBaseMessage, error.message),
            {
                kind: 'http',
                endpoint,
                status: error.status,
                retryable: retryable.http ?? false,
                ambiguous: ambiguous.http ?? false,
                cause: error
            }
        );
    }

    return createLongOperationError(
        appendOperationDetail(defaultMessage, error?.message),
        {
            kind: 'unknown',
            endpoint,
            retryable: retryable.unknown ?? false,
            ambiguous: ambiguous.unknown ?? false,
            cause: error
        }
    );
};

const postLongOperation = async (endpoint, data, timeout, config) => {
    try {
        return await post(endpoint, data, timeout);
    } catch (error) {
        throw normalizeLongOperationError(error, { endpoint, ...config });
    }
};

const LONG_OPERATION_CONFIG = {
    saveGeneralSettings: {
        timeoutMessage: 'Saving settings timed out. The controller may still be applying the new configuration. Verify the controller status before retrying.',
        networkMessage: 'Saving settings failed because the device could not be reached. Verify the current settings before retrying.',
        httpBaseMessage: 'Saving settings failed.',
        defaultMessage: 'Saving settings failed.',
        abortMessage: 'Saving settings was cancelled.',
        retryable: {
            http: true
        },
        ambiguous: {
            timeout: true
        }
    },
    setSystemDateTimeAndTimezone: {
        timeoutMessage: 'Updating system date and time timed out. The device may still be applying the change. Verify the displayed date and time before retrying.',
        networkMessage: 'Updating system date and time failed because the server could not be reached.',
        httpBaseMessage: 'Updating system date and time failed.',
        defaultMessage: 'Updating system date and time failed.',
        abortMessage: 'Updating system date and time was cancelled.',
        retryable: {
            http: true
        },
        ambiguous: {
            timeout: true
        }
    },
    rebootSystem: {
        timeoutMessage: 'Reboot request timed out. The device may already be restarting. Wait for it to reconnect before trying again.',
        networkMessage: 'Connection was lost while sending the reboot request. The device may already be restarting. Wait for it to reconnect before trying again.',
        httpBaseMessage: 'Reboot could not be started.',
        defaultMessage: 'Reboot could not be confirmed.',
        abortMessage: 'Reboot request was cancelled.',
        retryable: {
            http: true
        },
        ambiguous: {
            timeout: true,
            network: true
        }
    }
};

/**
 * Generic fetch wrapper with timeout support
 * @param {string} endpoint - API endpoint (e.g., '/api/settings/camera')
 * @param {object} options - Fetch options
 * @param {number} timeout - Timeout in milliseconds
 * @returns {Promise<Response>}
 */
const fetchWithTimeout = async (endpoint, options = {}, timeout = DEFAULT_TIMEOUT) => {
    const controller = new AbortController();
    const externalSignal = options.signal;
    let didTimeout = false;

    const handleExternalAbort = () => {
        controller.abort();
    };

    if (externalSignal) {
        if (externalSignal.aborted) {
            controller.abort();
        } else {
            externalSignal.addEventListener('abort', handleExternalAbort, { once: true });
        }
    }

    const timeoutId = setTimeout(() => {
        didTimeout = true;
        controller.abort();
    }, timeout);

    try {
        const response = await fetch(resolveApiUrl(endpoint), {
            ...options,
            signal: controller.signal
        });
        return response;
    } catch (error) {
        if (didTimeout) {
            throw createTimeoutError(endpoint, timeout);
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
        externalSignal?.removeEventListener('abort', handleExternalAbort);
    }
};

/**
 * Generic GET request
 * @param {string} endpoint - API endpoint
 * @param {number} timeout - Timeout in milliseconds
 * @param {object} options - Fetch options
 * @returns {Promise<any>}
 */
const get = async (endpoint, timeout = DEFAULT_TIMEOUT, options = {}) => {
    const response = await fetchWithTimeout(endpoint, {
        ...options,
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        }
    }, timeout);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw createHttpError('GET', endpoint, response.status, errorData.detail || `GET ${endpoint} failed with status ${response.status}`);
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
        throw createHttpError('POST', endpoint, response.status, errorData.detail || `POST ${endpoint} failed with status ${response.status}`);
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
 * @param {number|object} timeoutOrOptions - Timeout in milliseconds or fetch options
 * @returns {Promise<object>} Hot zone settings object
 */
export const getHotZoneSettings = async (timeoutOrOptions = DEFAULT_TIMEOUT) => {
    if (typeof timeoutOrOptions === 'number') {
        return get('/api/settings/hot-zone', timeoutOrOptions);
    }

    const { timeout = DEFAULT_TIMEOUT, signal } = timeoutOrOptions ?? {};
    return get('/api/settings/hot-zone', timeout, { signal });
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
    return postLongOperation('/api/settings/general', generalSettings, timeout, LONG_OPERATION_CONFIG.saveGeneralSettings);
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
// AI Agent Runtime Activation API
// ============================================

/**
 * Get runtime AI Agent activation state.
 * @returns {Promise<object>} Activation state response.
 */
export const getAIAgentActivation = async () => {
    return get('/api/aiagent/activation');
};

/**
 * Fully activate the AI Agent runtime motor control.
 * @returns {Promise<object>} Activation state response.
 */
export const activateAIAgent = async () => {
    return post('/api/aiagent/activate', {});
};

/**
 * Stop runtime AI Agent motor control.
 * @returns {Promise<object>} Activation state response.
 */
export const stopAIAgent = async () => {
    return post('/api/aiagent/stop', {});
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
    return postLongOperation('/api/system/datetime-and-timezone', settings, 30000, LONG_OPERATION_CONFIG.setSystemDateTimeAndTimezone);
};

/**
 * Reboot the system
 * @returns {Promise<object>} Response from server
 */
export const rebootSystem = async () => {
    return postLongOperation('/api/system/reboot', {}, 30000, LONG_OPERATION_CONFIG.rebootSystem);
};

// ============================================
// New-style authenticated fetch helpers
// ============================================

const TOKEN_KEY = 'token';

function authHeaders() {
    const token = localStorage.getItem(TOKEN_KEY);
    return token ? { Authorization: `Bearer ${token}` } : {};
}

function notifyAuthExpired(status) {
    if (status === 401 || status === 403) {
        window.dispatchEvent(new CustomEvent('token:auth-expired'));
    }
}

/** POST JSON body, return parsed JSON response. Throws on non-2xx. */
export async function apiFetch(url, body) {
    const resp = await fetch(resolveApiUrl(url), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        notifyAuthExpired(resp.status);
        const data = await resp.json().catch(() => ({}));
        const detail = data.detail;
        const message = Array.isArray(detail)
            ? detail.map(e => `${e.loc?.slice(-1)?.[0] ?? 'field'}: ${e.msg}`).join('; ')
            : (detail ?? `Request failed (${resp.status})`);
        throw new Error(message);
    }
    return resp.json();
}

/** GET plain text response. Throws on non-2xx. */
export async function apiText(url, { headers = {}, signal } = {}) {
    const resp = await fetch(resolveApiUrl(url), {
        method: 'GET',
        headers: { ...authHeaders(), ...headers },
        signal,
    });
    if (!resp.ok) {
        notifyAuthExpired(resp.status);
        const contentType = resp.headers.get('content-type') ?? '';
        let message = '';
        if (contentType.includes('application/json')) {
            const data = await resp.json().catch(() => ({}));
            const detail = data.detail;
            message = Array.isArray(detail)
                ? detail.map(e => `${e.loc?.slice(-1)?.[0] ?? 'field'}: ${e.msg}`).join('; ')
                : (detail ?? '');
        } else {
            message = (await resp.text().catch(() => '')).trim();
        }
        throw new Error(message || `Request failed (${resp.status})`);
    }
    return resp.text();
}

/**
 * POST FormData with XHR so upload progress can be tracked.
 * Options: onProgress(percent), signal (AbortSignal)
 */
export function apiUploadWithProgress(url, formData, { onProgress, signal } = {}) {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) {
            reject(new DOMException('Upload cancelled', 'AbortError'));
            return;
        }
        const xhr = new XMLHttpRequest();
        signal?.addEventListener('abort', () => xhr.abort());
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && onProgress) {
                onProgress(Math.round((e.loaded / e.total) * 100));
            }
        };
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try { resolve(JSON.parse(xhr.responseText)); }
                catch { resolve({}); }
            } else {
                notifyAuthExpired(xhr.status);
                let message;
                try {
                    const data = JSON.parse(xhr.responseText);
                    const detail = data.detail;
                    message = Array.isArray(detail)
                        ? detail.map(e => `${e.loc?.slice(-1)?.[0] ?? 'field'}: ${e.msg}`).join('; ')
                        : (detail ?? `Upload failed (${xhr.status})`);
                } catch { message = `Upload failed (${xhr.status})`; }
                reject(new Error(message));
            }
        };
        xhr.onerror = () => reject(new Error('Network error during upload'));
        xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'));
        const token = localStorage.getItem(TOKEN_KEY);
        xhr.open('POST', resolveApiUrl(url));
        if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        xhr.send(formData);
    });
}

/** POST FormData (file upload). Do NOT set Content-Type – browser does it. */
export async function apiUpload(url, formData) {
    const resp = await fetch(resolveApiUrl(url), {
        method: 'POST',
        headers: authHeaders(),
        body: formData,
    });
    if (!resp.ok) {
        notifyAuthExpired(resp.status);
        const data = await resp.json().catch(() => ({}));
        const detail = data.detail;
        const message = Array.isArray(detail)
            ? detail.map(e => `${e.loc?.slice(-1)?.[0] ?? 'field'}: ${e.msg}`).join('; ')
            : (detail ?? `Upload failed (${resp.status})`);
        throw new Error(message);
    }
    return resp.json();
}

/** POST JSON body, trigger a browser file download from the response blob. */
export async function apiDownload(url, body, filename) {
    const resp = await fetch(resolveApiUrl(url), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        notifyAuthExpired(resp.status);
        throw new Error(`Download failed (${resp.status})`);
    }
    const blob = await resp.blob();
    const href = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = href;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(href);
}

/** POST JSON body, return a temporary object URL for the response blob. */
export async function apiBlob(url, body) {
    const resp = await fetch(resolveApiUrl(url), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        notifyAuthExpired(resp.status);
        throw new Error(`Fetch failed (${resp.status})`);
    }
    return URL.createObjectURL(await resp.blob());
}

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
    getAIAgentActivation,
    activateAIAgent,
    stopAIAgent,
    getSystemInfo,
    getSystemPlatformInfo,
    setSystemDateTime,
    setSystemTimezone,
    setSystemDateTimeAndTimezone,
    rebootSystem
};
