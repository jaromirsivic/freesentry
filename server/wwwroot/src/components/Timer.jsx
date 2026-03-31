import { useCallback, useEffect, useLayoutEffect, useRef } from 'react';

/**
 * Timer - An invisible/renderless timer component.
 * 
 * @param {Object} props
 * @param {boolean} props.enabled - Whether the timer is active and can trigger events.
 * @param {number} [props.interval=1] - The interval in seconds between onInterval calls. Default is 1.
 * @param {function} [props.onStart] - Called when enabled changes from false to true.
 * @param {function} [props.onInterval] - Called every time the interval elapses while enabled.
 * @param {function} [props.onEnd] - Called when enabled changes from true to false, or when component unmounts while enabled.
 * 
 * @example
 * <Timer
 *     enabled={isTimerActive}
 *     interval={2}
 *     onStart={() => console.log('Timer started')}
 *     onInterval={() => console.log('Tick')}
 *     onEnd={() => console.log('Timer ended')}
 * />
 */
const Timer = ({
    enabled = false,
    interval = 1,
    onStart,
    onInterval,
    onEnd
}) => {
    const timeoutRef = useRef(null);
    const loopVersionRef = useRef(0);
    const wasEnabledRef = useRef(false);
    const isTickRunningRef = useRef(false);
    const pendingRestartRef = useRef(null);
    const onStartRef = useRef(onStart);
    const onIntervalRef = useRef(onInterval);
    const onEndRef = useRef(onEnd);
    const intervalValueRef = useRef(interval);
    const runTickRef = useRef(null);

    useLayoutEffect(() => {
        onStartRef.current = onStart;
    }, [onStart]);

    useLayoutEffect(() => {
        onIntervalRef.current = onInterval;
    }, [onInterval]);

    useLayoutEffect(() => {
        onEndRef.current = onEnd;
    }, [onEnd]);

    useLayoutEffect(() => {
        intervalValueRef.current = interval;
    }, [interval]);

    const clearScheduledTick = useCallback(() => {
        if (timeoutRef.current !== null) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }
    }, []);

    const getIntervalMs = useCallback(() => Math.max(0, intervalValueRef.current * 1000), []);

    const scheduleNextTick = useCallback((version, delayMs) => {
        clearScheduledTick();
        timeoutRef.current = setTimeout(() => {
            timeoutRef.current = null;
            void runTickRef.current?.(version);
        }, Math.max(0, delayMs));
    }, [clearScheduledTick]);

    const runTick = useCallback(async (version) => {
        if (!wasEnabledRef.current || loopVersionRef.current !== version || isTickRunningRef.current) {
            return;
        }

        isTickRunningRef.current = true;
        const startedAt = performance.now();

        try {
            await onIntervalRef.current?.();
        } catch (error) {
            console.error('Timer interval callback failed:', error);
        }

        isTickRunningRef.current = false;

        if (!wasEnabledRef.current) {
            return;
        }

        const pendingRestart = pendingRestartRef.current;
        if (pendingRestart && pendingRestart.version === loopVersionRef.current) {
            pendingRestartRef.current = null;

            const nextDelayMs = pendingRestart.mode === 'fresh'
                ? getIntervalMs()
                : Math.max(0, getIntervalMs() - (performance.now() - startedAt));

            scheduleNextTick(pendingRestart.version, nextDelayMs);
            return;
        }

        if (loopVersionRef.current !== version) {
            return;
        }

        const elapsedMs = performance.now() - startedAt;
        scheduleNextTick(version, Math.max(0, getIntervalMs() - elapsedMs));
    }, [getIntervalMs, scheduleNextTick]);

    useLayoutEffect(() => {
        runTickRef.current = runTick;
    }, [runTick]);

    useEffect(() => {
        if (!enabled) {
            clearScheduledTick();
            pendingRestartRef.current = null;

            if (wasEnabledRef.current) {
                loopVersionRef.current += 1;
                wasEnabledRef.current = false;
                onEndRef.current?.();
            }

            return;
        }

        if (!wasEnabledRef.current) {
            const version = loopVersionRef.current + 1;
            loopVersionRef.current = version;
            wasEnabledRef.current = true;
            onStartRef.current?.();

            if (isTickRunningRef.current) {
                pendingRestartRef.current = { version, mode: 'fresh' };
            } else {
                scheduleNextTick(version, getIntervalMs());
            }

            return;
        }

        const version = loopVersionRef.current + 1;
        loopVersionRef.current = version;

        if (isTickRunningRef.current) {
            pendingRestartRef.current = { version, mode: 'steady' };
        } else {
            pendingRestartRef.current = null;
            scheduleNextTick(version, getIntervalMs());
        }
    }, [clearScheduledTick, enabled, getIntervalMs, interval, scheduleNextTick]);

    // Cleanup on unmount - trigger onEnd if timer was enabled
    useEffect(() => {
        return () => {
            clearScheduledTick();
            pendingRestartRef.current = null;
            loopVersionRef.current += 1;

            if (wasEnabledRef.current) {
                wasEnabledRef.current = false;
                onEndRef.current?.();
            }
        };
    }, [clearScheduledTick]);

    // Renderless component - returns nothing
    return null;
};

export default Timer;

