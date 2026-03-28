import { useEffect, useRef } from 'react';

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
    const intervalRef = useRef(null);
    const wasEnabledRef = useRef(false);
    const onEndRef = useRef(onEnd);

    // Keep onEnd ref updated to avoid stale closure issues
    useEffect(() => {
        onEndRef.current = onEnd;
    }, [onEnd]);

    useEffect(() => {
        if (enabled && !wasEnabledRef.current) {
            // Timer just became enabled - trigger onStart
            wasEnabledRef.current = true;
            if (onStart) {
                onStart();
            }

            // Start the interval
            intervalRef.current = setInterval(() => {
                if (onInterval) {
                    onInterval();
                }
            }, interval * 1000);
        } else if (!enabled && wasEnabledRef.current) {
            // Timer just became disabled - trigger onEnd
            wasEnabledRef.current = false;
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            if (onEndRef.current) {
                onEndRef.current();
            }
        }
    }, [enabled, interval, onStart, onInterval]);

    // Cleanup on unmount - trigger onEnd if timer was enabled
    useEffect(() => {
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            if (wasEnabledRef.current && onEndRef.current) {
                onEndRef.current();
            }
        };
    }, []);

    // Renderless component - returns nothing
    return null;
};

export default Timer;

