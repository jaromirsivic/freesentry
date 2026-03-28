import React from 'react';
import Panel from './components/Panel';
import HorizontalSeparator from './components/HorizontalSeparator';

const MainPage = () => {
    return (
        <div className="page-container">
            <Panel title="SubMOA: High-Precision AI Manipulation for Heavy Instrumentation">
                <div className="main-page-content">
                <p className="mb-4">
                    <strong>SubMOA</strong> is the advanced open-source solution for the automated control of heavy payload systems. Designed for scenarios requiring <strong>sub-minute-of-angle (Sub-MOA) accuracy</strong>, it bridges the gap between manual operation and fully autonomous, AI-driven precision.
                </p>
                <p className="mb-4">
                    Whether managing heavy cinematic camera rigs or sensitive telescopic microphones, SubMOA provides unparalleled stability and tracking capabilities for equipment where every millimeter counts.
                </p>

                <HorizontalSeparator label="Mobile First & AI Boosted" fullWidth={true} bleed="1.5rem" />
                <p className="mb-2">
                    The application is built with a <strong>Mobile First</strong> philosophy, ensuring a responsive and intuitive interface on any device. 
                    The system is <strong>AI Boosted</strong>, leveraging advanced machine learning models to enhance precision and automate complex tracking tasks seamlessly.
                </p>

                <HorizontalSeparator label="Next-Generation AI Tracking" fullWidth={true} bleed="1.5rem" />
                <p className="mb-2">SubMOA goes beyond simple motion detection. Utilizing state-of-the-art computer vision, it identifies and tracks specific human body parts in real-time.</p>
                <ul className="list-disc mb-4 space-y-1">
                    <li><strong>Granular Focus:</strong> Target specific anatomical features rather than just a center of mass.</li>
                    <li><strong>Dynamic Acquisition:</strong> Seamlessly track subjects whether they are static or in rapid motion.</li>
                    <li><strong>Fully Local Execution:</strong> All AI processing runs completely on-device. No internet connection is required, ensuring maximum reliability and data privacy in the field.</li>
                    <li><strong>Mission Protocol:</strong> Define complex engagement rules, external triggers, and automated <strong>exit strategies</strong> upon mission completion.</li>
                </ul>

                <HorizontalSeparator label="Advanced Kinematics: Turret & Turretless" fullWidth={true} bleed="1.5rem" />
                <p className="mb-2">The software is engineered to handle complex mechanical setups with heavy, off-center loads.</p>
                <ul className="list-disc mb-4 space-y-1">
                    <li><strong>Turret Systems:</strong> Native support for high-torque pan/tilt units.</li>
                    <li><strong>Turretless & Linear Actuators:</strong> Specialized algorithms for <strong>self-compressing, free-floating setups</strong>. SubMOA dynamically compensates for center-of-gravity shifts in real-time.</li>
                </ul>

                <HorizontalSeparator label="Enterprise-Grade Open Source" fullWidth={true} bleed="1.5rem" />
                <p className="mb-4">
                    SubMOA is professionally maintained open-source software released under the <strong>AGPL-3.0 License</strong>. It is continuously evolved by a dedicated team of developers to ensure compatibility with modern hardware and the latest breakthroughs in AI logic.
                </p>

                <div className="mb-8">
                    <a href="#" className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                        Download SubMOA
                    </a>
                </div>

                <HorizontalSeparator label="Getting Started: Turretless Configuration" fullWidth={true} bleed="1.5rem" />
                <p className="mb-4">Follow these steps to build and configure a high-precision turretless system.                </p>

                <HorizontalSeparator label="1. Initial Software Configuration" fullWidth={true} bleed="1.5rem" />
                <ul className="list-disc mb-4 space-y-1">
                    <li>Launch the application.</li>
                    <li>Navigate to <strong>Menu &gt; Hot Zone</strong>.</li>
                    <li>Configure the geometry settings for your specific <strong>Turretless System</strong>.</li>
                </ul>

                <HorizontalSeparator label="2. Hardware Requirements" fullWidth={true} bleed="1.5rem" />
                <p className="mb-2">Acquire the necessary components for the build:</p>

                <HorizontalSeparator label="Core Components" fullWidth={true} bleed="1.5rem" />
                <ul className="list-disc mb-2 space-y-1">
                    <li><strong>Raspberry Pi 5</strong> (2GB Starter Kit or newer).</li>
                    <li><strong>Camera Module</strong> (Official Pi Camera 3 recommended).</li>
                    <li><strong>Motor Driver:</strong> BTS7960 (also known as HW-039 or IBT_2).</li>
                    <li><strong>Wiring:</strong> Dupont Wires (40x 20cm).</li>
                    <li><strong>Linear Actuator:</strong> 12V, 1000+ N force.
                        <ul className="list-[circle] mt-1 space-y-1">
                            <li><em>Stroke Size:</em> Typically 20cm (8") to 45cm (18") depending on needs.</li>
                            <li><em>Speed:</em> 5mm/s - 10mm/s recommended (slower speeds provide higher precision).</li>
                        </ul>
                    </li>
                    <li><strong>Rigging:</strong> Kevlar line capable of supporting at least 80 kg.</li>
                </ul>

                <HorizontalSeparator label="Optional & Assembly Tools" fullWidth={true} bleed="1.5rem" />
                <ul className="list-disc mb-4 space-y-1">
                    <li><strong>Teflon Tubing:</strong> ID 1.75mm or 2mm / OD 3mm or 4mm (for smooth line guidance).</li>
                    <li><strong>Power Supply:</strong> 12V 10A PSU, battery pack, or car battery.</li>
                    <li><strong>Power Cables:</strong> Twin-core copper wire (min. 1mm² cross-section).</li>
                    <li><strong>Electrical Terminal Blocks.</strong></li>
                    <li><strong>Screwdriver, Zip Ties, Duct Tape.</strong></li>
                </ul>

                <HorizontalSeparator label="3. Motor Configuration" fullWidth={true} bleed="1.5rem" />
                <ul className="list-disc mb-4 space-y-1">
                    <li>Connect your motor drivers.</li>
                    <li>Navigate to <strong>Menu &gt; Setup &gt; Motors</strong> to assign and calibrate the motors.</li>
                </ul>

                <HorizontalSeparator label="4. Camera Setup" fullWidth={true} bleed="1.5rem" />
                <ul className="list-disc mb-4 space-y-1">
                    <li>Connect the camera module.</li>
                    <li>Navigate to <strong>Menu &gt; Setup &gt; Cameras</strong> to initialize the video feed and adjust optical settings.</li>
                </ul>

                <HorizontalSeparator label="5. AI & Tracking Setup" fullWidth={true} bleed="1.5rem" />
                <ul className="list-disc mb-4 space-y-1">
                    <li>Navigate to the AI settings menu to select detection models and configure tracking parameters.</li>
                </ul>
                </div>
            </Panel>
        </div>
    );
};

export default MainPage;
