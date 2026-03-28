# SubMOA: High-Precision AI Manipulation for Heavy Instrumentation

**SubMOA** is the advanced open-source solution for the automated control of heavy payload systems. Designed for scenarios requiring **sub-minute-of-angle (Sub-MOA) accuracy**, it bridges the gap between manual operation and fully autonomous, AI-driven precision.

Whether managing heavy cinematic camera rigs or sensitive telescopic microphones, SubMOA provides unparalleled stability and tracking capabilities for equipment where every millimeter counts.

---

## Next-Generation AI Tracking
SubMOA goes beyond simple motion detection. Utilizing state-of-the-art computer vision, it identifies and tracks specific human body parts in real-time.
*   **Granular Focus:** Target specific anatomical features rather than just a center of mass.
*   **Dynamic Acquisition:** Seamlessly track subjects whether they are static or in rapid motion.
*   **Fully Local Execution:** All AI processing runs completely on-device. No internet connection is required, ensuring maximum reliability and data privacy in the field.
*   **Mission Protocol:** Define complex engagement rules, external triggers, and automated **exit strategies** upon mission completion.

## Advanced Kinematics: Turret & Turretless
The software is engineered to handle complex mechanical setups with heavy, off-center loads.
*   **Turret Systems:** Native support for high-torque pan/tilt units.
*   **Turretless & Linear Actuators:** Specialized algorithms for **self-compressing, free-floating setups**. SubMOA dynamically compensates for center-of-gravity shifts in real-time.

## Enterprise-Grade Open Source
SubMOA is professionally maintained open-source software released under the **AGPL-3.0 License**. It is continuously evolved by a dedicated team of developers to ensure compatibility with modern hardware and the latest breakthroughs in AI logic.

[**Download SubMOA**](#)

---

## Getting Started: Turretless Configuration

Follow these steps to build and configure a high-precision turretless system.

### 1. Initial Software Configuration
*   Launch the application.
*   Navigate to **Menu > Hot Zone**.
*   Configure the geometry settings for your specific **Turretless System**.

### 2. Hardware Requirements
Acquire the necessary components for the build:

**Core Components:**
*   **Raspberry Pi 5** (2GB Starter Kit or newer).
*   **Camera Module** (Official Pi Camera 3 recommended).
*   **Motor Driver:** BTS7960 (also known as HW-039 or IBT_2).
*   **Wiring:** Dupont Wires (40x 20cm).
*   **Linear Actuator:** 12V, 1000+ N force.
    *   *Stroke Size:* Typically 20cm (8") to 45cm (18") depending on needs.
    *   *Speed:* 5mm/s - 10mm/s recommended (slower speeds provide higher precision).
*   **Rigging:** Kevlar line capable of supporting at least 80 kg.

**Optional & Assembly Tools:**
*   **Teflon Tubing:** ID 1.75mm or 2mm / OD 3mm or 4mm (for smooth line guidance).
*   **Power Supply:** 12V 10A PSU, battery pack, or car battery.
*   **Power Cables:** Twin-core copper wire (min. 1mm² cross-section).
*   **Electrical Terminal Blocks.**
*   **Screwdriver, Zip Ties, Duct Tape.**

### 3. Motor Configuration
*   Connect your motor drivers.
*   Navigate to **Menu > Setup > Motors** to assign and calibrate the motors.

### 4. Camera Setup
*   Connect the camera module.
*   Navigate to **Menu > Setup > Cameras** to initialize the video feed and adjust optical settings.

### 5. AI & Tracking Setup
*   Navigate to the AI settings menu to select detection models and configure tracking parameters.
