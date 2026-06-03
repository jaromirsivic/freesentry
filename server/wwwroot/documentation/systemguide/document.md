<a id="big-picture"></a>
# Big Picture

This system combines camera streams, motor control, geometry setup, and an optional AI-driven operating mode into one browser interface. A typical workflow starts with configuring the hardware and geometry, continues with validating camera and motor behavior, and ends with either manual operation or AI-assisted operation.

The main navigation is available from the menu button in the page header. When AI operation is fully active, the menu is replaced by a red **Stop AI** button so the operator can stop autonomous operation from anywhere in the interface. Full-screen camera pages can also pause or restart their streams while the navigation menu is open.

Most setup pages use panels that summarize current values. When a panel has an **Edit** button, pressing it opens a modal window where values can be changed. **Save** applies the changes and closes the modal. **Cancel** closes the modal without saving. Some pages also include **Apply**, **Reset**, or confirmation dialogs; those actions are described in the relevant sections below.

<a id="main-page"></a>
# Main Page

The Main Page is the starting point for the interface. It presents a short product overview, introduces the system capabilities, and gives the user a high-level path through setup and operation.

The page contains static onboarding content. It may include feature summaries, hardware guidance, and setup steps. A **Download** link is shown as part of the introductory content; in the current interface it is a placeholder link and does not start an application-specific download.

<a id="manual-control"></a>
# Manual Control

The Manual Control page is used for direct operator control. It displays a live camera stream and overlays controls for enabled motors. This page is intended for inspection, calibration, and manual movement, not for autonomous AI operation.

The camera view can show the selected camera feed with a reticle overlay. If motors are enabled, one-dimensional joystick controls appear as overlays. A central two-dimensional control can be used for the first two motor axes. The video view supports panning and zooming where the camera viewport component allows it.

The **Fullscreen** button switches the browser into or out of full-screen mode. The **Setup** button opens the Manual Control Settings modal.

<a id="manual-control-settings"></a>
## Manual Control Settings

This modal controls which camera and motors are used during manual operation. Pressing **Save** stores the session settings and restarts the stream with the new configuration. Pressing **Cancel** discards unsaved changes and returns to the camera view.

<a id="manual-control-settings-cameras"></a>
### Cameras

The camera selector chooses the active camera for the Manual Control page. The available choices are the scope camera and the configured spotter cameras.

**HTTP Stream Quality** controls the quality level used by the live stream. Higher quality can make the image clearer, while lower quality can reduce bandwidth and processing load.

**Display Mode** controls what type of image is requested from the selected camera. Raw mode shows the normal camera image. Mask mode shows the configured mask view. AI mode shows the image variant used for AI-related visualization.

The **Reset AI Engagements** button clears the current AI engagement counters. This action is useful when testing or starting a new operating session.

<a id="manual-control-settings-motors"></a>
### Motor Devices

Each configured motor can be enabled or disabled for manual control. Disabled motors do not show an active manual control overlay.

The **Mode** field chooses how the motor is represented on the Manual Control page. Joystick mode uses a joystick-style control. Slider mode uses a slider-style control where supported by the interface.

The motor name is shown so the operator can identify which physical device is being controlled.

<a id="ai-agent"></a>
# AI Agent

The AI Agent page is used to start and monitor autonomous AI operation. This mode can control real hardware, so the page starts with a confirmation flow before activation.

Before activation, the page shows an activation modal. After activation, the page displays the live scope camera stream in AI display mode with the configured reticle. The **Fullscreen** button switches the camera view into or out of browser full-screen mode.

When AI operation is active, the header shows **Stop AI**. Pressing **Stop AI** deactivates the AI agent and returns the interface to the normal navigation state.

<a id="ai-agent-activation"></a>
## AI Agent Activation

The activation modal explains that this mode starts real AI operation rather than manual simulation.

<a id="ai-agent-activation-actions"></a>
### Actions

Press **Start AI** to reset engagement counters, activate the AI agent, and open the live AI camera view. Once activation completes, the system remains on the AI Agent page.

Press **Cancel** to close the activation modal and return to the Main Page without starting AI operation.

<a id="settings-import-export"></a>
# Import / Export

The Import / Export page is used to back up, restore, or reset system settings. It is part of the Settings section.

Press **Import from file** to choose a JSON settings file from the local computer. The browser asks for confirmation before importing because the action overwrites current settings.

Press **Export Settings** to download the current settings as a `settings.json` file. This file can be stored as a backup or moved to another device with the same expected configuration format.

Press **Restore Defaults** to restore factory default settings. The browser asks for confirmation before the reset is performed.

<a id="settings-general-setup"></a>
# General Setup

The General Setup page contains system information, WiFi configuration, controller configuration, and reboot controls.

The **System** panel shows operating system and hardware information, the current user, CPU and memory information, the current date and time, and the configured time zone. Press **Refresh** to reload the displayed system and platform information. Press **Edit** to open the System Date and Time modal. Press **Reboot** to open a reboot confirmation modal.

The **Wifi** panel shows the current WiFi mode and saved values for Access Point and Client configurations. Press **Edit** to open the Edit Wifi modal.

The **Controller Setup** panel shows whether control is local or remote and displays remote host, remote port, and initialization status. The edit control is present but disabled in the current interface.

<a id="system-date-and-time"></a>
## System Date and Time

This modal changes the system clock and time zone.

<a id="system-date-and-time-fields"></a>
### Date, Time, and Time Zone

The date and time picker sets the device date and time. Use it when the device clock is wrong or when an offline device must be adjusted manually.

The **Time Zone** field selects the local time zone used by the device. Changing it affects how local dates and times are displayed and interpreted.

Press **Save** to apply the new system time settings. Press **Cancel** to close the modal without changing them.

<a id="edit-wifi"></a>
## Edit Wifi

This modal configures wireless networking. After saving, the interface may display an information message explaining that changes apply on supported hardware and may require a reboot.

<a id="edit-wifi-mode"></a>
### Mode

The **Mode** field chooses whether WiFi is disabled, configured as an Access Point, or configured as a Client.

Disabled mode turns off the configured WiFi role. Access Point mode lets the device create its own wireless network. Client mode lets the device join an existing wireless network.

<a id="edit-wifi-access-point"></a>
### Access Point

**Connection Name** identifies the saved access point connection.

**SSID** is the network name that the device broadcasts in Access Point mode.

**Security** shows the wireless security type. In the current interface this is fixed to WPA-PSK.

**Password** is the access point password. It must be long enough for the selected security mode.

<a id="edit-wifi-client"></a>
### Client

**Connection Name** identifies the saved client connection.

**SSID** is the name of the wireless network the device should join.

**Security** shows the wireless security type used by the client connection.

**Password** is the password for the selected wireless network.

<a id="reboot"></a>
## Reboot

The Reboot modal confirms that the device should restart. Confirming the modal starts the reboot action. Canceling leaves the system running.

<a id="settings-motors"></a>
# Motors

The Motors page configures the linear actuators or motor channels used by the system. Each motor appears in its own panel. A disabled motor is visually de-emphasized.

Each motor panel can show the motor name, inertia value, stroke length, pin assignments, PWM frequency, duty-cycle settings, and forward or reverse speed histogram data. Press **Edit** to open the Edit Motor modal for that motor. Press **Test Motor** to open a live motor test modal.

<a id="edit-motor"></a>
## Edit Motor

The Edit Motor modal changes the selected motor configuration. Press **Save** to store the new configuration. Press **Cancel** to discard changes.

<a id="edit-motor-general"></a>
### General

**Name** identifies the motor in panels and control overlays. For general-purpose motor roles, the name can be edited. For fixed roles, the name may be read-only.

**Inertia** describes how much movement continues after a command changes. This value helps the system account for physical response.

**Stroke Length** defines the physical travel range of the actuator, usually in centimeters.

<a id="edit-motor-pins"></a>
### Pins

The pin fields assign hardware pins for a BTS7960B-style motor driver.

**Forward** and **Reverse** select the direction control pins. **Forward Enable** and **Reverse Enable** select the corresponding enable or PWM pins.

**PWM Frequency** controls the PWM frequency used for motor drive output.

When available, **Display Pin Diagram** shows a visual pin diagram so the user can compare the selected values with the physical connector.

<a id="edit-motor-duty-cycle"></a>
### Duty Cycle

The advanced duty-cycle section limits how long the motor can run and how long it must rest.

**Enabled** turns duty-cycle protection on or off for the motor.

**Max Run Time** limits continuous motor activity.

**Min Rest Time** defines the required rest time after the motor reaches its run limit.

<a id="edit-motor-histogram-quick-test"></a>
### Histogram Quick Test

The histogram quick test helps collect or verify motor speed data.

**PWM Multiplier** chooses the output strength used during the quick test.

**Motor Action** chooses Reverse, Stop, or Forward for the timed test action.

The timer controls how long the quick test runs.

The **Speed Histogram** table records measured forward and reverse timing data for selected PWM multipliers. Editing this table changes the values used by the system to estimate motor behavior.

<a id="test-motor"></a>
## Test Motor

The Test Motor modal provides a horizontal joystick for live motor testing. Moving the joystick sends a motor speed command. Negative values drive one direction, positive values drive the opposite direction, and the center position stops movement.

Press **Close** to exit the test. Closing the modal stops the motor.

<a id="settings-cameras"></a>
# Cameras

The Cameras page configures the scope camera and the spotter cameras. Each camera has a panel showing its device index, resolution, and frame rate.

Press **Edit** on a camera panel to open the Edit Camera modal with a live preview. Press **Reset all** to open a confirmation dialog that resets all camera configurations to their defaults.

<a id="edit-camera"></a>
## Edit Camera

The Edit Camera modal changes one camera configuration. The preview area helps the user verify the current image, crop, mask, reticle, and display mode.

Press **Apply** to apply changes without closing the modal. Press **Save** to store changes and close the modal. Press **Close** to leave the modal. If unsaved changes exist, the interface asks whether to discard them.

<a id="edit-camera-general-setup"></a>
### General Setup

**Input Device** chooses the camera device index.

**Resolution** selects the camera capture resolution.

**FPS** selects the requested frame rate.

**Reset Camera** opens a confirmation dialog for resetting only the current camera.

<a id="edit-camera-flip-and-rotate"></a>
### Flip and Rotate

**Flip H** mirrors the image horizontally.

**Flip V** mirrors the image vertically.

**Rotate** rotates the image by the selected angle. This is useful when a camera is physically mounted in a different orientation.

<a id="edit-camera-crop"></a>
### Crop

The crop sliders remove part of the image before it is displayed or processed.

**Top**, **Left**, **Bottom**, and **Right** define the percentage removed from each side.

<a id="edit-camera-stretch-after-crop"></a>
### Stretch after Crop

**Stretch Enabled** controls whether the cropped image is resized to a configured output size.

**Width** and **Height** define the stretched image dimensions.

<a id="edit-camera-image-settings"></a>
### Image Settings

Brightness, contrast, hue, saturation, sharpness, gamma, gain, and backlight sliders tune the image produced by the camera.

These values affect the preview and can also affect the quality of manual viewing, masking, and AI detection.

<a id="edit-camera-controls"></a>
### Controls

**Auto WB** enables or disables automatic white balance.

**White Balance Temperature** sets the color temperature when manual white balance is used.

**Auto Focus** enables or disables automatic focus.

**Focus** sets the manual focus value when automatic focus is disabled.

**Auto Exposure** enables or disables automatic exposure.

**Exposure** sets the manual exposure value when automatic exposure is disabled.

<a id="edit-camera-static-reticle"></a>
### Static Reticle

**X Coord** and **Y Coord** set the reticle position in the image.

**Color** chooses the reticle color.

**Outline** controls whether the reticle uses an outline for visibility.

**Size** changes the reticle size.

<a id="edit-camera-preview"></a>
### Preview

**Masked Area Color** controls the color used to show masked regions.

**HTTP Stream Quality** controls the live preview stream quality.

**Display Mode** chooses Raw, Mask, or AI preview output.

The polygon mask editor lets the user edit masked areas directly on the preview. These masks affect how the camera image is displayed or processed in modes that use masking.

<a id="camera-confirmations"></a>
## Camera Confirmation Dialogs

<a id="camera-unsaved-changes"></a>
### Unsaved Changes

This dialog appears when closing the Edit Camera modal with pending changes. Confirming discards the changes. Canceling returns to the editor.

<a id="camera-reset-camera"></a>
### Reset Camera

This dialog appears after pressing **Reset Camera** in the Edit Camera modal. Confirming resets only the selected camera to its default configuration.

<a id="camera-reset-all-cameras"></a>
### Reset All Cameras

This dialog appears after pressing **Reset all** on a camera panel. Confirming resets all camera configurations to their defaults.

<a id="settings-ai-setup"></a>
# AI Setup

The AI Setup page configures the rules used by AI operation. It is divided into AI Essentials, Mission, and Exit Strategy panels.

The **AI Essentials** panel shows activation timing, model and device settings, minimum allowed FPS, arming duration, reticle detection radius, AI statistics display state, and organ detection settings. Press **Edit** to open the AI Essentials modal.

The **Mission** panel shows the mission type, duty-cycle override behavior, delay between engagements, engagement duration, and motor position actions used during the mission. Press **Edit** to open the Mission modal.

The **Exit Strategy** panel shows the conditions that end or change AI operation, including engagement count, timeout, fixed date and time, exit duration, and motor position actions. Press **Edit** to open the Exit Strategy modal.

<a id="ai-essentials"></a>
## AI Essentials

The AI Essentials modal controls when AI can activate and how detections are accepted.

<a id="ai-essentials-core-fields"></a>
### Core Fields

**Activation Date/Time** defines when AI operation may become active.

**AI Model** selects the model used for detection.

**Device** selects the processing device used by the model.

**Minimum Allowed FPS** defines the lowest acceptable frame rate for operation.

**Arming Duration** defines how long the target condition must remain valid before engagement.

**Reticle Detection Radius** defines the distance from the reticle within which detections are considered relevant.

**Draw AI Stats** controls whether AI statistics are drawn on the camera image.

<a id="ai-essentials-organ-sections"></a>
### Organ Detection Sections

Each organ section, such as Brain, Chest, Heart, Liver, and Abdomen, controls one detection category.

**Detection Enabled** turns that category on or off.

**Size Multiplier** adjusts the expected detection size.

**Confidence Threshold** sets the minimum confidence required before a detection is accepted.

**Organ Minimum Radius** sets the minimum radius used when evaluating the detected area.

<a id="mission"></a>
## Mission

The Mission modal controls behavior while AI operation is running.

<a id="mission-core-fields"></a>
### Core Fields

**Mission Type** shows the selected mission pattern. In the current interface, Random Walk is the active mission type.

**Disable Duty Cycle After 1st Engagement** controls whether motor duty-cycle protection is disabled after the first engagement.

**Delay Between Engagements** defines the waiting time between engagement attempts.

**Engagement Duration** defines how long an engagement action lasts.

<a id="mission-motor-position"></a>
### Set Motor Position

Each motor can have an enabled position action.

**Enabled** turns the motor action on or off.

**Speed** sets the commanded motor speed from reverse through stop to forward.

<a id="exit-strategy"></a>
## Exit Strategy

The Exit Strategy modal controls how AI operation exits or transitions after configured conditions are met.

<a id="exit-strategy-core-fields"></a>
### Core Fields

**After N Engagements** ends or changes behavior after the configured number of engagements.

**Timeout after 1st Engagement** defines how long the system may continue after the first engagement.

**Fixed Date/Time** defines a fixed UTC time used as an exit condition.

**Exit Strategy Duration** defines how long exit behavior runs.

<a id="exit-strategy-motor-position"></a>
### Set Motor Position

Each motor can have an enabled exit position action.

**Enabled** turns the motor action on or off.

**Speed** sets the commanded motor speed used during the exit strategy.

<a id="hot-zone"></a>
# Hot Zone

The Hot Zone page provides a 3D visualization of the geometry used for reachable-area planning. It helps the user understand how center pole, arm, and stroke settings affect the available operating space.

The page shows horizontal and vertical field-of-view values and an interactive 3D scene. Dragging or zooming in the scene changes the camera view. Press **Reset** to restore the 3D view. Press **Edit** to open the Hot Zone modal.

<a id="hot-zone-modal"></a>
## Hot Zone

The Hot Zone modal edits the geometry used by the 3D visualization. Slider changes update the preview while the modal is open. Press **Save** to validate and store the geometry. Press **Cancel** to discard changes and restore the saved preview.

<a id="hot-zone-general"></a>
### General

**Units** chooses whether dimensions are displayed in centimeters or inches.

**Computation Grid** controls the resolution used for hot-zone computation. A finer grid can improve detail but may require more processing.

<a id="hot-zone-center-pole"></a>
### Center Pole

**Const Radius** defines the center pole radius used in the geometry model.

**Height** defines the center pole height.

**Y Distance** defines the center pole offset on the Y axis.

<a id="hot-zone-arm-poles"></a>
### Arm Poles

**Height** defines the height of the arm poles.

**X Distance** defines the horizontal spacing of the arm poles.

<a id="hot-zone-arms"></a>
### Arms

**Symmetric Arms** controls whether left and right arm settings are linked.

**Left Arm Min Length** and **Right Arm Min Length** define the minimum arm lengths.

**Left Arm Stroke Len.** and **Right Arm Stroke Len.** define the available actuator stroke lengths for each arm.

<a id="tutorials"></a>
# Tutorials

The Tutorials section contains built-in documentation pages rendered as scrollable markdown.

The **System Guide** page displays this guide.

The **What to Buy** page is reserved for purchasing or parts guidance.

The **Electronics** page is reserved for wiring and electronics guidance.

Users can scroll the documentation and follow internal links where the documentation provides them. These pages do not have application-specific modal windows.

<a id="about"></a>
# About

The About page is a placeholder page for general information about the application or device. In the current interface it contains placeholder text only and has no user actions or modal windows.

<a id="developers-sandbox"></a>
# Developers Sandbox

The Developers Sandbox appears only when debug mode is enabled. It contains component demonstration pages for development and testing. These pages are not required for normal operation.

Available sandbox pages can include Components Demo, Modal Windows Demo, Editable Chart, Table Demo, Chart2D Demo, Polygon Zoom Pan, DateTimePicker Demo, and Joystick1D Demo.

Use these pages only when validating interface components or developing the application. They are not part of the normal operator workflow.
