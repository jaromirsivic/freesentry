# XTX3 - Engineering Specification for Efficient Head-and-Torso Pose Detection

## Role and expected outcome

Act as a senior Python and PyTorch computer-vision engineer. Create a new, production-quality project named **XTX3** for training and inference of an extremely efficient human head-and-torso pose detector.

XTX3 must be a specialised successor to the existing XTX2 project. Study the existing XTX2 implementation in the sibling `xtx2` directory before designing XTX3. Reuse sound ideas where they are justified, but do not preserve a design merely because XTX2 already uses it. XTX3 may use a substantially different architecture, preprocessing strategy, training method, and deployment design. The output must be a maintainable, tested implementation, not a conceptual prototype.

The principal deployment target is **XTX3-n on Raspberry Pi**. It must be lightweight and highly efficient on that device. XTX3-m and XTX3-l target NVIDIA GPUs and must provide real-time inference while prioritising high accuracy. All claims about speed, size, or accuracy must be backed by reproducible measurements. Do not claim that XTX3 is better than YOLO26n-pose or XTX2 unless the compared data, preprocessing, thresholds, and hardware are stated.

## Product scope

Detect people and return only the following nine landmarks for each person, in exactly this order:

1. `nose`
2. `left_eye`
3. `right_eye`
4. `left_ear`
5. `right_ear`
6. `left_shoulder`
7. `right_shoulder`
8. `left_hip`
9. `right_hip`

Each landmark must include normalized coordinates and a visibility state:

- `0`: absent, out of frame, or unlabelled
- `1`: occluded
- `2`: visible

The system must also return a person bounding box and a person confidence score. Arms below the shoulders, wrists, hands, knees, ankles, feet, and full-body skeleton reconstruction are explicitly outside the scope of XTX3.

## Why this is a dedicated model

XTX2 is a 17-keypoint full-body pose model. Reducing its keypoint-output head from 17 to nine points is necessary but not sufficient. It reduces the per-location keypoint output from 102 values to 54 values when using XTX2's six values per point, but the backbone, neck, person-detection branch, and multi-level inference remain most of the end-to-end workload.

Design XTX3 as a head-and-torso model from the beginning. Optimise the complete pipeline for this task, including training data, backbone capacity, detection scales, keypoint head, preprocessing, postprocessing, export, and Raspberry Pi runtime. Retain a component only when its contribution is validated.

## Required architecture and variants

Create `n`, `m`, and `l` variants with a shared public API and checkpoint format. The design may be inspired by modern anchor-free, NMS-free pose detectors, but must remain self-contained and must not require Ultralytics or another pose framework at runtime.

- **XTX3-n:** The primary edge model. Prefer operators that run efficiently through the selected Raspberry Pi inference backend. Minimise end-to-end latency, peak memory, model size, and installation complexity while retaining useful accuracy.
- **XTX3-m:** A balanced, high-accuracy NVIDIA GPU model that runs in real time.
- **XTX3-l:** A higher-accuracy NVIDIA GPU model that also runs in real time. Its increased compute must be justified by measurable quality gains.

The architecture must be documented in a separate architecture document with the layer families, feature scales, output representation, parameter counts, per-pass compute estimates, and deployment rationale for each variant.

## Input and central-region strategy

The existing XTX2 system uses three 384 by 384 views from a virtual 1536 by 1536, aspect-preserving canvas: full frame, centre quarter, and centre sixteenth. This can recover small distant people near the image centre, but it triples network passes and introduces merge cost.

XTX3 must evaluate this strategy rather than accept it automatically:

1. Implement a baseline whole-frame mode.
2. Implement the existing three-level central strategy only if its coordinate mapping is exact and the implementation avoids unnecessary full-canvas materialisation and repeated resampling.
3. Benchmark whole-frame and multi-level modes on the intended camera scenes, including images with small central subjects.
4. Make the selected mode configurable. The default for Raspberry Pi must be justified by measured accuracy-latency trade-offs.

If multiple views are retained, execute equal-size views as one batch where that is beneficial on the selected backend. Map boxes and keypoints back to original image coordinates precisely. Deduplicate the same person across views deterministically and document the priority or score policy.

## Dataset contract

Train on the official COCO human-keypoint dataset. Implement a robust COCO dataset adapter that reads COCO's 17 keypoints and deterministically selects only the nine retained points. Preserve left/right semantics, convert coordinates to the project's chosen internal representation, validate ranges and visibility, and report invalid or skipped records clearly. Never silently reinterpret point order.

The project may also support a separate local `train` and `test` dataset layout for convenience, but COCO is the required baseline training and validation source. XTX3 must not require compatibility with XTX2 annotations, model checkpoints, or output artifacts.

Training augmentation must transform boxes and keypoints correctly. Horizontal flipping must exchange all left/right points: eyes, ears, shoulders, and hips. Cropping, mosaic, resizing, and affine transforms must not mark a landmark as visible when it is outside the transformed image.

The training data and evaluation data must represent deployment conditions: camera angle, person distance, partial body framing, face size, lighting, occlusion, motion blur, and multiple people. Document known coverage gaps.

## Training pipeline

Provide a Windows launcher and a Python command-line entry point for training. The interactive workflow must ask for:

1. whether to resume from an existing checkpoint when one is available,
2. the model variant (`n`, `m`, or `l`),
3. the dataset path, defaulting to `./dataset`,
4. the output directory for checkpoints.

Keep configuration in a root JSON file. Include sensible defaults for epochs, checkpoint interval, seed, image size, augmentation, optimizer, scheduler, confidence threshold, and model-specific settings. Save a last checkpoint each epoch, periodic recovery checkpoints, and the best checkpoint according to the declared validation metric.

The console output must report data validation findings, training and validation loss components, model size, elapsed training progress, validation results, and checkpoint locations. All errors must explain what failed and how the user can correct it.

## Evaluation and acceptance metrics

Do not reuse COCO's 17-keypoint OKS sigmas as if they were automatically valid for this nine-point task. Define and document task-appropriate landmark evaluation parameters. Report at least:

- person detection precision and recall,
- aggregate pose quality for the nine-point definition,
- per-landmark quality and visibility classification quality,
- results separated by face and torso scale where possible,
- model parameters, artifact size, peak memory, and end-to-end latency,
- preprocessing, network execution, decoding, and cross-view merge time separately.

Benchmark XTX3-n on the designated Raspberry Pi using the exported runtime, not only desktop PyTorch. Compare it to the current XTX2-n baseline under the same COCO validation images, preprocessing mode, confidence threshold, and hardware. Benchmark XTX3-m and XTX3-l on the designated NVIDIA GPU and report their complete real-time latency and throughput, including preprocessing and postprocessing. The acceptance decision must be based on retained-landmark accuracy and full-pipeline resource use.

## Inference API

Provide a small Python package API that accepts a BGR NumPy image and returns a list or array of poses. Each pose must expose:

- `bbox_xyxy` in original-image pixels,
- person score,
- keypoints as a `(9, 3)` array of original-image `x`, `y`, and confidence,
- normalized keypoints,
- nine visibility classes,
- source view or level when multi-view inference is enabled.

The flat array representation must contain exactly 32 values:

`[x1, y1, x2, y2, person_score, 9 * (x, y, keypoint_confidence)]`

Document the coordinate system, point order, confidence meaning, visibility meaning, and behavior for occluded or outside-image points. XTX3 does not need to load XTX2 checkpoints or exports.

Provide a simple Windows inference launcher and Python example that load a checkpoint, process an image, optionally save annotated output, and optionally save JSON results. Visualisation must draw only meaningful face, shoulder, and hip connections.

## Edge export

Provide a repeatable XTX3-n export path to ONNX and an edge runtime suitable for Raspberry Pi, preferably NCNN when all exported operators are supported efficiently. Keep export graphs simple and stable. If raw network outputs are decoded on the host, document the tensor layout and supply the complete host-side decoder.

Verify exported runtime results against PyTorch on deterministic test inputs within defined numerical tolerance. Include instructions for required converter and runtime dependencies on Raspberry Pi.

## Quality requirements

- Use typed, clear Python and structured logging.
- Avoid hidden global assumptions about 17 keypoints.
- Keep landmark names, skeleton edges, evaluation metadata, and output dimensions in one shared source of truth.
- Add unit tests for annotation conversion, left/right flip semantics, geometric transforms, pose-head shapes, losses, decoding, API output shape, cross-view merging, checkpoint compatibility, and export-output decoding.
- Add benchmark tools for parameters, compute estimates, preprocessing, PyTorch inference, and edge-runtime inference.
- Keep documentation current with the implementation.

## Deliverables

Create the XTX3 source project, its architecture document, configuration, training and inference launchers, sample commands, tests, benchmark tools, dataset conversion tool, and edge export path. Include a concise report with measured XTX3-n results against the current XTX2-n baseline. Explicitly state any target that was not achieved or could not be measured.
