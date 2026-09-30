# LBW Detection: Training and Integration Guide

This project aims to estimate LBW outcomes from a delivery video. It is not a full Hawk-Eye system: it does not reconstruct a calibrated 3D ball path from multiple cameras. Object detection finds the ball and wicket; tracking and LBW rules must then evaluate the delivery.

## 1. What the Current Project Does

The current checkpoint is `weights/best.pt`. It has two classes:

| Class ID | Class name | Purpose |
|---|---|---|
| 0 | `ball` | Find ball candidates for tracking |
| 1 | `stump` | Find the wicket and estimate the batter-end lane |

The current detector in `src/detector.py` treats every class other than `ball` as a stump. The current pipeline estimates pitching and impact, but it does not yet produce a complete, law-aware `OUT` / `NOT OUT` decision. Do not plug in extra classes until the detector has explicit routing for them.

A public cricket-ball checkpoint is also present at `weights/cricket_ball_pretrained.pt`. It detects only `ball`; it has no stump class, and tests on these clips were mixed. It is not a drop-in replacement for `weights/best.pt`.

## 2. Recommended Classes

Build in two phases so ball tracking can be improved and measured before adding more behavior.

### Phase A: Improve the Existing Detector

Keep the two existing class names exactly as written:

| ID | Name | Annotation definition |
|---|---|---|
| 0 | `ball` | Tight box around the visible ball. Label it in every frame where it can be located reliably. |
| 1 | `stump` | One consistent box around the wicket set at an end, including all three stumps and bails when visible. |

This two-class model can use the current detector without adding new object types.

### Phase B: Add LBW Contact Cues

For a more complete LBW workflow, extend the model and code to use:

| ID | Name | Annotation definition |
|---|---|---|
| 2 | `pad` | Box around each visible batting pad. |
| 3 | `bat` | Box around the bat, including when it is near the ball. |

Before using this four-class checkpoint, update `FrameDet` and `detect_video()` in `src/detector.py` so `pad` and `bat` are stored separately instead of being treated as stumps. Update the pipeline and tests to consume those detections. Bat-first contact and whether the batter attempted a shot may need temporal/action logic; a single-frame object detector cannot reliably answer those questions by itself.

## 3. Collect and Annotate Data

1. Collect varied deliveries: different grounds, camera positions, zoom levels, lighting, ball colors, motion blur, and partial occlusion. Ten clips are useful for a smoke test but are not enough for a robust detector. A practical starting target is 50-100 distinct deliveries and 2,000-5,000 representative labeled frames; treat this as a starting point, not a guarantee.
2. Include frames before release, during flight, around bounce, around impact, and after impact. Include frames with no visible ball so the model learns hard negatives.
3. For tiny or blurred balls, label only when a human can identify the ball location with reasonable confidence. Do not invent boxes for fully hidden balls.
4. Keep box definitions consistent. Do not label a whole player as a `pad`, or vary between boxing one stump and the whole wicket.
5. Use CVAT or Roboflow to export annotations in Ultralytics YOLO detection format. Each label line is `class_id x_center y_center width height`, with coordinates normalized from 0 to 1.
6. Split by complete video/delivery, not by individual frame. No frames from one delivery should appear in both training and validation/test splits.

Suggested dataset layout:

```text
cricket_dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Example `data.yaml` for Phase A:

```yaml
path: C:/path/to/cricket_dataset
train: images/train
val: images/val
test: images/test

names:
  0: ball
  1: stump
```

For Phase B, add `2: pad` and `3: bat` only after both your labels and application code support those classes.

## 4. Fine-Tune a Pretrained YOLO Model

Use a general pretrained YOLO checkpoint as initialization, then fine-tune it on your cricket annotations. Do not train from random weights for this small dataset.

From the project root in PowerShell, install the declared dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

Example training command for a GPU:

```powershell
yolo detect train model=yolo11s.pt data=C:/path/to/cricket_dataset/data.yaml project=runs/lbw name=ball_stump_v1 imgsz=1280 epochs=100 batch=8 device=0 patience=20
```

If GPU memory is limited, reduce `batch` to 4 or 2. If you do not have a supported GPU, use `device=cpu`; training will be much slower. `yolo11n.pt` is a smaller alternative, while `yolo11s.pt` may give better capacity if hardware permits.

After training, the checkpoint should be at:

```text
runs/lbw/ball_stump_v1/weights/best.pt
```

Keep the current `weights/best.pt` unchanged until the new checkpoint passes validation.

## 5. Evaluate Before Replacing the Current Model

1. Run Ultralytics validation on the held-out test set and record per-class precision, recall, and mAP. For this task, ball recall and false ball detections per frame matter especially; a good overall mAP alone is not sufficient.
2. Run the project's detector health check on test clips:

```powershell
python tools/check_detections.py videos/input/1.mp4
```

3. Run the application with the candidate two-class model:

```powershell
python main.py videos/input/1.mp4 --weights runs/lbw/ball_stump_v1/weights/best.pt --imgsz 1280
```

4. Inspect the output video and JSON. Record whether the ball is tracked from release through bounce and impact, how many track points were detected versus interpolated, and whether batter stumps were found.
5. Compare old and new checkpoints on the same unseen deliveries and same settings. Keep the new checkpoint only if track continuity improves without introducing convincing false tracks or losing stump detections.

Do not evaluate by running only on frames used for training. Keep a small, fixed test set of complete deliveries that you do not use while tuning.

## 6. Implement the LBW Decision Separately

A detector does not decide LBW. The decision layer needs to use the tracked ball and calibrated wicket geometry to estimate:

1. Where the ball pitched relative to the wicket line.
2. Where the ball hit the batter, if it hit the batter.
3. Whether bat contact happened before pad contact, where the video makes this observable.
4. Whether the projected ball path would hit the wicket.
5. The relevant law conditions, including pitching outside leg, impact position, and whether a genuine shot was attempted for the outside-off condition.

Represent uncertain or occluded cases as `unclear` rather than forcing `OUT` or `NOT OUT`. Validate the rules with labeled examples and, where possible, review decisions against reliable human annotations. Camera perspective and calibration matter: a pixel-space lane from a single uncalibrated view is only an approximation.

## 7. Acceptance Checklist

- [ ] Training, validation, and test sets are split by delivery/video.
- [ ] Ball boxes are tight and consistently labeled across motion blur and ball colors.
- [ ] Stump boxes consistently represent the wicket set.
- [ ] New two-class checkpoint improves held-out ball tracking and retains stump detection.
- [ ] Extra classes have explicit detector routing before a four-class checkpoint is loaded.
- [ ] LBW rules are evaluated separately from detector metrics.
- [ ] Ambiguous video cases can return `unclear`.
