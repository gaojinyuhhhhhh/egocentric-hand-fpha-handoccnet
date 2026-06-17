# Brief Survey of Related Work

## 1. Egocentric hand pose estimation

Egocentric hand pose estimation studies infer hand pose from first-person camera views. Compared with third-person setups, egocentric views are harder because the hands are often partially outside the frame, self-occluded, or interacting with objects close to the camera.

## 2. FPHA dataset

FPHA (First-Person Hand Action) is a first-person RGB-D dataset with 3D hand joint annotations. It is commonly used for evaluating hand pose estimation under egocentric interaction and object occlusion.

## 3. HandOccNet

HandOccNet is a pretrained occlusion-robust 3D hand mesh estimation network. It predicts hand joints and MANO mesh from cropped RGB input and is designed to be robust under hand-object occlusions.

Strengths:
- Strong 3D hand mesh output.
- Better robustness under occlusion than many lightweight detectors.
- Good fit for FPHA-style egocentric scenes.

Limitations:
- Still sensitive to heavy occlusion and crop misalignment.
- Single-frame inference has depth ambiguity.
- Performance depends on accurate hand crop and MANO compatibility.

## 4. MediaPipe Hands

MediaPipe Hands is a fast and lightweight hand landmark detector. It is useful as a baseline for 2D hand keypoints, but it is not a full 3D mesh estimator.

Strengths:
- Easy to run.
- Fast inference.
- Good for visible hands in standard scenes.

Limitations:
- Less suitable for occluded egocentric scenes.
- No direct MANO mesh output.
- 3D interpretation is limited compared with HandOccNet.

## 5. InterHand2.6M

InterHand2.6M is a large-scale dataset for 2D/3D hand pose estimation, especially for two-hand interaction scenarios. It is useful for broader hand pose research, but FPHA is more directly aligned with egocentric action understanding.

## 6. What this project uses

This project uses:
- FPHA as the dataset;
- HandOccNet as the main pretrained baseline;
- GT visualization and depth-based occlusion analysis;
- optional MediaPipe as a lightweight baseline reference.

## 7. Key takeaway

For egocentric RGB hand pose estimation, the main technical challenge is not only 2D detection but also occlusion-aware 3D reasoning. HandOccNet is a reasonable baseline because it provides both pose and mesh output, while FPHA offers a good benchmark for first-person interaction scenes.
