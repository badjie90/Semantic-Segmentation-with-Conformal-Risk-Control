# DEL semantic segmentation scripts

This repository contains training and evaluation scripts for three semantic segmentation backbones with Dirichlet evidential learning and split conformal selective prediction. The target classes are path, object, and background.

## Repository contents

| File | Purpose |
| --- | --- |
| Segformer-DEL-train.py | Train SegFormer with cross entropy pretraining and evidential fine tuning |
| Segformer-DEL-test.py | Evaluate SegFormer with uncertainty metrics and conformal selective prediction |
| Mask2former-train.py | Train Mask2Former with the same DEL protocol |
| Mask2Former-DEL-Test.py | Evaluate Mask2Former with the same runtime protocol |
| DinoV3-DEL-train.py | Train DINOv3 or DINOv2 segmentation decoder with the same DEL protocol |
| DinoV3-DEL-test.py | Evaluate DINOv3 or DINOv2 with the same runtime protocol |

## Methodology

The scripts train multiclass semantic segmentation models using paired RGB images and integer label masks. Training uses a two stage protocol.

First, a task specific baseline is optimized with per pixel cross entropy. Second, the baseline is fine tuned with a Dirichlet evidential objective. The evidential stage converts logits to Dirichlet evidence with softplus and adds one to obtain positive concentration parameters. The loss combines cross entropy, evidential mean square error, predictive variance, annealed KL divergence to a uniform Dirichlet prior, and L2 SP regularization relative to the baseline weights. Staged freezing controls backbone updates during fine tuning.

The data are split into training, calibration, and test subsets with ratios 0.70, 0.15, and 0.15. Runtime evaluation computes mean class probabilities, total entropy, aleatoric expected entropy, epistemic mutual information, calibration error, risk coverage curves, and per class segmentation metrics. Calibration pixels define predicted class conditional conformal thresholds. Test pixels are accepted when selected uncertainty scores are below the threshold assigned to the predicted class.

## Data format

Place RGB images and masks in separate directories. File ordering must produce paired image and mask lists after sorting. Mask pixels must contain class indices.

| Class index | Class name |
| --- | --- |
| 0 | path |
| 1 | object |
| 2 | background |

## Environment

Use Python 3.10 or a compatible version with CUDA enabled PyTorch for GPU training.

```bash
pip install torch torchvision transformers pillow numpy pandas scikit-learn tqdm matplotlib onnx onnxruntime
```

DINO training also requires a compatible Torch Hub environment for the selected DINO backbone.

## Smoke tests

Run smoke tests before full training. These commands validate tensor shapes, loss computation, and runtime uncertainty functions.

```bash
python Segformer-DEL-train.py --smoke-test
python Segformer-DEL-test.py --smoke-test
python Mask2former-train.py --smoke-test
python Mask2Former-DEL-Test.py --smoke-test
python DinoV3-DEL-train.py --smoke-test
python DinoV3-DEL-test.py --smoke-test
```

## Training

Replace paths with local directories.

```bash
python Segformer-DEL-train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --output-dir outputs/segformer/train \
  --split-dir outputs/segformer/splits \
  --device cuda
```

```bash
python Mask2former-train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --output-dir outputs/mask2former/train \
  --split-dir outputs/mask2former/splits \
  --device cuda
```

```bash
python DinoV3-DEL-train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --output-dir outputs/dino/train \
  --split-dir outputs/dino/splits \
  --device cuda
```

Training writes the best baseline checkpoint, final baseline checkpoint, best evidential checkpoint, final evidential checkpoint, split file, training summary, and ONNX model.

## Evaluation

Use the split file and best evidential checkpoint generated during training.

```bash
python Segformer-DEL-test.py \
  --split-json outputs/segformer/splits/split.json \
  --model-path outputs/segformer/train/evidential_theta_best_segformer.pth \
  --output-dir outputs/segformer/test \
  --device cuda
```

```bash
python Mask2Former-DEL-Test.py \
  --split-json outputs/mask2former/splits/split.json \
  --model-path outputs/mask2former/train/evidential_theta_best_mask2former.pth \
  --output-dir outputs/mask2former/test \
  --device cuda
```

```bash
python DinoV3-DEL-test.py \
  --split-json outputs/dino/splits/split.json \
  --model-path outputs/dino/train/evidential_theta_best.pth \
  --output-dir outputs/dino/test \
  --device cuda
```

Evaluation writes segmentation metrics, conformal risk coverage metrics, uncertainty plots, threshold plots, and runtime summary files.

## Main outputs

| Output | Description |
| --- | --- |
| training_summary.json | Training configuration and checkpoint paths |
| split.json | Train, calibration, and test partition |
| evidential_theta_best*.pth | Best PyTorch evidential checkpoint |
| evidential_theta_best*_del.onnx | ONNX export for inference deployment |
| per_class_segmentation_metrics.csv | Per class IoU, precision, recall, F1, AP, and support |
| class_conditional_conformal_metrics.csv | Risk, coverage, and class conditional thresholds |
| runtime_summary.json | Runtime configuration and selective prediction rule |

## Hardware deployment

Use the exported ONNX model for hardware inference. The standard deployment path is ONNX Runtime for CPU, CUDA GPU, or embedded GPU. TensorRT conversion can be used when the target device supports TensorRT engines.

Deployment procedure.

1. Train one model and keep the best evidential checkpoint.
2. Use the ONNX file written by the training script.
3. Verify ONNX inference on a validation image with the same image size used during training.
4. Apply the same image normalization and resizing used by the corresponding Hugging Face image processor or DINO preprocessing path.
5. Run inference on the target device with ONNX Runtime or a TensorRT engine.
6. Convert output logits or probabilities to Dirichlet parameters with the same rule used in the runtime script.
7. Apply the saved conformal thresholds when selective prediction is required.
8. Store rejected pixels as abstentions or pass them to a safety controller.

ONNX Runtime deployment requires the exported model, one preprocessed float32 input tensor, and the same provider order used during validation.

The input tensor must use shape N, C, H, W. Use the image size encoded by the trained model. Keep preprocessing, class order, and threshold files unchanged between validation and deployment.

## Reproducibility notes

The scripts set the random seed to 42 by default. Report the model family, checkpoint name, image size, class names, train calibration test split, loss weights, conformal alpha values, selected uncertainty scores, and hardware backend for each experiment.
