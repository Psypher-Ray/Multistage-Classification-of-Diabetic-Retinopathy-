👁️ Diabetic Retinopathy (DR) Severity Classification

This repository provides a comprehensive deep learning pipeline for Multistage Classification of Diabetic Retinopathy (DR) from fundus images. Our approach showcases sequential training strategies, robust evaluation, and interpretability features.

✨ Features

5-Stage DR Classification: Predicts severity from 0: No DR to 4: Proliferative DR.

Sequential Training: Progressive model refinement through multiple training stages.

Global & Local Features: Demonstrates combining global (ResNet50) and simplified local (ResNet18 central crop) image features.

Imbalance Handling: Implements WeightedRandomSampler and class-weighted loss for sparse DR stages.

Interpretability: Integrates Grad-CAM for visualizing model's focus on lesion hotspots.

Comprehensive Evaluation: Reports standard metrics (Accuracy, F1-score) alongside Quadratic Weighted Kappa (QWK), crucial for ordinal classification.

Advanced Augmentation: Incorporates techniques like RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, and ColorJitter.


📂 Data Preparation

Acquire Data: Get a fundus image dataset (e.g., Messidor-2).

Extract: If zipped, extract to /content/extracted_messidor/.

code
Bash
download
content_copy
expand_less
unzip -q "/content/archive (1).zip" -d /content/extracted_messidor

Structure:

messidor_data.csv at /content/extracted_messidor/messidor_data.csv.

Image files (.png/.jpg) in /content/extracted_messidor/messidor-2/messidor-2/preprocess.

Adjust csv_path and image_folder variables in the code if your paths differ.

📋 Usage Guide (Notebook Flow)

The code is designed to be executed step-by-step, mimicking a Jupyter/Colab notebook.

1. Initial Data Loading & Basic Preprocessing

➡️ Sets up data paths and loads messidor_data.csv.

🔗 Links image files to labels, drops missing entries.

📊 Visualizes DR stage distribution (important for imbalance).

📦 Defines MessidorDataset for PyTorch, including basic image transforms.

2. Global Branch Training & Evaluation

🧠 Model: DRGlobalModel (ResNet50 backbone).

⚖️ Imbalance: Uses WeightedRandomSampler in DataLoader and class-weighted CrossEntropyLoss.

💻 Training: Trains for 9 epochs, tracking loss & accuracy.

💾 Saves: dr_global_branch.pth.

✅ Evaluation: Calculates QWK, generates classification_report and confusion_matrix.

3. Grad-CAM Visualization

🔍 Loads dr_global_branch.pth.

🎨 Generates and displays Grad-CAM heatmaps for a high-severity image.

💡 Provides visual insights into model's decision-making.

4. Expert-Level Training with Advanced Augmentation

📈 Augmentation: Introduces RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter.

🔄 Model/Optimizer: Re-initializes ResNet50, uses AdamW with weight_decay.

⏰ Scheduler: Implements CosineAnnealingLR for learning rate scheduling.

🏆 Training: Runs for 20 epochs, saving expert_dr_model.pth.

5. Triple-Granularity Fusion Model Training

🧩 Dataset: TripleGranDataset provides both global image and a central-crop local image.

🔬 Model: DRFusionModel

global_net: ResNet50 (pre-loaded with expert_dr_model.pth weights).

local_net: ResNet18 (for the local crop).

fusion_layer: Combines features from both nets for final prediction.

🔄 Training: Trains for 10 epochs, saving final_dr_fusion_model.pth.

🖼️ Test-Time Augmentation (TTA): Applies image flipping during evaluation and averages predictions for robustness.

6. Final Expert Evaluation

🏅 Evaluates the final_dr_fusion_model.pth (with TTA).

📊 Reports FINAL SCIENTIFIC KAPPA SCORE and displays the Clinical Agreement Matrix.

🎯 Provides an assessment against an "expert-level" QWK threshold (0.80).

🧠 Architectural Overview

Our pipeline progresses through increasingly sophisticated model configurations:

Global Branch: A standard ResNet50 for initial feature extraction from the entire fundus.

"Expert" Training: Enhances the Global Branch with advanced data augmentation for improved generalization.

Triple-Granularity Fusion:

Combines the Global Branch's features with a simplified Local Branch (ResNet18 processing a central image crop).

Note: The full "diffusion-guided" aspect, while in the title, is not explicitly demonstrated in these code snippets.

🛠️ Dependencies

pandas

matplotlib

torch, torch.nn, torch.utils.data

torchvision (transforms, models)

Pillow (PIL)

numpy

seaborn

scikit-learn (metrics)

opencv-python (cv2)

time

📜 License

This project is licensed under the MIT License.
