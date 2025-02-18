# I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue

This repository provides all the resources and scripts needed for the ARR February 2025 project.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation
1. **1. Download Pose Data** 
   - Download the pose data from [Google Drive](https://drive.google.com/file/d/15SwxhEXC4JOJ0XYiQ-WcrGSmEVVvdIDB/view).
   - Make sure the downloaded files are extracted and placed in the `data/final_poses/` directory within this repository.

2. **2. Prepare Dialogue Data** 
A significant portion of the project involved preparing dialogue data. Relevant scripts can be found in `dialog_utils/`. The following links provide the data used in the project:
   - Segmented Gestures: Download the segmented gesture data from [Google Drive](https://drive.google.com/file/d/1Bt59e0q4KzGje8HyRhlxfs3nZ9VD6HAM/view?usp=sharing) and place the file in the `data/segmented_gestures/` directory within this repository.
   - Transcribed and aligned speech of CABB-S is placed in `dialog_utils/data/aligned_tagged_speech_per_word_small.csv`
   - Transcribed and aligned speech of CABB-L is placed in `dialog_utils/data/aligned_tagged_speech_per_word_large.csv`
   - Annotated gestures with referents are in `dialog_utils/data/gestures_referents_info.csv`
   - Gesture pairs annotated in terms of form similarity are in `data/gesture_form_similarity_coding.csv`

### Model Pre-Training
2. **Train the Model:** 
   - Once the data is prepared, you can train the model by executing the following command:
     ```bash
     python pre_train_main.py --config configs/pretraining/multimimodal-x/train_multimimodal-x_semantic.yaml
     ```
   - The trained model will be saved in the `workdir/` directory.
   - For pre-training models with variant objectives, please check the config files in `configs/pretraining/`. E.g., you can change the text-semantic modality to speech based encoder (wav2vec2) by changing the `modalities` parameter in the config file.

### Using Pre-Trained Model
If you prefer to use the pre-trained model, follow these steps:
1. **Download Pre-Trained Model:** 
   - We provide a pre-trained model for unimodal, multimodal, and multimodal-x models (based on the saved models with good correlation with human judgments).
   - Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/11qR3dO6vpsC6VvFVJWVWhT12iuFE1O9i/view?usp=sharing) and place it in the `pretrained_models/` directory within this repository.
2. **Extract Embeddings:** 
   - You can use the pre-trained model to generate gesture embedings by executing:
     ```bash
     python save_extracted_embeddings.py
     ```
   - The extracted embeddings will be saved in the `data/` directory.

### Evaluation
5. **Reference Resolution:** 
   - You can evaluate the model by executing:
     ```bash
     python reference_resolution_classification.py
     ```
   - The evaluation results will be saved in the `results/` directory.
6. **Reference Resolution With Dialogue History:**
   - You can evaluate the model with dialogue history by
       ```bash
       python reference_resolution_with_dialogue_impact.py
       ```
   - The evaluation results will be saved in the `results/` directory.

## Results
The results of the evaluation can be found in the `results/` directory.
### Citation
TBD