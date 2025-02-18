# I see what you mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue

This repository contains resources and scripts necessary for the project submitted to ARR February 2025.

## Getting Started
These instructions will guide you to setup and run the project on your local machine.

### Prerequisites
Setup virtual environment and install dependencies:
```
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

### Downloading Data
1. **Download Poses:** 
   - Download the pose data from [Google Drive](https://drive.google.com/file/d/15SwxhEXC4JOJ0XYiQ-WcrGSmEVVvdIDB/view).
   - Make sure the downloaded files are extracted and placed in the `data/final_poses/` directory within this repository.

### Dialogue Data Preprapation
2. **A large part of the work was conducted to prepared dialogue data, with scripts that can be found in dialog_utils/**
   The following links provide the data used in the project:
2. **Dialogue Data:** 
   - Segmented Gestures: Download the segmented gesture data from [Google Drive](https://drive.google.com/file/d/1Bt59e0q4KzGje8HyRhlxfs3nZ9VD6HAM/view?usp=sharing) and place it in the `data/segmented_gestures/` directory within this repository.
   - Transcribed and aligned speech of CABB-S is placed in `dialog_utils/data/aligned_tagged_speech_per_word_small.csv`
   - Transcribed and aligned speech of CABB-L is placed in `dialog_utils/data/aligned_tagged_speech_per_word_large.csv`
   - Annotated gestures with referents are in `dialog_utils/data/gestures_referents_info.csv`
   - Gesture pairs annotated in terms of form similarity are in `data/gesture_form_similarity_coding.csv`

### Model Pre-Training
2. **Train the Model:** 
   - After preparing the data, you can train the model by executing:
     ```bash
     python pre_train_main.py --config configs/pretraining/crossmodal/train_ssl_crossmodal_mmcontrastive_jointsformer_semantic.yaml
     ```

### Using Pre-Trained Model
3. **Download Pre-Trained Model:** 
   - Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/11qR3dO6vpsC6VvFVJWVWhT12iuFE1O9i/view?usp=sharing) and place it in the `pretrained_models/` directory within this repository.
4. **Extract Embeddings:** 
   - You can use the pre-trained model to generate gesture embedings by executing:
     ```bash
     python save_extracted_embeddings.py
     ```
   - The extracted embeddings will be saved in the `data/extracted_embeddings/` directory.

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