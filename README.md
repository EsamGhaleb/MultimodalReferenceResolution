# I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue

### Prerequisites

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation
1. **1. Download Pose Data** 
   - Download the pose data from [Google Drive](https://drive.google.com/file/d/1NdrVSsrZxp3MWrGsBybKops_QuTbi4Gq/view?usp=sharing).
   - Make sure the downloaded files are extracted and placed in the `data/selected_poses/` directory within this repository.

2. **2. Prepare Dialogue Data** 
A significant portion of the project involved preparing dialogue data. Relevant scripts can be found in `dialog_utils/`. The following links provide the data used in the project:
   - Segmented Gestures: Download the segmented gesture data from [Google Drive](https://drive.google.com/file/d/15zoDB3aDv2HViJttQPj0tnNOPUvSyz3U/view?usp=sharing) and place the file in the `data/segmented_gestures/` directory within this repository.
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
   - Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1jPt3NZsbDbL5YSsci61zCQi5i6AfdGE7/view?usp=sharing) and place it in the `pretrained_models/` directory within this repository.
2. **Extract Embeddings:** 
   - You can use the pre-trained model to generate gesture embedings by executing:
     ```bash
     python save_extracted_embeddings.py
     ```
   - The extracted embeddings will be saved in the `data/` directory.

### Evaluation
1. **Reference Resolution:** 
   - You can evaluate the model by executing:
     ```bash
     python reference_resolution_classification.py
     ```
   - The evaluation results will be saved in the `results/` directory.
2. **Reference Resolution With Dialogue History:**
   - You can evaluate the model with dialogue history by
       ```bash
       python reference_resolution_with_dialogue_impact.py
       ```
   - The evaluation results will be saved in the `results/` directory.
3. **The reported results in the paper are available in the `results/` directory.**
   - The notebook `results/PaperResults.ipynb` provides a detailed analysis of the results.
## Results
The results of the evaluation can be found in the `results/` directory.
### Citation
   ```
   @inproceedings{ghaleb-etal-acl-2025, 
     author={Ghaleb, Esam and Khaertdinov, Bulat and {\"O}zy{\"u}rek, Asl{\i} and Fern{\'a}ndez, Raquel},
     title={I see what you mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue},
     booktitle={Proceedings of the of the 63rd Conference of the Association for Computational Linguistics (ACL Findings)},
     year={2025},
     note={To appear.},
     url={https://arxiv.org/abs/2503.00071},
     url_github={https://github.com/EsamGhaleb/MultimodalReferenceResolution},
     abstract={In face-to-face interaction, we use multiple modalities, including speech and gestures, to communicate information and resolve references to objects. However, how representational co-speech gestures refer to objects remains understudied from a computational perspective. In this work, we address this gap by introducing a multimodal reference resolution task centred on representational gestures, while simultaneously tackling the challenge of learning robust gesture embeddings. We propose a self-supervised pre-training approach to gesture representation learning that grounds body movements in spoken language. Our experiments show that the learned embeddings align with expert annotations and have significant predictive power. Moreover, reference resolution accuracy further improves when (1) using multimodal gesture representations, even when speech is unavailable at inference time, and (2) leveraging dialogue history. Overall, our findings highlight the complementary roles of gesture and speech in reference resolution, offering a step towards more naturalistic models of human-machine interaction.}
   }
