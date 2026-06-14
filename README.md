# SA-ESegNet_water: A Shadow Attention-Driven Framework for Accurate Arbitrary-Shaped Water Region Segmentation in Complex Aerial Imagery


SA-ESegNet is a deep learning framework designed for accurate segmentation of arbitrary-shaped water regions in complex aerial imagery. The proposed architecture incorporates a **Shadow Attention Module (SAM)** to handle shadow-induced ambiguities, effectively improving water segmentation performance.

## Project Structure

```text
├── shadow_det.py          # Generate shadow-detected label images
├── Training.py            # Train the pre-trained model on Dataset II
├── ft.py                  # Fine-tune the pre-trained model on Dataset I
├── Dataset/
│   ├── Dataset_I/         # Target dataset used for fine-tuning
│   ├── Dataset_II/        # Source dataset used for pre-training
│   └── Dataset_III/       # Additional dataset for evaluation/testing
├── weights/
    ├── pre-trained_model_weight.hdf5      # Pre-trained model weights
    ├── ft_model_weights.hdf5              # Fine-tuned model weights
└── README.md
```


## Requirements

* Python 2.7
* TensorFlow
* Keras
* OpenCV
* NumPy
* SciPy
* Matplotlib

Install the required packages before running the project.

## Workflow

### Step 1: Generate Shadow Labels

Run the following script to generate shadow-detected label images:

```bash
python shadow_det.py
```

This script processes the training images and creates shadow-aware label masks that are used during model training.

### Step 2: Train the Pre-trained Model on Dataset II

Train the network using Dataset II:

```bash
python Training.py
```

This step initializes and trains the model using Dataset II to learn general water segmentation features. The pre-trained model weights can be found in the weights folder.

### Step 3: Fine-tune the Model on Dataset I

Fine-tune the trained model on Dataset I:

```bash
python ft.py
```

This step adapts the pre-trained model to the task-specific dataset, Dataset I, improving segmentation performance on the target dataset. The fine-tuned model weights can be found in the weights folder


### Step 4: Evaluation on Dataset III

```bash
python eva.py
```


## Citation (To be updated)

If you use this code in your research, please cite:

```bibtex
@article{SAESegNet,
  title={SA-ESegNet: A Shadow Attention-Driven Framework for Accurate Arbitrary-Shaped Water Region Segmentation in Complex Aerial Imagery},
  author={},
  journal={},
  year={}
}
```
