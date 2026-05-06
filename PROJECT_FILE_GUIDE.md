# Project File Guide

This is a plain-English guide to the project. It tells you what each important file does and how the pieces fit together.

## Big Picture

The project has three main parts:

- The **training code** at the root of the project.
- The **core model logic** inside [src/](src).
- The **website** inside [web/](web).

If you only want the main workflow, it is:

1. Prepare the dataset.
2. Train the model.
3. Evaluate the saved checkpoint.
4. Run the website to upload audio and get a prediction.

## Main Root Files

These are the files you are most likely to open first.

- [README.md](README.md): Main overview of the project.
- [INDEX.md](INDEX.md): Quick index for finding the right file.
- [00_README_START_HERE.txt](00_README_START_HERE.txt): Short “start here” note for beginners.
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md): Fast command cheat sheet.
- [SETUP.md](SETUP.md): Setup steps for installing everything.
- [USAGE_GUIDE.md](USAGE_GUIDE.md): How to use the scripts in the project.
- [START_TRAINING.md](START_TRAINING.md): Simple instructions for starting training.
- [TECHNICAL_SPECS.md](TECHNICAL_SPECS.md): Technical details about the model and pipeline.
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md): Short summary of what the project is.
- [PROJECT_COMPLETION_STATUS.md](PROJECT_COMPLETION_STATUS.md): Progress/status report.
- [COLLEGE_PROJECT_COMPLETE_REPORT.md](COLLEGE_PROJECT_COMPLETE_REPORT.md): Full report for college submission.

## Training and Utility Scripts

These files do the actual work of preparing data, training the model, and checking results.

- [train.py](train.py): Main training script. It loads data, builds the model, trains it, saves checkpoints, and can test after training.
- [evaluate.py](evaluate.py): Loads a saved model and runs evaluation on the test set.
- [prepare_dataset.py](prepare_dataset.py): Creates the folder structure for the dataset or organizes audio files into train, validation, and test folders.
- [organize_data.py](organize_data.py): Copies audio files from another folder into the project dataset structure.
- [check_gpu.py](check_gpu.py): Checks whether the GPU/CUDA setup is ready.
- [quick_check.py](quick_check.py): Very quick check to see if the project environment looks okay.
- [status.py](status.py): Prints a full “ready to train” status message.
- [monitor_training.py](monitor_training.py): Watches a training run and shows whether checkpoints are being created.
- [check_metrics.py](check_metrics.py): Looks at saved evaluation metrics.
- [analyze_metrics.py](analyze_metrics.py): Helps interpret the metrics and training results.
- [extract_pdf_report.py](extract_pdf_report.py): Pulls text out of PDF reports and saves it as text files.

## Documentation and Extracted Text Files

These are mostly notes, guides, or copied report text.

- [COLLEGE_PROJECT_COMPLETE_REPORT.md](COLLEGE_PROJECT_COMPLETE_REPORT.md): Final project report.
- [report_review_extract.txt](report_review_extract.txt): Extracted report text.
- [senior_report_extracted.txt](senior_report_extracted.txt): Extracted report text from another source.
- [truevoice_report_extract.txt](truevoice_report_extract.txt): Extracted TrueVoice report text.
- [truevoice_pdf_extract.txt](truevoice_pdf_extract.txt): Text copied from a PDF.
- [truevoice_pdf2_extract.txt](truevoice_pdf2_extract.txt): Another PDF text extract.
- [truevoice_pdf9_extract.txt](truevoice_pdf9_extract.txt): Another PDF text extract.

## `data/`

This folder holds the audio dataset used for training and testing.

- [data/manifest.json](data/manifest.json): List of dataset files and labels.
- [data/train/](data/train): Training data.
- [data/train/real/](data/train/real): Real speech samples used for training.
- [data/train/fake/](data/train/fake): Fake speech samples used for training.
- [data/val/](data/val): Validation data.
- [data/val/real/](data/val/real): Real validation samples.
- [data/val/fake/](data/val/fake): Fake validation samples.
- [data/test/](data/test): Test data.
- [data/test/real/](data/test/real): Real test samples.
- [data/test/fake/](data/test/fake): Fake test samples.

## `models/`

This folder stores trained model checkpoints.

- [models/20260305_232325/best_model.pt](models/20260305_232325/best_model.pt): Best model saved during training.
- [models/20260305_232325/latest_checkpoint.pt](models/20260305_232325/latest_checkpoint.pt): Most recent checkpoint saved during training.

## `results/`

This folder stores output files produced after training or evaluation.

- [results/training_summary.json](results/training_summary.json): Training history and summary data.

## `logs/`

- Used for training logs and other runtime logs.

## `src/` Package

This is the core machine learning code.

- [src/__init__.py](src/__init__.py): Marks the folder as a Python package.
- [src/config.py](src/config.py): Stores settings like sample rate, batch size, model type, and folder paths.
- [src/data_loader.py](src/data_loader.py): Loads audio, converts it to spectrograms, and prepares PyTorch datasets.
- [src/model.py](src/model.py): Defines the deepfake detector model.
- [src/trainer.py](src/trainer.py): Contains the training loop, validation logic, and checkpoint saving.
- [src/metrics.py](src/metrics.py): Calculates accuracy, precision, recall, F1, AUC, EER, and confusion-matrix values.
- [src/augmentation.py](src/augmentation.py): Adds spectrogram augmentation like SpecAugment.
- [src/inference.py](src/inference.py): Simple helper for running predictions on audio files.
- [src/utils.py](src/utils.py): Shared helper functions for manifests, checkpoints, seeds, devices, and printing configs.

## `web/` Folder

This folder contains the website version of the project.

- [web/README.md](web/README.md): Overview of the web app.
- [web/DEPLOYMENT.md](web/DEPLOYMENT.md): How to deploy the backend and frontend.

### `web/backend/`

- [web/backend/app.py](web/backend/app.py): Flask API that loads the model and predicts whether uploaded audio is real or fake.
- [web/backend/requirements.txt](web/backend/requirements.txt): Python dependencies for the backend server.

### `web/frontend/`

- [web/frontend/package.json](web/frontend/package.json): Frontend dependencies and npm scripts.
- [web/frontend/vite.config.js](web/frontend/vite.config.js): Vite setup for the React app.
- [web/frontend/index.html](web/frontend/index.html): Main HTML page for the web app.
- [web/frontend/src/main.jsx](web/frontend/src/main.jsx): React entry file that starts the app.
- [web/frontend/src/App.jsx](web/frontend/src/App.jsx): Main user interface for uploading audio, checking results, and viewing history.
- [web/frontend/src/styles.css](web/frontend/src/styles.css): Styling for the website.
- [web/frontend/public/](web/frontend/public): Static files like icons or other assets.

## Simple “What Do I Run?”

- Run [train.py](train.py) to train the model.
- Run [evaluate.py](evaluate.py) to test a saved model checkpoint.
- Run [web/backend/app.py](web/backend/app.py) and the frontend in [web/frontend/](web/frontend) to use the website.

## Short Note

Some files are only there for documentation, status checks, or copied report text. They do not change how the model works.