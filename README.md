# ML Projects Suite

This repository contains a complete multi-spec machine learning project suite with
training notebooks, inference code, desktop launchers, tests, and a single
Streamlit dashboard.

Repository:

```text
https://github.com/Ilham0310/ML_PROJECTS.git
```

## Projects Included

| No. | Project | Main Techniques | Runtime Entry |
|---|---|---|---|
| 1 | Long-Hair Gender Identification | MobileNetV2 age, hair-length, and gender models | `streamlit_app.py`, `app.py` |
| 2 | Senior Citizen Identification | YOLOv8 person detection, age and gender models | `run_senior.py` |
| 3 | Sign Language Detection | MediaPipe hand detection, CNN classifier, image and webcam modes | `run_sign_language.py` |
| 4 | Car Colour Detection | YOLOv8 car/person detection, colour CNN, annotation | `run_car_colour.py` |
| 5 | Nationality Detection | CNN nationality, emotion, age, dress-colour routing | `run_nationality.py` |
| 6 | Age-Emotion Voice Detection | Audio validation, 28-dimensional feature extraction, dense models | `run_voice.py` |

## What Is Committed

The repository contains only source files needed to reproduce and run the
project:

- application and inference code under `src/`
- launchers such as `run_car_colour.py`, `run_nationality.py`, and `run_voice.py`
- training scripts
- Kaggle training notebooks under `kaggle_notebooks/`
- tests under `tests/`
- Streamlit dashboard
- bundled dashboard sample pack under `data/dashboard_samples/`
- dependency file and documentation

The repository intentionally does **not** include:

- datasets
- trained `.keras` model files
- YOLO `.pt` model files
- Kaggle output zip files
- local virtual environments
- cache folders
- personal machine paths or private data

These large or generated artifacts are ignored by `.gitignore`.

## Repository Structure

```text
.
|-- app.py
|-- streamlit_app.py
|-- submission_dashboard.py
|-- train.py
|-- train_nationality.py
|-- train_sign_language.py
|-- train_voice_models.py
|-- run_car_colour.py
|-- run_nationality.py
|-- run_senior.py
|-- run_sign_language.py
|-- run_voice.py
|-- kaggle_notebooks/
|   |-- 01_long_hair_gender_identification.ipynb
|   |-- 02_senior_citizen_identification.ipynb
|   |-- 03_sign_language_detection.ipynb
|   |-- 04_car_colour_detection.ipynb
|   |-- 05_nationality_detection.ipynb
|   |-- 06_voice_age_emotion_detection.ipynb
|   `-- README.md
|-- src/
|   |-- annotation/
|   |-- audio/
|   |-- car_colour/
|   |-- classification/
|   |-- data/
|   |-- detection/
|   |-- gui/
|   |-- inference/
|   |-- models/
|   |-- nationality/
|   |-- senior/
|   |-- sign_language/
|   `-- utils/
|-- tests/
|-- models/
|   `-- .gitkeep
|-- data/
|   `-- dashboard_samples/
|       |-- samples_manifest.json
|       `-- 30 sample media files
|-- requirements.txt
`-- README.md
```

## Environment Setup

Python 3.10 to 3.12 is recommended.

```powershell
git clone https://github.com/Ilham0310/ML_PROJECTS.git
cd ML_PROJECTS

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Required Datasets

Use these exact Kaggle datasets when running the training notebooks.

| Project | Kaggle Dataset Slug |
|---|---|
| Long-Hair Gender Identification | `jangedoo/utkface-new` |
| Senior Citizen Identification | `jangedoo/utkface-new` |
| Sign Language Detection | `grassknoted/asl-alphabet` |
| Car Colour Detection | `landrykezebou/vcor-vehicle-color-recognition-dataset` |
| Nationality Detection | `jangedoo/utkface-new`, `msambare/fer2013`, `kaiska/apparel-dataset` |
| Age-Emotion Voice Detection | `rohitzaman/gender-age-and-emotion-detection-from-voice` |

UTKFace is used for face-age/gender tasks and as a nationality proxy in the
nationality notebook. It cannot train sign-language, car-colour, or voice models.

## Training On Kaggle

Kaggle training is the recommended path because several models are expensive on
CPU.

1. Open Kaggle.
2. Create a new notebook.
3. Use `File -> Import Notebook`.
4. Upload the relevant notebook from `kaggle_notebooks/`.
5. Click `Add data`.
6. Add the dataset slugs listed above.
7. Set accelerator to `GPU T4` or `GPU T4 x2`.
8. Turn internet on when YOLO download or package install is needed.
9. Run all cells.
10. Download the output zip from `/kaggle/working/`.

Expected notebook outputs:

| Notebook | Output Zip |
|---|---|
| `01_long_hair_gender_identification.ipynb` | `long_hair_gender_models.zip` |
| `02_senior_citizen_identification.ipynb` | `senior_citizen_models.zip` |
| `03_sign_language_detection.ipynb` | `sign_language_models.zip` |
| `04_car_colour_detection.ipynb` | `car_colour_models.zip` |
| `05_nationality_detection.ipynb` | `nationality_models.zip` |
| `06_voice_age_emotion_detection.ipynb` | `voice_models.zip` |

Place the downloaded zip files in the repository root. Do not commit them.

## Model Files Expected Locally

The dashboard can extract each zip into an isolated folder under
`models/submission/`. This avoids filename collisions such as multiple
`age_estimator.keras` files.

Expected files per spec:

| Project | Expected Files |
|---|---|
| Long-Hair Gender | `age_estimator.keras`, `hair_classifier.keras`, `gender_predictor.keras`, `config.json` |
| Senior Citizen | `senior_age_estimator.keras`, `senior_gender_predictor.keras`, `yolov8n.pt`, `senior_config.json` |
| Sign Language | `sign_language_cnn.keras`, `sign_language_config.json` |
| Car Colour | `colour_classifier.keras`, `yolov8n.pt`, `car_colour_config.json` |
| Nationality | `nationality_detector.keras`, `emotion_predictor.keras`, `age_estimator.keras`, `dress_colour_classifier.keras`, `nationality_config.json` |
| Voice | `voice_gender_classifier.keras`, `voice_age_estimator.keras`, `voice_emotion_detector.keras`, `voice_config.json` |

## Running The Submission Dashboard

After the six model zips are in the repository root, run:

```powershell
.venv\Scripts\streamlit.exe run submission_dashboard.py
```

Then open:

```text
http://localhost:8501
```

If port 8501 is busy, Streamlit will show the alternate URL in the terminal.

Dashboard flow:

- the left sidebar lists only the six project specs
- each spec has `Use sample` and `Upload file` input modes when local samples are available
- each spec has a single `Run` button
- outputs are shown directly as metrics, tables, images, or audio-route text
- model files are extracted automatically into `models/submission/`

Bundled demo samples are committed under `data/dashboard_samples/`. A fresh
clone immediately shows `Use sample` in the dashboard for all six specs. The
folder contains five samples per spec plus `samples_manifest.json`; datasets,
model files, uploads, and Kaggle output zips are still ignored.

Input and output expectations:

| Spec | Input | Expected Output |
|---|---|---|
| Long-Hair Gender | single face image | gender label, confidence, estimated age, age route |
| Senior Citizen | image frame with visible people | annotated people, ages, genders, senior decisions |
| Sign Language | ASL hand/sign image | sign label and confidence |
| Car Colour | traffic or car image | annotated image, car count, person count, car colours |
| Nationality | single face image | nationality, emotion, optional age and dress colour |
| Voice Age Emotion | WAV, MP3, FLAC, or OGG voice clip | gender route, age, senior status, optional emotion |

## Running Individual Applications

After the required models are available locally:

```powershell
python run_sign_language.py
python run_car_colour.py
python run_nationality.py
python run_voice.py
python run_senior.py --gui
streamlit run streamlit_app.py
```

For senior citizen CLI processing:

```powershell
python run_senior.py --source sample_video.mp4 --output results.csv --format csv
```

For webcam mode:

```powershell
python run_senior.py --source 0 --gui
```

## Running Tests

```powershell
python -m pytest -q
```

The current local verification passed with:

```text
339 passed, 9 skipped
```

## Project Details

### Long-Hair Gender Identification

This spec predicts gender from a face image. The system first estimates age.
For people aged 20 to 30 inclusive, gender is intentionally routed through the
hair-length classifier:

```text
long hair  -> Female
short hair -> Male
low confidence -> Undetermined
```

For ages outside 20 to 30, the regular gender predictor is used.

### Senior Citizen Identification

This spec detects people with YOLOv8 and runs age and gender prediction on each
person crop. The senior router classifies a person as senior when age is above
60 and confidence is high enough. It can run in GUI mode, CLI video mode, or as
a single-frame dashboard demo.

### Sign Language Detection

This spec supports two modes:

- image upload mode
- webcam mode

MediaPipe locates the hand region in GUI/webcam usage. The CNN classifier then
predicts the ASL class from the preprocessed hand crop.

### Car Colour Detection

YOLOv8 detects cars and people. Each detected car crop is passed to a colour CNN.
The annotation engine draws boxes and labels and returns object counts.

Supported colour labels:

```text
black, blue, green, orange, red, silver, white, yellow
```

### Nationality Detection

The nationality pipeline detects one face, predicts nationality and emotion, and
then conditionally routes to age or dress-colour models:

- Indian and US/American: age is estimated
- Indian and African: dress colour is estimated
- all valid inputs: nationality and emotion are returned

The Kaggle notebook uses UTKFace race labels as a practical nationality proxy
when no public clean nationality-face dataset is available.

### Age-Emotion Voice Detection

The voice pipeline accepts WAV, MP3, FLAC, or OGG files of at least one second.
It extracts a 28-dimensional acoustic feature vector and routes as follows:

1. classify gender
2. if female, return `Upload male voice`
3. if male, estimate age
4. if male and age is above 60, estimate emotion

## Reproducibility Notes

- Kaggle notebooks set fixed random seeds where practical.
- Model zips are generated artifacts and are intentionally excluded from Git.
- Datasets must be downloaded from Kaggle by the person reproducing the project.
- The dashboard includes a small committed demo sample pack and also supports
  files uploaded by the user.

## Privacy And Repository Hygiene

This repository is prepared for public submission:

- no personal local paths are required
- no private data is committed
- no virtual environment is committed
- no model weights or datasets are committed
- generated caches are ignored
- only source, notebooks, tests, and documentation are intended for GitHub

## License

This project is for academic and educational use. See `LICENSE` for the license
included with the repository.
