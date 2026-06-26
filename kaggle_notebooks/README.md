# Kaggle Training Notebooks

This folder contains Kaggle notebooks for every project spec in this repo.
Run the notebook for the spec, download the generated zip from Kaggle Output,
extract it locally into this repo's `models/` directory, then run the app.

## Notebook Map

| Spec | Notebook | Output zip | Expected local files |
|---|---|---|---|
| Long-Hair Gender Identification | `01_long_hair_gender_identification.ipynb` | `long_hair_gender_models.zip` | `models/age_estimator.keras`, `models/hair_classifier.keras`, `models/gender_predictor.keras`, `models/config.json` |
| Senior Citizen Identification | `02_senior_citizen_identification.ipynb` | `senior_citizen_models.zip` | `models/senior_age_estimator.keras`, `models/senior_gender_predictor.keras`, `models/yolov8n.pt`, `models/senior_config.json` |
| Sign Language Detection | `03_sign_language_detection.ipynb` | `sign_language_models.zip` | `models/sign_language_cnn.keras`, `models/sign_language_config.json` |
| Car Colour Detection | `04_car_colour_detection.ipynb` | `car_colour_models.zip` | `models/colour_classifier.keras`, `models/yolov8n.pt`, `models/car_colour_config.json` |
| Nationality Detection | `05_nationality_detection.ipynb` | `nationality_models.zip` | `models/nationality_detector.keras`, `models/emotion_predictor.keras`, `models/age_estimator.keras`, `models/dress_colour_classifier.keras`, `models/nationality_config.json` |
| Age-Emotion Voice Detection | `06_voice_age_emotion_detection.ipynb` | `voice_models.zip` | `models/voice_gender_classifier.keras`, `models/voice_age_estimator.keras`, `models/voice_emotion_detector.keras`, `models/voice_config.json` |

## General Kaggle Steps

1. Open Kaggle.
2. Create a new notebook.
3. Use `File -> Import Notebook` and upload the required `.ipynb` from this folder.
4. In the right panel, click `Add data` and attach the dataset required by that notebook.
5. In notebook settings, select `Accelerator -> GPU T4 x2` or at least `GPU T4`.
6. Turn internet on if the notebook needs package installation or YOLO weight download.
7. Edit the `DATA_DIR` or dataset path variables in the first code cell if your Kaggle dataset folder names differ.
8. Run all cells.
9. Download the output zip from `/kaggle/working/`.
10. Extract the zip contents into this repo's `models/` directory.
11. Run tests locally with `python -m pytest -q`.

## Dataset Expectations

### UTKFace based notebooks

Use for:
- Long-Hair Gender Identification
- Senior Citizen Identification

Attach a UTKFace dataset. The notebooks search common Kaggle locations, but the final image files must use the UTKFace naming pattern:

```text
{age}_{gender}_{race}_{timestamp}.jpg.chip.jpg
```

Gender in UTKFace filenames is expected as:
- `0` male
- `1` female

### Sign Language Detection

Attach an ASL alphabet dataset with one directory per class:

```text
asl_alphabet_train/
  A/
  B/
  C/
  ...
```

The notebook trains on all class folders it finds.

### Car Colour Detection

Attach a car colour crop dataset with one directory per colour:

```text
car_colour/
  black/
  blue/
  green/
  orange/
  red/
  silver/
  white/
  yellow/
```

The notebook also downloads or saves `yolov8n.pt` so local detection can run.

### Nationality Detection

Attach these Kaggle datasets:

```text
jangedoo/utkface-new
msambare/fer2013
kaiska/apparel-dataset
```

The notebook now prepares the class folders automatically:

- UTKFace filenames are parsed for age and race labels. Race is used as a demo nationality proxy: `3 -> Indian`, `1 -> African`, `0 -> US_American`, `2/4 -> Other`.
- FER2013 folders are normalized, including `fear -> fearful` and `surprise -> surprised`.
- Apparel colour labels are read from colour folders, colour filenames, or CSV columns such as `color`, `colour`, or `baseColour`.

If you have your own cleaner nationality dataset, you can still attach it in directory classification format:

```text
nationality/
  Indian/
  US_American/
  African/
  Other/
emotion/
  happy/
  sad/
  angry/
  surprised/
  neutral/
  fearful/
  disgusted/
dress_colour/
  red/
  blue/
  green/
  black/
  white/
  yellow/
  brown/
  grey/
  orange/
  pink/
  purple/
```

For age estimation, provide a CSV named `age_labels.csv` with:

```text
image_path,age
relative/or/absolute/path.jpg,42
```

  ### Voice Detection

Attach this Kaggle dataset:

```text
rohitzaman/gender-age-and-emotion-detection-from-voice
```

The notebook now supports three input styles:

- a feature CSV with `f0..f27,gender,age,emotion`
- separate statistical feature CSVs for gender, age, and emotion, such as the files in `rohitzaman/gender-age-and-emotion-detection-from-voice`
- a flexible manifest CSV with audio paths plus any available `gender`, `age`, `emotion`, `sex`, or `label` columns
- raw audio folders with common dataset naming patterns such as RAVDESS, TESS, SAVEE, and CREMA-D

If the dataset has a feature CSV, columns can be:

```text
f0,f1,...,f27,gender,age,emotion
```

Where:
- `gender` is `male` or `female`
- `age` is numeric
- `emotion` is one of `happy,sad,angry,fearful,neutral,surprised,disgusted`

If you have raw audio instead, use a manifest CSV with:

```text
path,gender,age,emotion
```

The notebook can also run without a manifest CSV by scanning audio files under `/kaggle/input` and inferring labels from file names, folders, and demographic CSV files when available. When age, gender, or emotion is missing, the notebook prints the defaulted label counts and continues so the project still produces runnable models.

## Local App Commands

After copying models into `models/`:

```powershell
python run_sign_language.py
python run_car_colour.py
python run_nationality.py
python run_voice.py
python run_senior.py --gui
streamlit run streamlit_app.py
```
