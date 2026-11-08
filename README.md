# Not a Blood Test

Reproducible record for the preprint *Not a Blood Test: Duplicate Leakage, Label Editing, and a 339-Scan Failure of Fingerprint ABO Classification* (Sukkur IBA University, Dadu Campus).

This system does not determine blood group. It must not be used to screen, triage, or type blood. The live log matches the stated ABO letter on 28.0% of 339 scans, which is what the prediction margins already imply (Cohen’s κ = −0.008), and it loses to the constant answer “O” (47.2%).

## What is published here

| Path | Role in the paper |
| --- | --- |
| `features_clean_dataset.csv` | Feature table. 7,576 rows, 29 numeric features, eight folder labels. This is the working copy after the Kaggle set was downloaded, copied, augmented, and partly moved between folders. |
| `feedback_data.jsonl` | Live correction log. 339 events. Each row stores a feature vector, the predicted letter, the stated group, and whether the user accepted the answer. No fingerprint image is stored. |
| `quality_report.json` | Quality-gate counts for that feature table. |
| `api_server.py` | Local FastAPI server: features, forest, rules, and the k-nearest-neighbour memory. |
| `bloodgroup_ABO_calibrated.py` | Descriptor and centroid rules used by the server. |
| `scanner_integration.py` | Optional Suprema BioMini capture path. |
| `frontend/` | React page used during the demonstration. |
| `tools/` | Feature extraction, quality check, and a forest trainer. |
| `paper/reproduce_evaluation.py` | Refits the forest and writes `paper/results.json`. |
| `paper/render_revised_figures.py` | Draws the figures in the preprint from that JSON file. |
| `paper/Not_a_Blood_Test.pdf` | The preprint. |

The raw Kaggle images are not in this repository. Their license is not ours to extend, and the analysis uses the feature table. The 24 centroid images are also withheld: some are named fingerprint scans, and the paper's numbers do not require them.

## What is left out on purpose

`*.joblib` model files are not included. Several local copies are about 200 MB each and are backups of the same forest. The offline accuracies in the paper are a new fit on `features_clean_dataset.csv`, not a reload of those files. The live numbers come from `feedback_data.jsonl`.

To train a local forest, which is gitignored:

```bash
python tools/train_model.py --features features_clean_dataset.csv --out model_rf.joblib
```

## Reproduce the reported numbers

Use Python 3.11 or newer with the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
python paper/reproduce_evaluation.py
python paper/render_revised_figures.py
```

The first script refits a balanced random forest three times (120 trees, depth 20, seed 42) under a random split and a same-print split, and it scores the live log. The second script writes:

- `paper/figures/rev_copies.png`
- `paper/figures/rev_accuracy.png`
- `paper/figures/rev_field.png`
- `paper/figures/rev_chance.png`

Expected headline figures: random-split ABO accuracy 91.2%; same-print ABO accuracy 52.4%; live letter match 28.0%; always answering O on that log 47.2%.

## Run the demonstration locally

```bash
pip install -r requirements.txt
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The page calls `http://127.0.0.1:8000`. Without a local `model_rf.joblib`, the server cannot score a new image with the forest. Train one with the command above, or treat this repository as the record of the study rather than as a deployable tester.

## Data notes

Labels in the feature table are folder labels from a public Kaggle copy, then edited locally. They are not laboratory blood types of the people in the images. Labels in the correction log are what the person at the scanner said. No ethics committee reviewed the study. People were told it was a student demonstration and not a blood test. They agreed to a scan and to state their blood group.

## Citation

Author names are still to be added on the preprint. Until then, cite the repository:

Aleerexaz. *Not a Blood Test.* GitHub, 2026. https://github.com/Aleerexaz/Not_a_Blood_Test
