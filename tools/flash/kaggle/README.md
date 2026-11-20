# Dara Flash drafter on Kaggle

1. `python -I tools/flash/kaggle/make_kit.py` builds `build/flash/kaggle_kit/` (code, tokenizer files, 4,249 seed prompts, a 12-sequence validation pack recorded by Dara).
2. Upload that folder as a private Kaggle dataset named `dara-flash-kit`; upload `dara_flash_kaggle.ipynb` as a notebook (or paste it), attach the dataset, pick GPU T4 x2 and Internet on.
3. Run the cells in order. Cell 3 (validation) must pass before generating data. Distill and train in the same session (data is on temporary disk).
4. Download `drafter_nano_v3.pt`, export with `tools/flash/export_drafter.py`, compare with `drafter_nano_v2.dflash` using `dara.exe --bench ... --flash`.

`python -I tools/flash/kaggle/smoke_test.py --hf-dir models/gemma4-e2b` checks the data path with a tiny random model (no weights needed).
`make_notebook.py` regenerates the notebook from plain Python.
