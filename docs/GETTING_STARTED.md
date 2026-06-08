# Getting started

## 1. Environment

Use the existing genetic_SHAPE venv (recommended until a dedicated venv is created):

```powershell
$py = "D:\Research\genetic_layout_design\genetic_SHAPE\genetic\Scripts\python.exe"
cd D:\Research\layout_design
& $py -m pip install -e .
```

Or any Python 3.10+ with: `numpy`, `scipy`, `pyvista`, `shapely`, `matplotlib`, `mne`, `pyyaml`.

## 2. Initialize data tree

```powershell
python -m app init-data
python -m app paths --subject 2
```

## 3. Copy a test subject (optional)

```powershell
.\scripts\copy_sample_data.ps1 -Subject 2
```

## 4. Pipeline commands

### Stage A — Preprocess (interactive)

```powershell
$env:LAYOUT_SUBJECT_ID = "2"   # set automatically by CLI
python -m app preprocess --subject 2 --step clear-islands
python -m app preprocess --subject 2 --step fiducials
python -m app preprocess --subject 2 --step cz
python -m app preprocess --subject 2 --step electrodes
python -m app preprocess --subject 2 --step assignments
```

### Stage B — Generate layout (synthesize)

```powershell
python -m app build-assignments --reference 1 --id s1_assignments
python -m app synthesize --assignments s1_assignments --target 2 --visualize
# --preset is an alias for --assignments
```

```powershell
# Or separately (default: 2D PNG + interactive 3D window):
python -m app visualize --applied data/output/layouts/synth_s2.json
python -m app visualize --applied data/output/layouts/synth_s2.json --mode 2d --no-show
python -m app visualize --applied data/output/layouts/synth_s2.json --mode 3d
python -m app visualize --applied ... --save-3d data/output/pics/synth_s2_3d.png  # optional screenshot
```

Outputs: `data/output/pics/synth_s2_2d.png`; 3D opens in PyVista (close window when done).

### Stage C — Polish (optional)

```powershell
python -m app polish --applied data/output/layouts/synth_s2.json --mode gentle
python -m app polish --applied data/output/layouts/synth_s2_repaired.json --mode refine
python -m app polish --applied data/output/layouts/synth_s2_refined.json --mode ga-short --subject 2 --generations 20
```

### Stage D — Postprocess

```powershell
python -m app smooth --applied data/output/layouts/synth_s2.json
python -m app export-bundle --input data/output/smooth/smooth_s2_final.json
python -m app list-electrodes --bundle data/output/bundles/subject_2
python -m app convert-gcode --bundle data/output/bundles/subject_2 --config config/postprocessor/subjects/example.yaml
```

Print config (`config/postprocessor/subjects/*.yaml`) holds physical landmarks `pm`: mount the end-effector on the print head, touch the three marked calibration points on the real head, and record machine XYZ (first point = origin). See `config/postprocessor/README.md`.

**Legacy:** `export-matlab` + `legacy_gcode_examples/gcodeConverter_final14.m` if you still need MATLAB G-code.

## 5. Project layout

See [README.md](../README.md) and [DATA_LAYOUT.md](DATA_LAYOUT.md).
