# Print registration (`pm`)

Per-subject YAML files store **physical landmarks only** — measured at print time on the real head with the CNC.

These three points (`landmark_central`, `landmark_left`, `landmark_back`) pair with the **digital** calibration picks in `data/json/fiducials_{id}.json`. `convert-gcode` uses `scan2phys` to rigidly align scan geometry to the machine so G-code lands on the physical head.

Trace type and channel selection are **CLI flags** on `convert-gcode`, not stored in the pm YAML.

---

## Two landmark systems (do not confuse)

| | Digital (scan) | Physical (print / `pm`) |
|--|----------------|-------------------------|
| **When** | Preprocess (`fiducials` step) | At the machine, before convert-gcode |
| **Where** | `data/json/fiducials_{id}.json` | `config/postprocessor/subjects/subject_{id}.yaml` |
| **How** | Right-click + Space on textured OBJ | Jog tip + keyboard (`record-pm`) or edit YAML |
| **Keys** | `landmark_central`, `landmark_left`, `landmark_back` | Same three, as rows of `physical_landmarks_mm` |
| **Frame** | Scan / mesh coordinates | Measurement frame: central = origin |

---

## Recommended: automated capture (`record-pm`)

`python -m app run` includes a **`record-pm`** stage after the empty `print-config` scaffold and **before** `gcode`. It is skipped when pm is already measured (unless `--force-record-pm`).

Standalone / re-capture:

```powershell
python -m app record-pm --subject 4
```

Reads the live CNC **work** DRO over UDP (same Mach4 publisher used by Orbbec CNC streaming). You jog the tip to each landmark and confirm with the keyboard.

### 1. Mach4 work-pose publisher

1. Copy [`scripts/mach4_work_pose_publisher.lua`](../../scripts/mach4_work_pose_publisher.lua) into your Mach4 profile macros folder, or paste it into the profile **PLC** script.
2. Edit at the top of the Lua file:
   - `TARGET_IP` — PC running `record-pm` (use `127.0.0.1` if Mach4 and this repo are on the same machine)
   - `TARGET_PORT` — default `62100`
3. Ensure **LuaSocket** is available to Mach4 (`socket.dll` under Mach4’s Lua API tree).
4. Call `PublishWorkPoseUdp()` every PLC cycle (or from a timer). The script uses `mc.mcAxisGetPos()` — **active work coordinates** (G54/G55… DRO), not machine coordinates.

**Packet format** (JSON over UDP):

```json
{"coord":"work","units":"mm","x":117.44,"y":116.58,"z":-14.2,"b":61.57,"c":20.72}
```

- `coord` must be `"work"` (also accepts `"g54"` / `"active"`).
- Prefer `"units":"mm"`. Inches are scaled by 25.4 if `"units":"in"`.

### 2. Mount and jog

1. Mount the end-effector on the printhead (same tip used for printing).
2. Jog in **work** coordinates so the tip can touch the three physical markers on the head:
   - **landmark_central**
   - **landmark_left**
   - **landmark_back**
3. Keep B/C at a convenient touch pose (often near machine zero for the first capture). The tool records B/C for audit but **pm registration uses XYZ only**.

### 3. Run the recorder

```bash
python -m app record-pm --subject 4
# Overwrite an existing YAML:
python -m app record-pm --subject 4 --force
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--subject` | *(required)* | Subject id → `subject_{id}.yaml` |
| `--port` | `62100` | UDP listen port |
| `--bind-ip` | `0.0.0.0` | Bind address |
| `--stale-ms` | `500` | Ignore packets older than this |
| `--output` | auto | Override output path |
| `--force` | off | Overwrite existing file |

**Live line** shows capture status and the current DRO, e.g.:

```text
>✓1:central  ·2:left  ·3:back  | work pose: live (40 ms) | X=… Y=… Z=…  B=… C=…
```

| Key | Action |
|-----|--------|
| **Enter** / **Space** | Capture current work XYZ; if all three are captured, save YAML |
| **1** / **2** / **3** | Jump to central / left / back |
| **n** / **p** | Next / previous landmark |
| **s** | Save YAML (all three must be captured) |
| **q** / **Esc** | Abort (nothing written) |

Capture order is normally central → left → back; you can re-capture any point with `1`/`2`/`3` then Enter.

### 4. What gets written

File: `config/postprocessor/subjects/subject_{id}.yaml`

```yaml
physical_landmarks_mm:
  - [0, 0, 0]           # landmark_central (always origin)
  - [dx, dy, dz]        # landmark_left  − central (work XYZ)
  - [dx, dy, dz]        # landmark_back  − central (work XYZ)

capture:                # audit only — not used by convert-gcode
  raw_work_xyz_mm: ...
  work_bc_deg: ...
  udp_port: 62100
```

**Math:** if the DRO at the three touches is \(p_0, p_1, p_2\), then

\[
\mathrm{pm}[0] = (0,0,0),\quad
\mathrm{pm}[1] = p_1 - p_0,\quad
\mathrm{pm}[2] = p_2 - p_0
\]

You do **not** need to zero work coordinates at central. Relative storage keeps the measurement frame consistent with `scan2phys`.

### 5. Troubleshooting

| Symptom | Check |
|---------|--------|
| `work pose: waiting` | Mach4 Lua running? `TARGET_IP` / port match? Firewall blocking UDP? |
| `work pose: stale` | Publisher stopped or PLC not calling `PublishWorkPoseUdp` often enough |
| Capture refused | Need a **live** packet; wait until status shows `live` |
| Wrong registration later | Same tip / same B/C habit as print; digital picks must be the same three markers; use matching `--rot0y`/`--rot0z` on convert and simulate |

---

## Manual scaffold + edit

```bash
python -m app init-print-config --subject 4
# → config/postprocessor/subjects/subject_4.yaml
```

Edit `physical_landmarks_mm` by hand:

1. Touch **landmark_central** → enter `[0, 0, 0]` (or subtract this DRO from the other two if you record absolute work XYZ).
2. Touch **landmark_left** / **landmark_back** → enter XYZ **relative to central**.

Prefer `record-pm` when Mach4 UDP is available.

---

## Coordinate frames

| Frame | Central landmark | C pivot @ machine zero |
|-------|------------------|------------------------|
| **pm measurement** (`convert-gcode` offline registration) | `(0, 0, 0)` | not at origin |
| **Controller machine** (runtime G-code, `simulate-gcode`) | `(0, −a, −(d+calgap_z))` | `(0, 0, 0)` |

Programmed G-code **X,Y,Z** are the **C pivot**. Tip touch ≠ pivot DRO; that is expected. `record-pm` stores the work DRO at tip contact and treats central as the measurement origin — the same convention as the previous manual pm workflow.

See [docs/MACHINE_KINEMATICS.md](../../docs/MACHINE_KINEMATICS.md).

---

## Machine geometry

Physical layout and B/C conventions follow **`gcodeConverter_final14.m`**: `findCaxisAngle` / `findBaxisAngle`, sin/cos(C) arm offset for **offline** G-code writing, rigid FK for **runtime** simulation.

Shared config: `machine_default.yaml` (`a_mm`, `d_mm`, `calgap_z_mm`, `c0_deg`, `b0_deg`).

---

## Convert to G-code

```bash
# Auto-loads config/postprocessor/subjects/subject_4.yaml from bundle subject id
python -m app convert-gcode --bundle data/output/bundles/subject_4

# Single channel, electrode pads
python -m app convert-gcode --bundle data/output/bundles/subject_4 \
  --trace electrode --electrode Fp1

# Explicit pm file
python -m app convert-gcode --bundle ... --pm-file config/postprocessor/subjects/subject_4.yaml
```

| CLI flag | Default | Purpose |
|----------|---------|---------|
| `--trace` | `both` | `both`, `interconnect`, or `electrode` |
| `--electrode` | `all` | `all`, channel name, or 1-based index |
| `--rot0y`, `--rot0z` | `0` | Optional head rotation on bed |

---

## Simulate G-code (3D viewer)

```bash
python -m app simulate-gcode \
  --gcode data/output/gcode/subject_4_post/allinterconnects.txt \
  --bundle data/output/bundles/subject_4
```

Use the **same** `pm` YAML and `--rot0y` / `--rot0z` as `convert-gcode`. Digital landmarks come from the bundle (fiducial picks) — not from `record-pm`.

---

## Files

| File | Purpose |
|------|---------|
| `machine_default.yaml` | Printer geometry and speeds (shared) |
| `subjects/subject_{id}.yaml` | **pm** for that subject (+ optional `capture:` audit) |
| `subjects/example.yaml` | Empty template |
| `subjects/synthetic.yaml` | Contract tests only |
| [`scripts/mach4_work_pose_publisher.lua`](../../scripts/mach4_work_pose_publisher.lua) | Mach4 → UDP work DRO publisher |

| Code | Role |
|------|------|
| `app/postprocess/cnc_work_pose.py` | UDP client / JSON parse |
| `app/postprocess/record_pm.py` | Keyboard capture loop |
| `app/postprocess/print_config.py` | Scaffold + load/save pm YAML |
