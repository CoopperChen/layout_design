# Print registration (`pm`)

`record-pm` writes `data/subjects/{id}/pm.yaml`. The digital triangle is `calibration` in `data/subjects/{id}/layout.json`. `gcode` uses `scan2phys` to align the scan to the machine.

| | Digital (scan) | Physical (print) |
|--|----------------|------------------|
| **When** | `layout` | At the machine, before `gcode` |
| **Where** | `layout.json` → `calibration` | `data/subjects/{id}/pm.yaml` |
| **How** | Right-click on the textured head | Jog the tip (`record-pm`) |
| **Keys** | `central`, `left`, `back` | Rows of `physical_landmarks_mm` |
| **Frame** | Scan coordinates | Central at the origin |

`python -m app run` skips `record-pm` when `pm.yaml` is already measured. Pass `--force-record-pm` to capture again. A standalone capture refuses to overwrite unless `--force`:

```powershell
python -m app record-pm --subject 2
python -m app record-pm --subject 2 --force
```

Reads the live CNC **work** DRO over UDP (same Mach4 publisher used by Orbbec CNC streaming). You jog the tip to each landmark and confirm with the keyboard.

### 1. Mach4 work-pose publisher

1. Copy [`scripts/mach4_work_pose_publisher.lua`](../scripts/mach4_work_pose_publisher.lua) into your Mach4 profile macros folder, or paste it into the profile **PLC** script.
2. Confirm `TARGETS` in the Lua script (default: dual publish to the tracking PC):
   - `192.168.208.10:62100` — Orbbec head tracking / `orbbec-head-stream-cnc`
   - `192.168.208.10:62101` — this repo’s `record-pm`
3. Both apps can run at once because Mach4 sends the same JSON to **two ports** (one unicast port cannot be shared).
4. Ensure **LuaSocket** is available to Mach4 (`socket.dll` under Mach4’s Lua API tree).
5. Call `PublishWorkPoseUdp()` **and** `PollCncCommandUdp()` every PLC cycle. Pose uses `mc.mcAxisGetPos()` — **active work coordinates** (G54/G55… DRO), not machine coordinates. Commands bind **`127.0.0.1:62110`** only (one datagram per PLC cycle).

**Packet format** (JSON over UDP):

```json
{"coord":"work","units":"mm","x":117.44,"y":116.58,"z":-14.2,"b":61.57,"c":20.72}
```

- `coord` must be `"work"` (also accepts `"g54"` / `"active"`).
- Prefer `"units":"mm"`. Inches are scaled by 25.4 if `"units":"in"`.

### Load and run G-code from Python (`cnc`)

Same Lua file also listens on **`127.0.0.1:62110`** for JSON commands from this PC (not the LAN). After `gcode`:

```powershell
python -m app cnc status
python -m app cnc home --confirm
python -m app cnc scan --confirm
python -m app cnc load --subject 2
python -m app cnc start --confirm
python -m app cnc hold
python -m app cnc stop
```

`start`, `home --confirm`, and `scan --confirm` require `--confirm`. `load`/`start`/`home --confirm`/`scan --confirm` run only when the machine is enabled and idle. This is **not** part of `python -m app run`.

`home` loads [`scripts/mach4_home_bc.txt`](../scripts/mach4_home_bc.txt) (B/C work-offset zeroing). `scan` loads [`scripts/mach4_scan_head.txt`](../scripts/mach4_scan_head.txt) (3D head-scan path). Without `--confirm` they only load the file.

Command packet (Python → Mach4):

```json
{"id":"…","cmd":"load","path":"D:\\Research\\Printer\\layout_design\\data\\subjects\\2\\electrodes.gcode"}
```

`home`/`scan` are Python helpers that `load` a bundled script (and `start` if `--confirm`).

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
| `--subject` | *(required)* | Subject id → `data/subjects/{id}/pm.yaml` |
| `--port` | `62101` | UDP listen port (`62100` is reserved for Orbbec) |
| `--bind-ip` | `0.0.0.0` | Bind address |
| `--stale-ms` | `500` | Ignore packets older than this |
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

File: `data/subjects/{id}/pm.yaml`

```yaml
physical_landmarks_mm:
  - [0, 0, 0]           # landmark_central (always origin)
  - [dx, dy, dz]        # landmark_left  − central (work XYZ)
  - [dx, dy, dz]        # landmark_back  − central (work XYZ)

capture:                # audit only — not used by gcode
  raw_work_xyz_mm: ...
  work_bc_deg: ...
  udp_port: 62101
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
| `work pose: waiting` | Mach4 Lua running? `TARGETS` include this PC:`62101`? Firewall blocking UDP? |
| `work pose: stale` | Publisher stopped or PLC not calling `PublishWorkPoseUdp` often enough |
| `cnc command UDP: bind failed ... already in use` | Leftover bind from an old PLC compile, or two copies of this script. **Disable / Enable** Mach4, then paste this file **once**. Do not close the command socket every PLC cycle. |
| `Mach4 is not listening` / no ACK | PLC is not calling `PollCncCommandUdp()`, or the bind failed. First run `python -m app cnc status`. Mach4 runs the PLC script top-to-bottom each cycle — keep the listener socket in the global `LayoutDesignCncCmdSock` (current script does this). |
| Capture refused | Need a **live** packet; wait until status shows `live` |
| Wrong registration later | Same tip / same B/C habit as print; digital picks must be the same three markers; use matching `--rot0y`/`--rot0z` on `gcode` and `simulate` |

---

## Coordinate frames

| Frame | Central landmark | C pivot @ machine zero |
|-------|------------------|------------------------|
| **pm measurement** (`gcode` registration) | `(0, 0, 0)` | not at origin |
| **Controller machine** (runtime G-code, `simulate`) | `(0, −a, −(d+calgap_z))` | `(0, 0, 0)` |

Programmed G-code **X,Y,Z** are the **C pivot**. Tip touch ≠ pivot DRO; that is expected. `record-pm` stores the work DRO at tip contact and treats central as the measurement origin — the same convention as the previous manual pm workflow.

See [docs/MACHINE_KINEMATICS.md](MACHINE_KINEMATICS.md).

---

## Machine geometry

Physical layout and B/C conventions follow **`gcodeConverter_final14.m`**: `findCaxisAngle` / `findBaxisAngle`, sin/cos(C) arm offset for **offline** G-code writing, rigid FK for **runtime** simulation.

Shared config: `config/machine.yaml` (`a_mm`, `d_mm`, `calgap_z_mm`, `c0_deg`, `b0_deg`).

---

## G-code and simulate

```powershell
python -m app gcode --subject 2
python -m app gcode --subject 2 --electrode Cz
python -m app simulate --subject 2
```

Digital landmarks come from `layout.json`. Physical landmarks come from `pm.yaml`. Use the same `--rot0y` / `--rot0z` on both commands.

[`scripts/mach4_work_pose_publisher.lua`](../scripts/mach4_work_pose_publisher.lua) publishes the work DRO and listens for load/run commands. Capture and CNC control live in `app/machine/`. Landmark load and save live in `app/subject.py`.
