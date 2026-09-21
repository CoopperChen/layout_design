# 5-axis machine kinematics (MATLAB convention)

Authoritative reference: `gcodeConverter_final14_updated20260506.m` (Postprocessor/matlab).

Python ports: `axis_angles.py`, `tool_offset.py`, `machine_zero.py`, `machine_fk.py`.

## Mechanism

```
                    Z screwbar (vertical)
                         |
    X,Y table -----------+--- C axis (rotate about +Z)
                         |
                    [head block]
                         |
              B axis ----+---- arm (+X at C=0, B=0; −Y at reference C=90, B=0)
                         |
                    printhead tip
```

| Parameter | Value | MATLAB comment |
|-----------|-------|----------------|
| `a_mm` | 180.7 | Horizontal distance **C-axis → tip** (longitudinal config) |
| `d_mm` | 57.59 | Vertical distance **B-axis → tip** |
| `gap_size_mm` | 15 | Standoff along surface normal when printing |
| `c0_deg` / `b0_deg` | 90 / 0 | Pose when **X/Y/Z machine zeros** were set |

## B and C from surface normal (commanded angles)

**C — `findCaxisAngle`:** azimuth of normal’s XY projection vs **+X**; sign from `cross(normal_xy, +X)_z`.

```matlab
c = sign(cross(pVC, cVC)(3)) * (-90 + angle(cVC, pVC));
```

**B — `findBaxisAngle`:** uses **−normal** vs XY projection; sign from **`sign(ny)`**; branch on **`nz >= 0`**.

```matlab
if cVB(3) >= 0
    b = sign(ny) * (angle(-cVB, pVB) - 90);
else
    b = sign(ny) * (270 - angle(-cVB, pVB));
end
```

Examples (outward normal):

| Normal | B | C |
|--------|---|---|
| +Y | +90° | ≈ 0° |
| −Y | −90° | ≈ 0° |
| +X / −X | 0° | ≈ 0° (degenerate) |
| [1,1,0] in XY | +90° | +45° |

**B is not used** in the `a` arm offset loop — only **C** (see below). Both are written to G-code.

## Machine zero (`c0=90`, `b0=0`)

Applied to scalp contact points **before** axis angles and tool offset:

```matlab
g(:,2) = g(:,2) - a;
g(:,3) = g(:,3) - (d + calgapZ);
```

## Tool offset arm term (sin/cos C)

From gcodeConverter lines 402–410 (`tool_offset.py`):

```
dX = nx * t - |cos(C)| * a
dY = ny * t + sin(C) * a
```

`t = d + gap`. Positive **C** moves the arm contribution from **−X** toward **+Y** (right-hand about **+Z**):

| C | Arm XY added to programmed coords |
|---|-----------------------------------|
| 0° | ΔX = **−a**, ΔY = 0 |
| +90° | ΔX = 0, ΔY = **+a** |
| −90° | ΔX = 0, ΔY = **−a** |

Z uses `cross(normal, [0,0,1])` (see `tool_offset.py`).

## Structural FK (`machine_fk.py`)

Rigid print head (arm and tool always perpendicular, fixed lengths):

```
center = (X, Y, Z)              # C pivot (G-code XYZ)
arm    = a · Rz(−C) · [1, 0, 0] # C=0 → +X; C=90 → −Y
B_pivot = center + arm
tool_dir = tool at B=0 is −Z projected ⊥ arm; then rotate +B about arm
tip = B_pivot + d · tool_dir
```

| Joint | Action |
|-------|--------|
| **C** | Rotate about screwbar (+Z); rigid arm uses **Rz(−C) · [1, 0, 0]** (C=0 → **+X**, C=90 → **−Y**) |
| **C–B** | Fixed length `a_mm` |
| **B** | Rotate tool about C–B arm at B pivot |
| **B–tip** | Fixed length `d_mm`, ⊥ C–B arm, rigidly attached |

Reference poses (commanded angles, structural FK):

| B | C | C→B arm | B→tip |
|---|---|---------|-------|
| 0 | 0 | **+X** | **−Z** |
| 90 | 0 | **+X** | **−Y** |
| 0 | 90 (`c0`, `b0`) | **−Y** | **−Z** |

### Machine zero (`c0=90`, `b0=0`) in machine frame

When X/Y/Z machine zeros were set:

```
C pivot (X0 Y0 Z0)     = (0, 0, 0)
nozzle tip             = (0, −a, −d)
central landmark       = (0, −a, −(d + calgap_z))
```

``pm[0] = [0, 0, 0]`` is the **central landmark** in the landmark-measurement frame. **simulate-gcode** shifts mesh and landmarks into **machine frame** via ``registration_to_machine_frame()`` so C pivot zero is ``(0,0,0)`` and central is ``(0, −a, −(d + calgap_z))`` — the same frame as G-code X,Y,Z.

When the tip touches the central landmark at ``(b0, c0)``, the C pivot reads ``(0, 0, −calgap_z)`` in machine frame — not ``(0, 0, 0)``.

## `correct_flip`

If **C** crosses zero with **|C| > 20°** on both sides, negate **B and C** on a prefix of the path (MATLAB lines 359–386).

Use `arm_offset_xy_matlab(c, a)` for the sin/cos arm term in the postprocessor (not the simulator rigid chain).

## Print feedrate (constant tip speed)

Controller **F** commands **C-pivot (XYZ)** speed. For each print segment:

```
F = V_tip · ||ΔC_pivot|| / ||Δtip_FK||
```

with ``V_tip = speed_mm_min``, clamped to ``max_speed_mm_min``.

``Δtip_FK`` is the rigid forward-kinematics tip travel for the programmed ``(X,Y,Z,B,C)`` poses — not the raw scalp chord — so B/C swings that move the tip are included. Using the path chord alone sets **F too high** during orientation changes (tip serpentine at max feed).

## Simulator (`simulate-gcode`)

**Runtime (confirmed on machine):** controller moves **C pivot** to programmed **X,Y,Z**; **B,C** command the arm/tool. No postprocessor runs at print time.

**Forward viewer:** `forward_states_from_gcode()` in `app/simulator/kinematics/machine_execute.py`:

```
C = (X, Y, Z)                         # G-code XYZ
B = C + a · Rz(−C) · [1, 0, 0]      # C=0 → +X; C=90 → −Y
tip = B + d · tool_dir(B, C)          # tool ⊥ arm; −Z at B=0
```

**Offline (`convert-gcode`):** scalp trace → `machine_zero` → `tool_offset` → writes **X,Y,Z,B,C**. That compensation is baked into G-code; the machine executes it literally.

**Trace coordinate semantics:**

| Trace | Bundle XYZ meaning | Tool offset in `convert-gcode` |
|-------|-------------------|--------------------------------|
| Interconnect | Scalp contact (material lands on surface) | `d + gap_size_mm` |
| Electrode | Pad plane at `surface + gap_size_mm` along normal | `d` only (gap already in geometry) |

Electrode disks are built at export (`export-bundle`) as coplanar perimeter zigzags in that offset plane; `convert-gcode --trace electrode` must not apply gap twice.

**Postprocessor decode** (`decode_postprocessor_paths`, `--verbose` only): inverts offline transforms in **landmark frame** (same as `convert-gcode`), then shifts scalp/tip to **machine frame** for comparison with forward FK and the viewer mesh. Standoff median should be ≈ ``gap_size_mm``; FK vs decode may still differ at arbitrary B,C.

### Machine-frame registration

`simulate-gcode` uses the same `scan2phys` as `convert-gcode`, then shifts into **controller machine frame**:

```
registration_to_machine_frame(p) = p + (0, −a, −(d + calgap_z))
```

Mesh, landmarks, and G-code paths then share one frame (C at origin). Use the same `--pm-file`, `--rot0y`, and `--rot0z` as `convert-gcode`.

**Viewer layers:** `m` mesh · `l` landmarks · `o` machine-zero reference · `c` C pivot path · `t` tip · `a` arm. Step HUD shows X, Y, Z, B, C.
