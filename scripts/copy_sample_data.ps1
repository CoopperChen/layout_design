# Copy subject artifacts from genetic_SHAPE for local pipeline testing.
param(
    [int]$Subject = 2,
    [switch]$IncludeLayouts
)

$src = "D:\Research\genetic_layout_design\genetic_SHAPE\app"
$dst = "D:\Research\layout_design\data"

function Copy-IfExists($from, $to) {
    if (Test-Path $from) {
        New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
        Copy-Item $from $to -Force
        Write-Host "  $to"
    }
}

Write-Host "Subject $Subject"
Copy-IfExists "$src\data\raw\$Subject.stl" "$dst\raw\$Subject.stl"
Copy-IfExists "$src\data\cleaned_scans\$Subject.stl" "$dst\cleaned_scans\$Subject.stl"
Copy-IfExists "$src\data\json\fiducials_$Subject.json" "$dst\json\fiducials_$Subject.json"
Copy-IfExists "$src\data\json\Cz_$Subject.json" "$dst\json\Cz_$Subject.json"
Copy-IfExists "$src\data\json\electrode_positions_$Subject.json" "$dst\json\electrode_positions_$Subject.json"
Copy-IfExists "$src\data\json\initial_terminal_assignments_$Subject.json" "$dst\json\initial_terminal_assignments_$Subject.json"
Copy-IfExists "$src\data\presets\subject1_best_v4.json" "$dst\presets\subject1_best_v4.json"

if ($IncludeLayouts) {
    Copy-IfExists "$src\data\output\applied_v4_s${Subject}_synth_slots.json" "$dst\output\layouts\synth_s$Subject.json"
}

Write-Host "Done. Run: python -m app synthesize --preset subject1_best_v4 --target $Subject"
