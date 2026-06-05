# Create data/ directories and print subject paths
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m app init-data
if ($args.Count -ge 1) {
    python -m app paths --subject $args[0]
}
