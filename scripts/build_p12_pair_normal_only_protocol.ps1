$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $projectRoot "data_protocol\p1_pair_case_isolated_v1"
$targetDir = Join-Path $projectRoot "data_protocol\p12_pair_normal_only_v1"
$reportDir = Join-Path $projectRoot "outputs\p12_pair_normal_only_v1"

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$trainRows = Import-Csv (Join-Path $sourceDir "train.csv")
$valRows = Import-Csv (Join-Path $sourceDir "val.csv")
$testRows = Import-Csv (Join-Path $sourceDir "test.csv")

$trainNormal = @($trainRows | Where-Object { [int]$_.label -eq 0 })

$trainNormal |
    Sort-Object chromosome_id, case_id, pair_key |
    Export-Csv -Path (Join-Path $targetDir "train.csv") -NoTypeInformation -Encoding UTF8

$valRows |
    Sort-Object chromosome_id, case_id, pair_key |
    Export-Csv -Path (Join-Path $targetDir "val.csv") -NoTypeInformation -Encoding UTF8

$testRows |
    Sort-Object chromosome_id, case_id, pair_key |
    Export-Csv -Path (Join-Path $targetDir "test.csv") -NoTypeInformation -Encoding UTF8

$summary = @(
    [pscustomobject]@{
        split = "train"
        total_pairs = $trainNormal.Count
        normal_pairs = @($trainNormal | Where-Object { [int]$_.label -eq 0 }).Count
        abnormal_pairs = @($trainNormal | Where-Object { [int]$_.label -eq 1 }).Count
        cases = @($trainNormal | Select-Object -ExpandProperty case_id -Unique).Count
    },
    [pscustomobject]@{
        split = "val"
        total_pairs = $valRows.Count
        normal_pairs = @($valRows | Where-Object { [int]$_.label -eq 0 }).Count
        abnormal_pairs = @($valRows | Where-Object { [int]$_.label -eq 1 }).Count
        cases = @($valRows | Select-Object -ExpandProperty case_id -Unique).Count
    },
    [pscustomobject]@{
        split = "test"
        total_pairs = $testRows.Count
        normal_pairs = @($testRows | Where-Object { [int]$_.label -eq 0 }).Count
        abnormal_pairs = @($testRows | Where-Object { [int]$_.label -eq 1 }).Count
        cases = @($testRows | Select-Object -ExpandProperty case_id -Unique).Count
    }
)

$summary | Export-Csv -Path (Join-Path $reportDir "split_summary.csv") -NoTypeInformation -Encoding UTF8

$notes = @(
    "# P12 Pair Normal-Only Protocol",
    "",
    "Definition",
    "- derived from data_protocol/p1_pair_case_isolated_v1",
    "- train keeps only normal homologous pairs",
    "- val/test stay unchanged and still contain both normal and abnormal pairs",
    "",
    "Goal",
    "- learn chromosome-conditional normal pair manifold only",
    "- detect abnormalities as deviation from normal pair structure",
    "",
    ("- train normal pairs = {0}" -f $summary[0].normal_pairs),
    ("- val total pairs = {0}" -f $summary[1].total_pairs),
    ("- test total pairs = {0}" -f $summary[2].total_pairs)
)
$notes | Set-Content -Path (Join-Path $reportDir "protocol_notes.md") -Encoding UTF8

Write-Host ""
Write-Host "Saved P12 normal-only protocol to: $targetDir"
Write-Host "Saved P12 report to: $reportDir"
