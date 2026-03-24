param(
    [string]$SourceDir = "data_protocol/nonpair_case_isolated_v1",
    [string]$TargetDir = "data_protocol/p1_pair_case_isolated_v1",
    [string]$OutputDir = "outputs/p1_pair_case_isolated_v1"
)

$ErrorActionPreference = "Stop"

function Build-PairSplit {
    param(
        [string]$CsvPath,
        [string]$SplitName
    )

    $rows = Import-Csv $CsvPath
    $groups = $rows | Group-Object pair_key
    $pairRows = @()

    foreach ($group in $groups) {
        $items = $group.Group
        if ($items.Count -ne 2) {
            continue
        }

        $left = $items | Where-Object { $_.side -eq "L" } | Select-Object -First 1
        $right = $items | Where-Object { $_.side -eq "R" } | Select-Object -First 1

        if (-not $left -or -not $right) {
            continue
        }

        $pairLabel = [int]([int]$left.label -or [int]$right.label)

        $pairRows += [PSCustomObject]@{
            left_path           = $left.image_path
            right_path          = $right.image_path
            label               = $pairLabel
            chromosome_id       = $left.chromosome_id
            case_id             = $left.case_id
            pair_key            = $left.pair_key
            left_single_label   = [int]$left.label
            right_single_label  = [int]$right.label
            left_filename       = $left.filename
            right_filename      = $right.filename
            split               = $SplitName
            abnormal_subtype_id = if ($pairLabel -eq 1) {
                if ([int]$left.label -eq 1) { $left.abnormal_subtype_id } else { $right.abnormal_subtype_id }
            } else {
                ""
            }
            subtype_status      = if ($pairLabel -eq 1) {
                if ([int]$left.label -eq 1) { $left.subtype_status } else { $right.subtype_status }
            } else {
                "normal"
            }
        }
    }

    return $pairRows
}

function Summarize-Split {
    param(
        [array]$Rows,
        [string]$SplitName
    )

    $caseCount = @($Rows | Select-Object -ExpandProperty case_id -Unique).Count
    $abnormalRows = @($Rows | Where-Object { [int]$_.label -eq 1 })
    $abnormalSubtypeCount = @($abnormalRows | Where-Object { $_.abnormal_subtype_id -ne "" } | Select-Object -ExpandProperty abnormal_subtype_id -Unique).Count

    return [PSCustomObject]@{
        split                     = $SplitName
        total_pairs               = @($Rows).Count
        normal_pairs              = @($Rows | Where-Object { [int]$_.label -eq 0 }).Count
        abnormal_pairs            = $abnormalRows.Count
        total_cases               = $caseCount
        abnormal_cases            = @($abnormalRows | Select-Object -ExpandProperty case_id -Unique).Count
        abnormal_subtypes_present = $abnormalSubtypeCount
    }
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$trainRows = Build-PairSplit -CsvPath (Join-Path $SourceDir "train.csv") -SplitName "train"
$valRows = Build-PairSplit -CsvPath (Join-Path $SourceDir "val.csv") -SplitName "val"
$testRows = Build-PairSplit -CsvPath (Join-Path $SourceDir "test.csv") -SplitName "test"

$trainRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $TargetDir "train.csv")
$valRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $TargetDir "val.csv")
$testRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $TargetDir "test.csv")

$splitSummary = @(
    Summarize-Split -Rows $trainRows -SplitName "train"
    Summarize-Split -Rows $valRows -SplitName "val"
    Summarize-Split -Rows $testRows -SplitName "test"
)
$splitSummary | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $OutputDir "split_summary.csv")

$notes = @"
# P1 Pair Case-Isolated v1

Source single-image protocol:

- `$SourceDir/train.csv`
- `$SourceDir/val.csv`
- `$SourceDir/test.csv`

Construction rule:

- group by `pair_key`
- require exactly one `L` and one `R`
- pair label = max(left_single_label, right_single_label)

Known exclusion:

- singleton sex-chromosome samples such as `xOnly` / `yOnly` cannot form homologous pairs
- therefore some X/Y abnormal cases present in N0 single-image protocol are dropped in pair protocol

This pair protocol is case-isolated because it is derived directly from the N0 non-pair case-isolated protocol.
"@

Set-Content -Encoding UTF8 -Path (Join-Path $OutputDir "protocol_notes.md") -Value $notes

Write-Host "Built pair protocol at $TargetDir"
Write-Host "Saved summaries to $OutputDir"
