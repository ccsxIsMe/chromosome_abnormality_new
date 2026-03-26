$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $projectRoot "data_protocol\p1_pair_case_isolated_v1"
$outDir = Join-Path $projectRoot "data_protocol\p7_pair_seen_alltrain_v1"
$reportDir = Join-Path $projectRoot "outputs\p7_pair_seen_alltrain_v1"

function Build-LabelMap {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Rows
    )

    $classNames = @(
        $Rows |
        Where-Object { [int]$_.label -eq 1 } |
        Select-Object -ExpandProperty abnormal_subtype_id -Unique |
        Sort-Object
    )

    $labelMap = @()
    for ($idx = 0; $idx -lt $classNames.Count; $idx++) {
        $name = $classNames[$idx]
        $labelMap += [pscustomobject]@{
            label = $idx
            class_name = $name
            chromosome_id = $name.Split("::")[0]
        }
    }
    return $labelMap
}

$trainSource = Import-Csv (Join-Path $sourceDir "train.csv")
$valSource = Import-Csv (Join-Path $sourceDir "val.csv")
$testSource = Import-Csv (Join-Path $sourceDir "test.csv")

$labelMap = Build-LabelMap -Rows $trainSource
$labelLookup = @{}
foreach ($row in $labelMap) {
    $labelLookup[$row.class_name] = [int]$row.label
}

function Convert-Split {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Rows,
        [Parameter(Mandatory = $true)]
        [string]$SplitName
    )

    $converted = @(
        $Rows |
        Where-Object {
            [int]$_.label -eq 1 -and
            $_.subtype_status -eq "seen" -and
            $labelLookup.ContainsKey($_.abnormal_subtype_id)
        } |
        ForEach-Object {
            $_.label = $labelLookup[$_.abnormal_subtype_id]
            $_ | Add-Member -NotePropertyName multiclass_label -NotePropertyValue $_.label -Force
            $_ | Add-Member -NotePropertyName protocol_split -NotePropertyValue $SplitName -Force
            $_ | Add-Member -NotePropertyName class_name -NotePropertyValue $_.abnormal_subtype_id -Force
            $_
        }
    )
    return $converted
}

$trainRows = Convert-Split -Rows $trainSource -SplitName "train"
$valRows = Convert-Split -Rows $valSource -SplitName "val"
$testRows = Convert-Split -Rows $testSource -SplitName "test"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$trainRows | Sort-Object label, case_id, pair_key | Export-Csv -Path (Join-Path $outDir "train.csv") -NoTypeInformation -Encoding UTF8
$valRows | Sort-Object label, case_id, pair_key | Export-Csv -Path (Join-Path $outDir "val.csv") -NoTypeInformation -Encoding UTF8
$testRows | Sort-Object label, case_id, pair_key | Export-Csv -Path (Join-Path $outDir "test.csv") -NoTypeInformation -Encoding UTF8
$labelMap | Export-Csv -Path (Join-Path $outDir "label_map.csv") -NoTypeInformation -Encoding UTF8

$classDistribution = foreach ($row in $labelMap) {
    $className = $row.class_name
    [pscustomobject]@{
        label = $row.label
        class_name = $className
        train_pairs = @($trainRows | Where-Object { $_.class_name -eq $className }).Count
        val_pairs = @($valRows | Where-Object { $_.class_name -eq $className }).Count
        test_pairs = @($testRows | Where-Object { $_.class_name -eq $className }).Count
        train_cases = @($trainRows | Where-Object { $_.class_name -eq $className } | Select-Object -ExpandProperty case_id -Unique).Count
        val_cases = @($valRows | Where-Object { $_.class_name -eq $className } | Select-Object -ExpandProperty case_id -Unique).Count
        test_cases = @($testRows | Where-Object { $_.class_name -eq $className } | Select-Object -ExpandProperty case_id -Unique).Count
    }
}
$classDistribution | Export-Csv -Path (Join-Path $reportDir "class_distribution.csv") -NoTypeInformation -Encoding UTF8

$splitSummary = @(
    [pscustomobject]@{
        split = "train"
        total_pairs = $trainRows.Count
        total_cases = @($trainRows | Select-Object -ExpandProperty case_id -Unique).Count
        classes_present = @($trainRows | Select-Object -ExpandProperty class_name -Unique).Count
    },
    [pscustomobject]@{
        split = "val"
        total_pairs = $valRows.Count
        total_cases = @($valRows | Select-Object -ExpandProperty case_id -Unique).Count
        classes_present = @($valRows | Select-Object -ExpandProperty class_name -Unique).Count
    },
    [pscustomobject]@{
        split = "test"
        total_pairs = $testRows.Count
        total_cases = @($testRows | Select-Object -ExpandProperty case_id -Unique).Count
        classes_present = @($testRows | Select-Object -ExpandProperty class_name -Unique).Count
    }
)
$splitSummary | Export-Csv -Path (Join-Path $reportDir "split_summary.csv") -NoTypeInformation -Encoding UTF8

$trainCounts = @($classDistribution | Select-Object -ExpandProperty train_pairs)
$totalTrain = ($trainCounts | Measure-Object -Sum).Sum
$numClasses = $trainCounts.Count
$weights = @()
foreach ($count in $trainCounts) {
    $weights += [math]::Round($totalTrain / ($numClasses * [double]$count), 6)
}
$countString = ($trainCounts -join ", ")

$notes = @(
    "# P7 Pair Seen-All-Train Protocol",
    "",
    "Definition",
    "- derived from data_protocol/p1_pair_case_isolated_v1",
    "- keep only abnormal pairs",
    "- train on all train-visible pair abnormal subtypes",
    "- evaluate only on seen abnormal subtypes in val/test",
    "- chromosome_id must NOT be used as model input for this protocol because class identity is chromosome-specific",
    "",
    ("- num_classes = {0}" -f $labelMap.Count),
    ("- train pairs = {0}" -f $splitSummary[0].total_pairs),
    ("- val pairs = {0}" -f $splitSummary[1].total_pairs),
    ("- test pairs = {0}" -f $splitSummary[2].total_pairs),
    ("- train class_counts = [{0}]" -f $countString),
    ("- recommended class_weights = [{0}]" -f ($weights -join ", "))
)
$notes | Set-Content -Path (Join-Path $reportDir "protocol_notes.md") -Encoding UTF8

Write-Host ""
Write-Host "Saved P7 pair seen-all-train protocol to: $outDir"
Write-Host "Saved P7 pair seen-all-train report to: $reportDir"
