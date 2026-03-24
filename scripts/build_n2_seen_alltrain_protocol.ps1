$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$n0Dir = Join-Path $projectRoot "data_protocol\nonpair_case_isolated_v1"
$outDir = Join-Path $projectRoot "data_protocol\n2_seen_alltrain_v1"
$reportDir = Join-Path $projectRoot "outputs\n2_seen_alltrain_v1"

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

$trainSource = Import-Csv (Join-Path $n0Dir "train.csv")
$valSource = Import-Csv (Join-Path $n0Dir "val.csv")
$testSource = Import-Csv (Join-Path $n0Dir "test.csv")

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
            [int]$_.label -eq 1 -and $_.subtype_status -eq "seen"
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

$trainRows | Sort-Object label, case_id, filename | Export-Csv -Path (Join-Path $outDir "train.csv") -NoTypeInformation -Encoding UTF8
$valRows | Sort-Object label, case_id, filename | Export-Csv -Path (Join-Path $outDir "val.csv") -NoTypeInformation -Encoding UTF8
$testRows | Sort-Object label, case_id, filename | Export-Csv -Path (Join-Path $outDir "test.csv") -NoTypeInformation -Encoding UTF8
$labelMap | Export-Csv -Path (Join-Path $outDir "label_map.csv") -NoTypeInformation -Encoding UTF8

$classDistribution = foreach ($row in $labelMap) {
    $className = $row.class_name
    [pscustomobject]@{
        label = $row.label
        class_name = $className
        train_images = @($trainRows | Where-Object { $_.class_name -eq $className }).Count
        val_images = @($valRows | Where-Object { $_.class_name -eq $className }).Count
        test_images = @($testRows | Where-Object { $_.class_name -eq $className }).Count
        train_cases = @($trainRows | Where-Object { $_.class_name -eq $className } | Select-Object -ExpandProperty case_id -Unique).Count
        val_cases = @($valRows | Where-Object { $_.class_name -eq $className } | Select-Object -ExpandProperty case_id -Unique).Count
        test_cases = @($testRows | Where-Object { $_.class_name -eq $className } | Select-Object -ExpandProperty case_id -Unique).Count
    }
}
$classDistribution | Export-Csv -Path (Join-Path $reportDir "class_distribution.csv") -NoTypeInformation -Encoding UTF8

$splitSummary = @(
    [pscustomobject]@{
        split = "train"
        total_images = $trainRows.Count
        total_cases = @($trainRows | Select-Object -ExpandProperty case_id -Unique).Count
        classes_present = @($trainRows | Select-Object -ExpandProperty class_name -Unique).Count
    },
    [pscustomobject]@{
        split = "val"
        total_images = $valRows.Count
        total_cases = @($valRows | Select-Object -ExpandProperty case_id -Unique).Count
        classes_present = @($valRows | Select-Object -ExpandProperty class_name -Unique).Count
    },
    [pscustomobject]@{
        split = "test"
        total_images = $testRows.Count
        total_cases = @($testRows | Select-Object -ExpandProperty case_id -Unique).Count
        classes_present = @($testRows | Select-Object -ExpandProperty class_name -Unique).Count
    }
)
$splitSummary | Export-Csv -Path (Join-Path $reportDir "split_summary.csv") -NoTypeInformation -Encoding UTF8

$trainCounts = @($classDistribution | Select-Object -ExpandProperty train_images)
$totalTrain = ($trainCounts | Measure-Object -Sum).Sum
$numClasses = $trainCounts.Count
$weights = @()
foreach ($count in $trainCounts) {
    $weights += [math]::Round($totalTrain / ($numClasses * [double]$count), 6)
}

$notes = @(
    "# N2 Seen-All-Train Protocol",
    "",
    "Definition",
    "- train on all train-visible abnormal subtypes from N0",
    "- evaluate only on seen abnormal rows in val/test",
    "",
    ("- num_classes = {0}" -f $labelMap.Count),
    ("- train images = {0}" -f $splitSummary[0].total_images),
    ("- val images = {0}" -f $splitSummary[1].total_images),
    ("- test images = {0}" -f $splitSummary[2].total_images),
    ("- recommended class_weights = [{0}]" -f ($weights -join ", "))
)
$notes | Set-Content -Path (Join-Path $reportDir "protocol_notes.md") -Encoding UTF8

Write-Host ""
Write-Host "Saved N2 seen-all-train protocol to: $outDir"
Write-Host "Saved N2 seen-all-train report to: $reportDir"
