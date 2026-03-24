$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$n0Dir = Join-Path $projectRoot "data_protocol\nonpair_case_isolated_v1"
$n0SummaryPath = Join-Path $projectRoot "outputs\baseline_n0_nonpair_case_isolated_v1\subtype_protocol_summary.csv"
$outDir = Join-Path $projectRoot "data_protocol\n2_seen_only_v1"
$reportDir = Join-Path $projectRoot "outputs\n2_seen_only_v1"

if (-not (Test-Path $n0SummaryPath)) {
    throw "Missing N0 subtype summary: $n0SummaryPath"
}

$summary = Import-Csv $n0SummaryPath
$seenEvalSubtypes = @(
    $summary |
    Where-Object { $_.protocol_role -eq "seen_eval" } |
    Sort-Object {[int]($_.chromosome_id -replace '\D','0')}, chromosome_id |
    Select-Object -ExpandProperty abnormal_subtype_id
)

if ($seenEvalSubtypes.Count -eq 0) {
    throw "No seen_eval subtypes found in N0 summary"
}

$labelMap = @()
for ($idx = 0; $idx -lt $seenEvalSubtypes.Count; $idx++) {
    $subtype = $seenEvalSubtypes[$idx]
    $chromosomeId = $subtype.Split("::")[0]
    $labelMap += [pscustomobject]@{
        label = $idx
        abnormal_subtype_id = $subtype
        chromosome_id = $chromosomeId
        class_name = $subtype
    }
}

$labelLookup = @{}
foreach ($row in $labelMap) {
    $labelLookup[$row.abnormal_subtype_id] = [int]$row.label
}

function Build-Split {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CsvPath,
        [Parameter(Mandatory = $true)]
        [string]$SplitName
    )

    $rows = Import-Csv $CsvPath
    $rows = @(
        $rows |
        Where-Object {
            [int]$_.label -eq 1 -and
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
    return $rows
}

$trainRows = Build-Split -CsvPath (Join-Path $n0Dir "train.csv") -SplitName "train"
$valRows = Build-Split -CsvPath (Join-Path $n0Dir "val.csv") -SplitName "val"
$testRows = Build-Split -CsvPath (Join-Path $n0Dir "test.csv") -SplitName "test"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$trainRows | Sort-Object label, case_id, filename | Export-Csv -Path (Join-Path $outDir "train.csv") -NoTypeInformation -Encoding UTF8
$valRows | Sort-Object label, case_id, filename | Export-Csv -Path (Join-Path $outDir "val.csv") -NoTypeInformation -Encoding UTF8
$testRows | Sort-Object label, case_id, filename | Export-Csv -Path (Join-Path $outDir "test.csv") -NoTypeInformation -Encoding UTF8
$labelMap | Export-Csv -Path (Join-Path $outDir "label_map.csv") -NoTypeInformation -Encoding UTF8

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

$classDistribution = foreach ($mapRow in $labelMap) {
    $className = $mapRow.class_name
    [pscustomobject]@{
        label = $mapRow.label
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

$trainCounts = @($classDistribution | Select-Object -ExpandProperty train_images)
$totalTrain = ($trainCounts | Measure-Object -Sum).Sum
$numClasses = $trainCounts.Count
$weights = @()
foreach ($count in $trainCounts) {
    $weights += [math]::Round($totalTrain / ($numClasses * [double]$count), 6)
}

$recipe = @(
    "# N2 Seen-Only Protocol",
    "",
    "Source",
    "- derived from data_protocol/nonpair_case_isolated_v1",
    "- keep only abnormal rows whose subtype belongs to protocol_role = seen_eval",
    "",
    "Classes",
    ("- num_classes = {0}" -f $labelMap.Count),
    ("- class_names = {0}" -f (($labelMap | Select-Object -ExpandProperty class_name) -join ", ")),
    "",
    "Split summary",
    ("- train: {0} images, {1} cases, {2} classes present" -f $splitSummary[0].total_images, $splitSummary[0].total_cases, $splitSummary[0].classes_present),
    ("- val: {0} images, {1} cases, {2} classes present" -f $splitSummary[1].total_images, $splitSummary[1].total_cases, $splitSummary[1].classes_present),
    ("- test: {0} images, {1} cases, {2} classes present" -f $splitSummary[2].total_images, $splitSummary[2].total_cases, $splitSummary[2].classes_present),
    "",
    ("- recommended class_weights = [{0}]" -f ($weights -join ", "))
)
$recipe | Set-Content -Path (Join-Path $reportDir "protocol_notes.md") -Encoding UTF8

Write-Host ""
Write-Host "Saved N2 seen-only protocol to: $outDir"
Write-Host "Saved N2 report to: $reportDir"
