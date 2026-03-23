$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceCsvs = @(
    (Join-Path $projectRoot "data_randomized\scheme1_single_image_random\train.csv"),
    (Join-Path $projectRoot "data_randomized\scheme1_single_image_random\val.csv"),
    (Join-Path $projectRoot "data_randomized\scheme1_single_image_random\test.csv")
)

$outDir = Join-Path $projectRoot "data_protocol\nonpair_case_isolated_v1"
$reportDir = Join-Path $projectRoot "outputs\baseline_n0_nonpair_case_isolated_v1"

function Normalize-ChromosomeId {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $text = $Value.Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return "UNK"
    }

    $lower = $text.ToLowerInvariant()
    if ($lower -eq "x" -or $lower -eq "xonly") {
        return "X"
    }
    if ($lower -eq "y" -or $lower -eq "yonly") {
        return "Y"
    }

    $digitMatch = [regex]::Match($lower, "\d+")
    if ($digitMatch.Success) {
        return ([int]$digitMatch.Value).ToString()
    }

    return $text.ToUpperInvariant()
}

function Add-ProtocolFields {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Rows,
        [Parameter(Mandatory = $true)]
        [string]$SplitName,
        [Parameter(Mandatory = $true)]
        [hashtable]$TrainSubtypeSet
    )

    foreach ($row in $Rows) {
        $row | Add-Member -NotePropertyName protocol_split -NotePropertyValue $SplitName -Force

        if ([int]$row.label -eq 1) {
            $subtypeId = "{0}::{1}" -f $row.chromosome_id, $row.abnormal_type
            $row | Add-Member -NotePropertyName abnormal_subtype_id -NotePropertyValue $subtypeId -Force

            $status = if ($TrainSubtypeSet.ContainsKey($subtypeId)) { "seen" } else { "unseen" }
            $row | Add-Member -NotePropertyName subtype_status -NotePropertyValue $status -Force
        }
        else {
            $row | Add-Member -NotePropertyName abnormal_subtype_id -NotePropertyValue "" -Force
            $row | Add-Member -NotePropertyName subtype_status -NotePropertyValue "normal" -Force
        }
    }
}

function Export-ProtocolReport {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$TrainRows,
        [Parameter(Mandatory = $true)]
        [object[]]$ValRows,
        [Parameter(Mandatory = $true)]
        [object[]]$TestRows,
        [Parameter(Mandatory = $true)]
        [string]$ReportDir
    )

    New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

    $allRows = @($TrainRows + $ValRows + $TestRows)
    $abnormalRows = @($allRows | Where-Object { [int]$_.label -eq 1 })

    $subtypeSummary = foreach ($group in ($abnormalRows | Group-Object abnormal_subtype_id | Sort-Object Name)) {
        $rows = @($group.Group)
        $caseIds = @($rows | Select-Object -ExpandProperty case_id -Unique)
        $trainCases = @($rows | Where-Object { $_.protocol_split -eq "train" } | Select-Object -ExpandProperty case_id -Unique)
        $valCases = @($rows | Where-Object { $_.protocol_split -eq "val" } | Select-Object -ExpandProperty case_id -Unique)
        $testCases = @($rows | Where-Object { $_.protocol_split -eq "test" } | Select-Object -ExpandProperty case_id -Unique)

        $bucket = if ($caseIds.Count -ge 3) {
            "head"
        }
        elseif ($caseIds.Count -eq 2) {
            "medium"
        }
        else {
            "tail"
        }

        $protocolRole = if ($trainCases.Count -eq 0) {
            "unseen"
        }
        elseif ($valCases.Count -gt 0 -or $testCases.Count -gt 0) {
            "seen_eval"
        }
        else {
            "train_only"
        }

        [pscustomobject]@{
            abnormal_subtype_id = $group.Name
            chromosome_id = ($rows[0].chromosome_id)
            abnormal_type = ($rows[0].abnormal_type)
            total_image_count = $rows.Count
            total_case_count = $caseIds.Count
            train_image_count = @($rows | Where-Object { $_.protocol_split -eq "train" }).Count
            val_image_count = @($rows | Where-Object { $_.protocol_split -eq "val" }).Count
            test_image_count = @($rows | Where-Object { $_.protocol_split -eq "test" }).Count
            train_case_count = $trainCases.Count
            val_case_count = $valCases.Count
            test_case_count = $testCases.Count
            frequency_bucket = $bucket
            protocol_role = $protocolRole
            train_case_ids = ($trainCases -join ",")
            val_case_ids = ($valCases -join ",")
            test_case_ids = ($testCases -join ",")
        }
    }

    $caseSummary = foreach ($caseGroup in ($abnormalRows | Group-Object case_id | Sort-Object Name)) {
        $rows = @($caseGroup.Group)
        $subtypes = @($rows | Select-Object -ExpandProperty abnormal_subtype_id -Unique)
        [pscustomobject]@{
            case_id = $caseGroup.Name
            protocol_split = ($rows[0].protocol_split)
            abnormal_image_count = $rows.Count
            subtype_count = $subtypes.Count
            abnormal_subtype_ids = ($subtypes -join ",")
            chromosome_ids = ((@($rows | Select-Object -ExpandProperty chromosome_id -Unique) | Sort-Object) -join ",")
        }
    }

    $splitSummary = @(
        [pscustomobject]@{
            split = "train"
            total_images = $TrainRows.Count
            abnormal_images = @($TrainRows | Where-Object { [int]$_.label -eq 1 }).Count
            normal_images = @($TrainRows | Where-Object { [int]$_.label -eq 0 }).Count
            total_cases = @($TrainRows | Select-Object -ExpandProperty case_id -Unique).Count
            abnormal_cases = @($TrainRows | Where-Object { [int]$_.label -eq 1 } | Select-Object -ExpandProperty case_id -Unique).Count
            seen_abnormal_subtypes = @($TrainRows | Where-Object { $_.subtype_status -eq "seen" } | Select-Object -ExpandProperty abnormal_subtype_id -Unique).Count
            unseen_abnormal_subtypes = 0
        },
        [pscustomobject]@{
            split = "val"
            total_images = $ValRows.Count
            abnormal_images = @($ValRows | Where-Object { [int]$_.label -eq 1 }).Count
            normal_images = @($ValRows | Where-Object { [int]$_.label -eq 0 }).Count
            total_cases = @($ValRows | Select-Object -ExpandProperty case_id -Unique).Count
            abnormal_cases = @($ValRows | Where-Object { [int]$_.label -eq 1 } | Select-Object -ExpandProperty case_id -Unique).Count
            seen_abnormal_subtypes = @($ValRows | Where-Object { $_.subtype_status -eq "seen" } | Select-Object -ExpandProperty abnormal_subtype_id -Unique).Count
            unseen_abnormal_subtypes = @($ValRows | Where-Object { $_.subtype_status -eq "unseen" } | Select-Object -ExpandProperty abnormal_subtype_id -Unique).Count
        },
        [pscustomobject]@{
            split = "test"
            total_images = $TestRows.Count
            abnormal_images = @($TestRows | Where-Object { [int]$_.label -eq 1 }).Count
            normal_images = @($TestRows | Where-Object { [int]$_.label -eq 0 }).Count
            total_cases = @($TestRows | Select-Object -ExpandProperty case_id -Unique).Count
            abnormal_cases = @($TestRows | Where-Object { [int]$_.label -eq 1 } | Select-Object -ExpandProperty case_id -Unique).Count
            seen_abnormal_subtypes = @($TestRows | Where-Object { $_.subtype_status -eq "seen" } | Select-Object -ExpandProperty abnormal_subtype_id -Unique).Count
            unseen_abnormal_subtypes = @($TestRows | Where-Object { $_.subtype_status -eq "unseen" } | Select-Object -ExpandProperty abnormal_subtype_id -Unique).Count
        }
    )

    $subtypeSummary |
        Sort-Object protocol_role, frequency_bucket, abnormal_subtype_id |
        Export-Csv -Path (Join-Path $ReportDir "subtype_protocol_summary.csv") -NoTypeInformation -Encoding UTF8

    $caseSummary |
        Sort-Object protocol_split, case_id |
        Export-Csv -Path (Join-Path $ReportDir "case_protocol_summary.csv") -NoTypeInformation -Encoding UTF8

    $splitSummary |
        Export-Csv -Path (Join-Path $ReportDir "split_summary.csv") -NoTypeInformation -Encoding UTF8

    $protocolLines = @(
        "# Baseline N0 Protocol",
        "",
        "Subtype definition",
        "- abnormal subtype = chromosome_id::abnormal_type",
        "- current dataset abnormal_type is inversion only, so subtype reduces to chromosome_id::inversion",
        "",
        "Split rules",
        "- case-isolated",
        "- train contains only seen abnormal subtypes",
        "- val/test each contain both seen and unseen abnormal subtype cases",
        "",
        "Frequency buckets",
        "- head: >= 3 cases in the full source pool",
        "- medium: 2 cases in the full source pool",
        "- tail: 1 case in the full source pool",
        "- unseen: subtype absent from train but present in val or test",
        "",
        "Evaluation slices",
        "- N1 detection: use all rows in each split",
        "- N2 seen abnormal classification: evaluate abnormal rows with subtype_status=seen",
        "- N3 unseen abnormal recognition: evaluate abnormal rows with subtype_status=unseen"
    )
    $protocolLines | Set-Content -Path (Join-Path $ReportDir "evaluation_protocol.md") -Encoding UTF8
}

$rows = foreach ($csvPath in $sourceCsvs) {
    if (-not (Test-Path $csvPath)) {
        throw "Missing source CSV: $csvPath"
    }
    Import-Csv $csvPath
}

$rows = $rows |
    Sort-Object image_path -Unique |
    ForEach-Object {
        $_.chromosome_id = Normalize-ChromosomeId $_.chromosome_id
        $_.chromosome_from_name = Normalize-ChromosomeId $_.chromosome_from_name
        $_.abnormal_type = if ([int]$_.label -eq 1) { "inversion" } else { "normal" }
        $_
    }

$caseRecords = foreach ($group in ($rows | Group-Object case_id)) {
    $caseRows = @($group.Group)
    $abnormalRows = @($caseRows | Where-Object { [int]$_.label -eq 1 })
    $subtypes = @(
        $abnormalRows |
        ForEach-Object { "{0}::{1}" -f $_.chromosome_id, $_.abnormal_type } |
        Select-Object -Unique
    )

    if ($subtypes.Count -ne 1) {
        throw "Expected exactly one abnormal subtype per case for case_id=$($group.Name), got $($subtypes.Count)"
    }

    [pscustomobject]@{
        case_id = $group.Name
        abnormal_subtype_id = $subtypes[0]
        abnormal_image_count = $abnormalRows.Count
        total_image_count = $caseRows.Count
    }
}

$splitCaseIds = @{
    train = @(
        "165070",
        "170215",
        "179083",
        "180371",
        "183038",
        "183079",
        "183214",
        "183262",
        "183786",
        "184752",
        "185290",
        "185824",
        "186193",
        "Q242362",
        "Q242664",
        "Q250230",
        "Q251136",
        "182418",
        "182747",
        "184793"
    )
    val = @(
        "185618",
        "Q250230M",
        "Q242362M",
        "169598",
        "79488",
        "183426"
    )
    test = @(
        "Y19361M",
        "Q242664M",
        "M182418",
        "Q250870",
        "75263",
        "76018"
    )
}

$assignedCaseIds = @($splitCaseIds.train + $splitCaseIds.val + $splitCaseIds.test)
$allCaseIds = @($caseRecords | Select-Object -ExpandProperty case_id)

if (($assignedCaseIds | Select-Object -Unique).Count -ne $assignedCaseIds.Count) {
    throw "Duplicate case assignment detected in splitCaseIds"
}

$allCaseIdsJoined = (($allCaseIds | Sort-Object) -join ",")
$assignedCaseIdsJoined = (($assignedCaseIds | Sort-Object) -join ",")
if ($allCaseIdsJoined -ne $assignedCaseIdsJoined) {
    throw "Split assignment does not cover all source cases exactly once"
}

$trainCaseSet = @{}
foreach ($caseId in $splitCaseIds.train) {
    $caseRecord = $caseRecords | Where-Object { $_.case_id -eq $caseId }
    $trainCaseSet[$caseRecord.abnormal_subtype_id] = $true
}

$trainRows = @($rows | Where-Object { $splitCaseIds.train -contains $_.case_id })
$valRows = @($rows | Where-Object { $splitCaseIds.val -contains $_.case_id })
$testRows = @($rows | Where-Object { $splitCaseIds.test -contains $_.case_id })

Add-ProtocolFields -Rows $trainRows -SplitName "train" -TrainSubtypeSet $trainCaseSet
Add-ProtocolFields -Rows $valRows -SplitName "val" -TrainSubtypeSet $trainCaseSet
Add-ProtocolFields -Rows $testRows -SplitName "test" -TrainSubtypeSet $trainCaseSet

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$trainRows | Sort-Object case_id, filename | Export-Csv -Path (Join-Path $outDir "train.csv") -NoTypeInformation -Encoding UTF8
$valRows | Sort-Object case_id, filename | Export-Csv -Path (Join-Path $outDir "val.csv") -NoTypeInformation -Encoding UTF8
$testRows | Sort-Object case_id, filename | Export-Csv -Path (Join-Path $outDir "test.csv") -NoTypeInformation -Encoding UTF8

Export-ProtocolReport -TrainRows $trainRows -ValRows $valRows -TestRows $testRows -ReportDir $reportDir

Write-Host ""
Write-Host "Saved protocol split to: $outDir"
Write-Host "Saved N0 report to: $reportDir"
