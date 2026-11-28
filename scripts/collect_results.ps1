# Copy the small, publishable outputs of a full run into results\ (for git).
# Leaves out what must not or cannot go to GitHub: perception caches, mined clip JSONs, evidence
# keyframes/crops (frames of the Nexar videos) and the videos themselves.
#
#   cd D:\work\SOTIF-Aligned
#   powershell -ExecutionPolicy Bypass -File scripts\collect_results.ps1 -Runs D:\work\sotif-miner\runs
param(
    [string]$Runs = "D:\work\sotif-miner\runs",
    [string]$Out = "results",
    [int]$MaxMB = 50
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Runs)) { throw "Runs folder not found: $Runs" }

$items = @(
    # models (recommended first)
    @{ src = "nexar_v6\stage1_model_drop.joblib";        dst = "models\stage1_model_drop.joblib" },
    @{ src = "nexar_v6\stage1_model.joblib";             dst = "models\stage1_model_full.joblib" },
    @{ src = "nexar_v6\stage1_model_legacy.joblib";      dst = "models\stage1_model_legacy_features.joblib" },
    @{ src = "baseline_200\stage1_model.joblib";         dst = "models\stage1_model_200clips.joblib" },
    # cross-validated metrics (1,300 clips) and experiments
    @{ src = "nexar_v6\stage1_cv_metrics*.json";         dst = "metrics\cv\" },
    @{ src = "baseline_200\stage1_cv_metrics.json";      dst = "metrics\cv\stage1_cv_metrics_200clips.json" },
    @{ src = "model_1300\stage1_cv_metrics.json";        dst = "metrics\cv\stage1_cv_metrics_1300clips_prefix.json" },
    # held-out / test metrics with bootstrap CIs, confound table
    @{ src = "*_ci.json";                                dst = "metrics\heldout\" },
    @{ src = "confound_v6.json";                         dst = "metrics\confound_v6.json" },
    # diagnosis (text cards only) and taxonomy
    @{ src = "nexar_v6_diag\cards.jsonl";                dst = "diagnosis\near_miss_cards.jsonl" },
    @{ src = "nexar_v6_diag\diagnosis_summary.json";     dst = "diagnosis\near_miss_summary.json" },
    @{ src = "nexar_heldout_v6_drop\cards.jsonl";        dst = "diagnosis\false_alarm_cards.jsonl" },
    @{ src = "nexar_heldout_v6_drop\diagnosis_summary.json"; dst = "diagnosis\false_alarm_summary.json" },
    @{ src = "taxonomy_v6\clusters.md";                  dst = "taxonomy\clusters.md" },
    @{ src = "taxonomy_v6\clusters.json";                dst = "taxonomy\clusters.json" },
    @{ src = "taxonomy_v6\assignments.json";             dst = "taxonomy\assignments.json" }
)

$copied = 0; $bytes = 0
foreach ($it in $items) {
    $files = Get-ChildItem -Path (Join-Path $Runs $it.src) -File -ErrorAction SilentlyContinue
    if (-not $files) { Write-Host "  skip (not found): $($it.src)"; continue }
    foreach ($f in $files) {
        if ($f.Length -gt $MaxMB * 1MB) { Write-Host "  skip (> $MaxMB MB): $($f.FullName)"; continue }
        $dst = Join-Path $Out $it.dst
        if ($it.dst.EndsWith("\")) { $dst = Join-Path $dst $f.Name }
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Copy-Item $f.FullName $dst -Force
        $copied++; $bytes += $f.Length
        Write-Host ("  {0,8:N1} KB  {1}" -f ($f.Length / 1KB), $dst)
    }
}
"{0} files, {1:N1} MB in {2}\" -f $copied, ($bytes / 1MB), $Out | Write-Host
