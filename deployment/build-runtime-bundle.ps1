$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$stageRoot = Join-Path $scriptRoot "dist\runtime-bundle\agentic_document_intelligence"

$relativePaths = @(
    "corpus\metadata\corpus_metadata.json",
    "corpus\metadata\corpus_manifest.json",
    "corpus\sources\Microsoft_FY2025_10K_Summary.pdf",
    "corpus\datasets\financial_performance_by_segment.csv",
    "corpus\datasets\geographic_revenue_mix.csv",
    "corpus\datasets\product_family_signals.csv",
    "artifacts\experiments\component6_sql_schema_packaging_live\sql_schema\sql_schema_package.json",
    "artifacts\experiments\component6_sqlite_database_build_live\sql_db\microsoft_fy2025_analyst_demo.sqlite",
    "artifacts\experiments\component5_kuzu_graph_build_live\kuzu_db\microsoft_fy2025_10k_summary.kuzu",
    "artifacts\experiments\component5_graph_schema_validation_live\graph_validated\microsoft_fy2025_10k_summary_graph_validated.json",
    "artifacts\experiments\component2_chunk_generation\chunks\microsoft_fy2025_10k_summary_chunks.json",
    "artifacts\experiments\component2_embedding_ready_records\embeddings\microsoft_fy2025_10k_summary_embedding_records.json"
)

if (Test-Path (Split-Path -Parent $stageRoot)) {
    Remove-Item (Split-Path -Parent $stageRoot) -Recurse -Force
}

foreach ($relativePath in $relativePaths) {
    $source = Join-Path $projectRoot $relativePath
    $destination = Join-Path $stageRoot $relativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item $source $destination -Force
}

Write-Output "Runtime bundle stage ready at $(Split-Path -Parent $stageRoot)"
