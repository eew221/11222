param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('A', 'B', 'B_retry')]
    [string]$Annotator,
    [int]$Port = 8795,
    [string]$AuditRoot = 'audit/independent_worker_state_random_20260827_v1'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path $AuditRoot).Path
python -u scripts/serve_worker_state_audit.py `
    --audit-root $root `
    --annotator $Annotator `
    --port $Port
