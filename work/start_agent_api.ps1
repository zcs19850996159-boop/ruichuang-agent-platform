param(
  [string]$HostName = "127.0.0.1",
  [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$readmePath = Join-Path $root "outputs\rag_agent\README.md"
if (Test-Path $readmePath) {
  $readme = Get-Content $readmePath -Raw
  if ($readme -match 'DEEPSEEK_API_KEY\s*=\s*([^\r\n]+)') {
    $env:DEEPSEEK_API_KEY = $Matches[1].Trim().Trim('"').Trim("'")
  }
}

$env:DEEPSEEK_TEMPERATURE = if ($env:DEEPSEEK_TEMPERATURE) { $env:DEEPSEEK_TEMPERATURE } else { "0.1" }
$env:API_USE_LLM_SELECTOR = if ($env:API_USE_LLM_SELECTOR) { $env:API_USE_LLM_SELECTOR } else { "1" }

$python = "D:\Apps\Python 3.11\python.exe"
$logDir = Join-Path $root "outputs\rag_agent"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$outLog = Join-Path $logDir "agent_api_$Port.out.log"
$errLog = Join-Path $logDir "agent_api_$Port.err.log"

& $python "work\agent_api.py" --host $HostName --port $Port 1>> $outLog 2>> $errLog
