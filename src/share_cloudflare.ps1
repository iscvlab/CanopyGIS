param(
    [switch]$LocalOnly,
    [int]$Port = 8000
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$helperPath = Join-Path $scriptDir "..\..\..\..\Tools\scripts\share_helper.ps1"

if (Test-Path $helperPath) {
    . $helperPath
    $isLocal = $LocalOnly -or ($env:LOCAL_ONLY -eq "true")
    Start-ServerAndTunnel `
        -appName "Canopy GIS" `
        -workDir $scriptDir `
        -serverScript "run_uvicorn.py" `
        -defaultPort $Port `
        -requireVenv $false `
        -tunnelUrlFile "active_tunnel.url" `
        -localOnly $isLocal
} else {
    Set-Location $scriptDir
    $env:PORT = "$Port"
    $cf = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cf -and -not $LocalOnly) {
        Start-Process python -ArgumentList "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$Port" -NoNewWindow
        & $cf.Source tunnel --url "http://127.0.0.1:$Port"
    } else {
        python -m uvicorn main:app --host 127.0.0.1 --port $Port
    }
}
