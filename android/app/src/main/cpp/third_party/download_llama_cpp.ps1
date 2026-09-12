# Repopulates third_party/llama.cpp with the llama.cpp PR #22836 head
# (STQ1_0 kernel, sjl623:STQ_0 @ 1e411d8f5a1e23525fa3265dfb4bd76265465397).
#
# Downloads via the gh-proxy.com GitHub mirror (fast, ~3.5MB/s on this
# network) with fallbacks to ghfast.top, ghproxy.net and codeload.github.com
# (codeload is usually reachable even when github.com git-over-HTTPS is
# unreliable, but cuts connections mid-transfer on this network, so it is
# the last resort). Each candidate is verified with `tar -tzf` (gzip
# integrity) before extraction.
#
# Usage:  pwsh -File download_llama_cpp.ps1 [-KeepTarball]

param(
    [switch]$KeepTarball
)

$ErrorActionPreference = 'Stop'
$dest   = Join-Path $PSScriptRoot 'llama.cpp'
$tar    = Join-Path $PSScriptRoot 'llama.cpp-pr22836.tar.gz'
$sha    = '1e411d8f5a1e23525fa3265dfb4bd76265465397'  # PR head, sjl623:STQ_0
$urls   = @(
    "https://gh-proxy.com/https://github.com/sjl623/llama.cpp/archive/$sha.tar.gz",
    "https://ghfast.top/https://github.com/sjl623/llama.cpp/archive/$sha.tar.gz",
    "https://ghproxy.net/https://github.com/sjl623/llama.cpp/archive/$sha.tar.gz",
    'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/pull/22836/head'
)

$ok = $false
foreach ($url in $urls) {
    foreach ($attempt in 1..3) {
        Write-Host "==> [$attempt/3] $url"
        # --speed-limit 10000 --speed-time 60: abort if throughput drops below 10KB/s for 60s
        curl.exe -sSfL --connect-timeout 20 --speed-limit 10000 --speed-time 60 -o $tar $url
        if ($LASTEXITCODE -eq 0 -and (Test-Path $tar)) {
            # Verify gzip integrity: a cut connection yields a truncated tarball
            # that only fails when extracted, so check it up front.
            tar.exe -tzf $tar | Out-Null
            if ($LASTEXITCODE -eq 0) { $ok = $true; break }
            Write-Host "    tarball integrity check failed (truncated?) — retrying"
        }
        Remove-Item $tar -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    if ($ok) { break }
}
if (-not $ok) {
    throw "Download failed from all mirrors — check network/proxy access to github.com"
}
Write-Host "    downloaded $((Get-Item $tar).Length) bytes (verified gzip)"

Write-Host "==> Extracting to $dest ..."
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dest | Out-Null
tar.exe -xzf $tar -C $dest --strip-components=1
if ($LASTEXITCODE -ne 0) { throw "tar extract failed" }

if (-not (Test-Path (Join-Path $dest 'CMakeLists.txt'))) { throw "extracted tree missing CMakeLists.txt" }
Write-Host "==> Done. llama.cpp PR #22836 ($sha) vendored at $dest"

if (-not $KeepTarball) { Remove-Item $tar -Force -ErrorAction SilentlyContinue }
