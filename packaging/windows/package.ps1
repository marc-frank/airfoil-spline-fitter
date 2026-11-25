<#
.SYNOPSIS
  Build and package Airfoil Spline Fitter for Windows.

.DESCRIPTION
  Produces, in $env:DIST (default: dist/release):
    spline-fitter-<version>-windows-<arch>.msi            per-machine installer (WiX v5)
    spline-fitter-<version>-windows-<arch>-portable.zip   spline-fitter.exe + README + licence
    AirfoilSplineFitter-<version>-windows-<arch>.exe      the app alone (x64 only: a plain download)

  The binary links the C runtime statically (+crt-static), so nothing else needs installing, and
  carries the app icon (crates/spline-fitter-gui/build.rs).

  Needs: Rust (MSVC toolchain + the target) and WiX v5:
    dotnet tool install --global wix --version 5.0.2
    wix extension add -g WixToolset.UI.wixext/5.0.2 WixToolset.Util.wixext/5.0.2

  After SolidCraft's packaging/windows/package.ps1 (Apache-2.0), without its signing phases.

.EXAMPLE
  powershell -File packaging/windows/package.ps1 -Arch x64
  powershell -File packaging/windows/package.ps1 -Arch arm64     # cross-compiled; needs the MSVC ARM64 build tools
  powershell -File packaging/windows/package.ps1 -Arch x64 -PortableOnly   # no WiX needed
#>
param(
  [ValidateSet('x64', 'x86', 'arm64')] [string] $Arch = 'x64',
  [switch] $SkipBuild,
  # Only the portable zip: no MSI, no plain .exe.
  [switch] $PortableOnly
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Invoke-Native([string] $What, [scriptblock] $Block) {
  Write-Output "==> $What"
  & $Block
  if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

# The version lives in one place: [package] version of the GUI crate.
$Version = $env:SPLINE_FITTER_VERSION
if (-not $Version) {
  $inPkg = $false
  foreach ($line in Get-Content (Join-Path $Root 'crates\spline-fitter-gui\Cargo.toml')) {
    if ($line -match '^\s*\[') { $inPkg = ($line.Trim() -eq '[package]'); continue }
    if ($inPkg -and $line -match '^\s*version\s*=\s*"([^"]+)"') { $Version = $Matches[1]; break }
  }
}
if (-not $Version) { throw 'could not read the version from crates/spline-fitter-gui/Cargo.toml' }
# MSI ProductVersion is numeric (major.minor.build); pre-release tags are dropped there.
$MsiVersion = ($Version -split '-')[0]

$Target = switch ($Arch) { 'x64' { 'x86_64-pc-windows-msvc' } 'x86' { 'i686-pc-windows-msvc' } 'arm64' { 'aarch64-pc-windows-msvc' } }
$Dist = if ($env:DIST) { $env:DIST } else { Join-Path $Root 'dist\release' }
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root 'target' }
New-Item -ItemType Directory -Force -Path $Dist | Out-Null

Write-Output "Airfoil Spline Fitter $Version for Windows $Arch ($Target)"

if (-not $SkipBuild) {
  # Static CRT: no VC++ redistributable needed. Scoped to the target so host build scripts and
  # proc-macros are unaffected.
  $flagVar = 'CARGO_TARGET_' + ($Target.ToUpper() -replace '-', '_') + '_RUSTFLAGS'
  [Environment]::SetEnvironmentVariable($flagVar, '-C target-feature=+crt-static')
  Invoke-Native "cargo build ($Target)" { cargo build --release --locked -p spline-fitter-gui --target $Target }
}
$Bin = Join-Path $TargetDir "$Target\release"
$Exe = Join-Path $Bin 'spline-fitter.exe'

# The PE header: Machine must match -Arch; Subsystem 2 = GUI (no console window).
$bytes = [System.IO.File]::ReadAllBytes($Exe)
$pe = [BitConverter]::ToInt32($bytes, 0x3C)
$machine = [BitConverter]::ToUInt16($bytes, $pe + 4)
$subsystem = [BitConverter]::ToUInt16($bytes, $pe + 0x5C)
$want = switch ($Arch) { 'x64' { 0x8664 } 'x86' { 0x14C } 'arm64' { 0xAA64 } }
if ($machine -ne $want) { throw "spline-fitter.exe is for machine 0x$('{0:X}' -f $machine), expected 0x$('{0:X}' -f $want) ($Arch)" }
if ($subsystem -ne 2) { throw "spline-fitter.exe has PE subsystem $subsystem, expected 2 (GUI)" }
Write-Output "ok spline-fitter.exe: $Arch, GUI subsystem"

$Stage = Join-Path $TargetDir "windows-package\$Arch"
Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
Copy-Item $Exe, (Join-Path $Root 'LICENSE') $Stage

$Icon = Join-Path $Root 'crates\spline-fitter-gui\assets\app_icon.ico'
$Outputs = @()

# ---- MSI ---------------------------------------------------------------------------------------
if (-not $PortableOnly) {
  $Msi = Join-Path $Dist "spline-fitter-$Version-windows-$Arch.msi"
  Invoke-Native 'wix build' {
    wix build (Join-Path $PSScriptRoot 'spline-fitter.wxs') -arch $Arch `
      -ext WixToolset.UI.wixext -ext WixToolset.Util.wixext -culture en-US `
      -d "Version=$MsiVersion" -d "DisplayVersion=$Version" -d "BinDir=$Stage" -d "IconPath=$Icon" `
      -o $Msi
  }
  # wix writes its debug symbols (.wixpdb) next to the MSI; keep them out of the release assets.
  Remove-Item -Force -ErrorAction SilentlyContinue ([IO.Path]::ChangeExtension($Msi, '.wixpdb'))
  $Outputs += $Msi
}

# ---- portable zip ------------------------------------------------------------------------------
$Portable = Join-Path $TargetDir "windows-package\spline-fitter-$Version-windows-$Arch-portable"
Remove-Item -Recurse -Force $Portable -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Portable | Out-Null
Copy-Item $Exe $Portable
foreach ($f in 'README.md', 'LICENSE') { Copy-Item (Join-Path $Root $f) $Portable }
$Zip = Join-Path $Dist "spline-fitter-$Version-windows-$Arch-portable.zip"
Remove-Item -Force $Zip -ErrorAction SilentlyContinue
Compress-Archive -Path $Portable -DestinationPath $Zip
$Outputs += $Zip

# ---- the app as a plain download (x64) ---------------------------------------------------------
if ($Arch -eq 'x64' -and -not $PortableOnly) {
  $Plain = Join-Path $Dist "AirfoilSplineFitter-$Version-windows-$Arch.exe"
  Copy-Item $Exe $Plain
  $Outputs += $Plain
}

Get-Item $Outputs | Format-Table Name, Length
