<#
安装脚本（PowerShell）
用途：在指定的 conda 环境中以 editable 模式安装仓库内的包： myafc。
用法：在 PowerShell 中运行（先确保 conda 可用）：
    .\install_packages.ps1 -EnvName zero_phi310
可选：添加 -DryRun 仅打印将要执行的操作而不实际安装。
#>

param(
    [string]$EnvName = "zero_phi310",
    [string[]]$PackageDirs = $null,
    [switch]$DryRun
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host "Script dir: $ScriptDir"

if (-not $PackageDirs) {
    $PackageDirs = @(
        Join-Path $ScriptDir "0mydev\myafc\src"
    )
}

Write-Host "Conda env: $EnvName"
Write-Host "Package directories to install (editable):"
$PackageDirs | ForEach-Object { Write-Host "  - $_" }

if ($DryRun) { Write-Host "Dry run mode: no changes will be made."; exit 0 }

Write-Host "Activating conda environment '$EnvName'..."
# Activate conda environment - this requires conda to be available in the shell
# If this fails in some CI shells, run: & "C:\ProgramData\Miniconda3\Scripts\activate" $EnvName
try {
    conda activate $EnvName
} catch {
    Write-Warning "Failed to run 'conda activate $EnvName' directly. If you see issues, make sure conda is initialized for PowerShell or activate manually and re-run this script."
}

# Upgrade pip and build tools inside the environment
Write-Host "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Write-Warning "pip upgrade returned non-zero exit code ($LASTEXITCODE)" }

$createdFiles = @()
try {
    foreach ($p in $PackageDirs) {
        if (-not (Test-Path $p)) {
            Write-Warning "Path not found, skipping: $p"
            continue
        }
        # Determine package root (if caller passed ...\src, use its parent as the project root)
        $pkgRoot = $p
        if ($pkgRoot -like "*\src") {
            $pkgRoot = Split-Path -Parent $pkgRoot
        }

        # If pyproject.toml or setup.py missing, create a minimal pyproject.toml to enable PEP 660 editable installs
        $pyprojectPath = Join-Path $pkgRoot "pyproject.toml"
        $setupPath = Join-Path $pkgRoot "setup.py"
        if (-not (Test-Path $pyprojectPath) -and -not (Test-Path $setupPath)) {
            Write-Host "pyproject.toml not found in $pkgRoot — creating minimal pyproject.toml"
            $content = @"
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "$(Split-Path $pkgRoot -Leaf)"
version = "0.0.1"
description = "Local $(Split-Path $pkgRoot -Leaf) package"
readme = "README.md"
license = {text = "MIT"}

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
"@
            $content | Out-File -FilePath $pyprojectPath -Encoding utf8
            Write-Host "Wrote: $pyprojectPath"
            $createdFiles += $pyprojectPath
        }

        Write-Host "Installing editable package from project root: $pkgRoot"
        pip install -e "$pkgRoot"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "pip install failed for $p (exit $LASTEXITCODE). Aborting."
            exit $LASTEXITCODE
        }
    }
} finally {
    # Clean up any temporary pyproject.toml files we created
    if ($createdFiles.Count -gt 0) {
        Write-Host "Removing temporary pyproject.toml files:`n$($createdFiles -join "`n")"
        foreach ($f in $createdFiles) {
            try {
                Remove-Item -LiteralPath $f -Force -ErrorAction Stop
                Write-Host "Removed: $f"
            } catch {
                Write-Warning "Failed to remove temporary file: $f - $_"
            }
        }
    }
}

Write-Host "Installation finished. Now verifying imports: myafc"
$py = @"
import importlib, sys
name = sys.argv[1]
try:
    m = importlib.import_module(name)
    print('OK', name, getattr(m, '__file__', '<package>'))
except Exception as e:
    print('FAIL', name, e)
"@

$mods = @('myafc')
foreach ($m in $mods) {
    Write-Host "Checking import: $m"
    python -c $py $m
}

Write-Host "Done. If some imports failed, check whether packages contain a proper setup.py/pyproject.toml or __init__.py and adjust paths accordingly."