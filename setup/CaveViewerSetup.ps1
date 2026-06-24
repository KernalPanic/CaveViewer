<#
.SYNOPSIS
    CaveViewer Setup -- a single "Install" button that takes a non-technical
    user from "freshly downloaded folder" to "double-click icon on Desktop"
    without ever opening a terminal themselves.

.DESCRIPTION
    One button runs three steps in sequence, stopping and re-enabling
    itself if any step fails (rather than silently continuing as if
    nothing went wrong):
      1. Install Python  -- checks if Python is already present; if not,
         downloads the official installer from python.org and runs it
         silently with PATH registration forced on (this exact setting is
         what caused confusion the first time this project was set up
         manually, so the installer here sets it automatically).
      2. Install Requirements -- runs `pip install -r requirements.txt`
         against this project's requirements file, streaming output into
         the on-screen log box.
      3. Create Desktop Shortcut -- writes a .lnk on the Desktop that runs
         `python caveviewer.py` with the working directory set correctly,
         so double-clicking it from the Desktop launches CaveViewer.

    Once all three steps succeed, the window shows a success message
    briefly, then closes itself automatically.

    This script is plain PowerShell + Windows Forms, which ship built into
    Windows 10/11 -- no separate install needed to RUN this setup tool
    itself, which is the whole point (you can't require Python to install
    Python).

.NOTES
    Must be run on Windows. Installing Python system-wide requires
    administrator privileges -- the script requests elevation automatically
    if it isn't already running elevated (a Windows UAC prompt will appear;
    this is expected and necessary).
#>

# -- Self-elevate if not already running as Administrator -------------------
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    $scriptPath = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" -Verb RunAs
    exit
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# -- Paths --------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$MainScript = Join-Path $ProjectRoot "caveviewer.py"

$PythonInstallerUrl = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
$PythonInstallerPath = Join-Path $env:TEMP "python-installer-caveviewer.exe"

# -- Form setup -----------------------------------------------------------------

$form = New-Object System.Windows.Forms.Form
$form.Text = "CaveViewer Setup"
$form.Size = New-Object System.Drawing.Size(560, 420)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false

$titleLabel = New-Object System.Windows.Forms.Label
$titleLabel.Text = "CaveViewer 1.0 -- Setup"
$titleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
$titleLabel.Location = New-Object System.Drawing.Point(20, 15)
$titleLabel.Size = New-Object System.Drawing.Size(500, 30)
$form.Controls.Add($titleLabel)

$subLabel = New-Object System.Windows.Forms.Label
$subLabel.Text = "Click Install. This window will tell you what's happening at every step."
$subLabel.Location = New-Object System.Drawing.Point(20, 48)
$subLabel.Size = New-Object System.Drawing.Size(500, 20)
$form.Controls.Add($subLabel)

# Single Install button -- runs all three steps in sequence.
$btnInstall = New-Object System.Windows.Forms.Button
$btnInstall.Text = "Install"
$btnInstall.Location = New-Object System.Drawing.Point(20, 80)
$btnInstall.Size = New-Object System.Drawing.Size(500, 44)
$btnInstall.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$form.Controls.Add($btnInstall)

# Log box
$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ScrollBars = "Vertical"
$logBox.ReadOnly = $true
$logBox.Font = New-Object System.Drawing.Font("Consolas", 9)
$logBox.Location = New-Object System.Drawing.Point(20, 135)
$logBox.Size = New-Object System.Drawing.Size(500, 245)
$logBox.BackColor = [System.Drawing.Color]::Black
$logBox.ForeColor = [System.Drawing.Color]::LightGreen
$form.Controls.Add($logBox)

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $logBox.AppendText("[$timestamp] $Message`r`n")
    $logBox.SelectionStart = $logBox.Text.Length
    $logBox.ScrollToCaret()
    [System.Windows.Forms.Application]::DoEvents()
}

function Invoke-CaveViewerDownload {
    <#
        Downloads $Url to $DestinationPath using System.Net.WebClient with
        a real timeout, rather than Invoke-WebRequest.

        Why not Invoke-WebRequest: its default progress-bar rendering is
        known to be extremely slow on Windows PowerShell 5.1 specifically
        (a long-standing, widely-reported issue) -- each progress update
        can add massive overhead to a large download, sometimes making a
        download that should take well under a minute appear to hang
        indefinitely with the window showing "Not Responding" and no way
        to recover except force-closing the whole setup tool. WebClient
        avoids that progress-rendering path entirely.

        Returns $true on success, $false on failure/timeout (network
        issue, server error, or exceeding $TimeoutSeconds) -- never
        throws, so callers can just check the return value rather than
        wrapping every call site in their own try/catch.
    #>
    param(
        [string]$Url,
        [string]$DestinationPath,
        [int]$TimeoutSeconds = 180
    )

    try {
        $webClient = New-Object System.Net.WebClient
        $downloadTask = $webClient.DownloadFileTaskAsync($Url, $DestinationPath)

        $elapsed = 0.0
        $lastLoggedSecond = -1
        while (-not $downloadTask.IsCompleted -and $elapsed -lt $TimeoutSeconds) {
            Start-Sleep -Milliseconds 500
            $elapsed += 0.5
            # log progress roughly every 5 seconds rather than every poll,
            # so the log box doesn't fill up with near-duplicate lines
            $currentSecond = [math]::Floor($elapsed)
            if (($currentSecond % 5 -eq 0) -and ($currentSecond -ne $lastLoggedSecond)) {
                $lastLoggedSecond = $currentSecond
                $sizeSoFar = if (Test-Path $DestinationPath) { (Get-Item $DestinationPath).Length } else { 0 }
                Write-Log "  ...downloading, $([math]::Round($sizeSoFar / 1MB, 1)) MB so far"
            }
        }

        if (-not $downloadTask.IsCompleted) {
            Write-Log "WARNING: download timed out after $TimeoutSeconds seconds."
            $webClient.CancelAsync()
            $webClient.Dispose()
            Remove-Item $DestinationPath -Force -ErrorAction SilentlyContinue
            return $false
        }

        if ($downloadTask.IsFaulted) {
            $innerMessage = if ($downloadTask.Exception -and $downloadTask.Exception.InnerException) {
                $downloadTask.Exception.InnerException.Message
            } else {
                "unknown error"
            }
            Write-Log "WARNING: download failed: $innerMessage"
            $webClient.Dispose()
            Remove-Item $DestinationPath -Force -ErrorAction SilentlyContinue
            return $false
        }

        $webClient.Dispose()
        return $true
    } catch {
        Write-Log "WARNING: download failed: $($_.Exception.Message)"
        Remove-Item $DestinationPath -Force -ErrorAction SilentlyContinue
        return $false
    }
}

function Test-PythonInstalled {
    <#
        Checks for a working `python` command on PATH. Returns the version
        string if found, or $null if not found / not runnable.

        Hardened against a known Windows quirk: on a fresh system with no
        Python installed, typing `python` doesn't always fail cleanly --
        Windows sometimes routes it through a built-in "app execution
        alias" stub that opens the Microsoft Store instead of erroring.
        That stub typically lives under WindowsApps and is a few KB in
        size (a real Python install is not), so as a second check beyond
        just "did the command run", we also reject suspiciously tiny
        python.exe files living in that specific stub location.
    #>
    try {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -like "*\WindowsApps\python.exe") {
            $fileInfo = Get-Item $cmd.Source -ErrorAction SilentlyContinue
            if ($fileInfo -and $fileInfo.Length -lt 100000) {
                return $null
            }
        }

        $output = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            return $output.ToString().Trim()
        }
    } catch {
        # python not found on PATH at all -- expected, not an error to surface
    }
    return $null
}

# -- Step functions -----------------------------------------------------------
# Each returns $true on success, $false on failure. The single Install
# button below runs them in order and stops at the first failure, rather
# than barreling ahead and pretending a failed step succeeded.

function Install-Python {
    Write-Log "Checking for an existing Python installation..."

    $existing = Test-PythonInstalled
    if ($existing) {
        Write-Log "Found existing Python: $existing -- skipping install."
        return $true
    }

    Write-Log "Python not found. Downloading the official installer from python.org..."
    Write-Log "(This may take a minute depending on your internet connection.)"

    $downloadOk = Invoke-CaveViewerDownload -Url $PythonInstallerUrl -DestinationPath $PythonInstallerPath -TimeoutSeconds 120
    if (-not $downloadOk) {
        Write-Log "ERROR: Failed to download the Python installer. Check your internet connection and try again."
        return $false
    }

    Write-Log "Download complete. Running the installer silently..."
    Write-Log "(InstallAllUsers + PrependPath are set automatically -- this is the"
    Write-Log " 'Add Python to PATH' step that has to be checked manually otherwise.)"

    $installArgs = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0"
    $proc = Start-Process -FilePath $PythonInstallerPath -ArgumentList $installArgs -Wait -PassThru

    if ($proc.ExitCode -ne 0) {
        Write-Log "ERROR: Python installer exited with code $($proc.ExitCode)."
        return $false
    }

    Write-Log "Python installed successfully."

    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"

    Start-Sleep -Seconds 1
    $verify = Test-PythonInstalled
    if ($verify) {
        Write-Log "Verified: $verify"
        return $true
    }

    Write-Log "WARNING: Python installed, but couldn't be verified in this window."
    Write-Log "This can happen if Windows hasn't fully refreshed PATH yet -- continuing anyway."
    return $true
}

function Install-Requirements {
    Write-Log "Installing required packages from requirements.txt..."
    Write-Log "(moderngl, moderngl-window, numpy, Pillow, pygltflib -- this may take a minute.)"

    if (-not (Test-Path $RequirementsFile)) {
        Write-Log "ERROR: Could not find requirements.txt at:"
        Write-Log "  $RequirementsFile"
        Write-Log "Make sure this setup folder is still inside the CaveViewer project folder."
        return $false
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "python"
    $psi.Arguments = "-m pip install -r `"$RequirementsFile`""
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi

    $outputHandler = {
        if (-not [string]::IsNullOrEmpty($EventArgs.Data)) {
            Write-Log $EventArgs.Data
        }
    }
    Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action $outputHandler | Out-Null
    Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -Action $outputHandler | Out-Null

    $proc.Start() | Out-Null
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()
    $proc.WaitForExit()

    Get-EventSubscriber | Where-Object { $_.SourceObject -eq $proc } | Unregister-Event

    if ($proc.ExitCode -ne 0) {
        Write-Log "ERROR: pip install failed (exit code $($proc.ExitCode))."
        return $false
    }

    Write-Log "All requirements installed successfully."
    return $true
}

function New-DesktopShortcut {
    Write-Log "Creating a CaveViewer shortcut on your Desktop..."

    try {
        $desktopPath = [System.Environment]::GetFolderPath("Desktop")
        $shortcutPath = Join-Path $desktopPath "CaveViewer.lnk"

        $pythonPath = (Get-Command python).Source
        $iconPath = Join-Path $ScriptDir "icon\caveviewer.ico"

        $wshShell = New-Object -ComObject WScript.Shell
        $shortcut = $wshShell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $pythonPath
        $shortcut.Arguments = "`"$MainScript`""
        $shortcut.WorkingDirectory = $ProjectRoot
        $shortcut.Description = "Launch CaveViewer"

        if (Test-Path $iconPath) {
            $shortcut.IconLocation = "$iconPath,0"
            Write-Log "Using custom CaveViewer icon."
        } else {
            $shortcut.IconLocation = "$pythonPath,0"
            Write-Log "Custom icon not found at $iconPath -- using default icon instead."
        }

        $shortcut.Save()
        Write-Log "Shortcut created: $shortcutPath"
        return $true
    } catch {
        Write-Log "ERROR: Failed to create the desktop shortcut."
        Write-Log $_.Exception.Message
        return $false
    }
}

# -- Single Install button: runs all three steps in sequence -----------------

$btnInstall.Add_Click({
    $btnInstall.Enabled = $false
    $btnInstall.Text = "Installing..."

    $ok = Install-Python
    if (-not $ok) {
        Write-Log ""
        Write-Log "Setup stopped -- Python installation failed. Click Install to try again."
        $btnInstall.Enabled = $true
        $btnInstall.Text = "Install"
        return
    }

    $ok = Install-Requirements
    if (-not $ok) {
        Write-Log ""
        Write-Log "Setup stopped -- installing requirements failed. Click Install to try again."
        $btnInstall.Enabled = $true
        $btnInstall.Text = "Install"
        return
    }

    $ok = New-DesktopShortcut
    if (-not $ok) {
        Write-Log ""
        Write-Log "Setup stopped -- could not create the desktop shortcut. Click Install to try again."
        $btnInstall.Enabled = $true
        $btnInstall.Text = "Install"
        return
    }

    Write-Log ""
    Write-Log "All done!"
    $btnInstall.Text = "Done"
    $btnInstall.Enabled = $false

    Show-InstallCompleteDialog
})

function Show-InstallCompleteDialog {
    <#
        Shown once setup finishes successfully -- replaces the previous
        behavior of just logging a success message and auto-closing after
        a few seconds. Gives the person an explicit choice: launch
        CaveViewer right now, or just close the setup window and launch
        it later from the Desktop shortcut.
    #>
    $dialog = New-Object System.Windows.Forms.Form
    $dialog.Text = "CaveViewer Setup"
    $dialog.Size = New-Object System.Drawing.Size(420, 200)
    $dialog.StartPosition = "CenterParent"
    $dialog.FormBorderStyle = "FixedDialog"
    $dialog.MaximizeBox = $false
    $dialog.MinimizeBox = $false

    $msgLabel = New-Object System.Windows.Forms.Label
    $msgLabel.Text = "Installation was successful!"
    $msgLabel.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
    $msgLabel.Location = New-Object System.Drawing.Point(20, 20)
    $msgLabel.Size = New-Object System.Drawing.Size(380, 30)
    $dialog.Controls.Add($msgLabel)

    $subLabel = New-Object System.Windows.Forms.Label
    $subLabel.Text = "Would you like to launch CaveViewer now?"
    $subLabel.Location = New-Object System.Drawing.Point(20, 55)
    $subLabel.Size = New-Object System.Drawing.Size(380, 24)
    $dialog.Controls.Add($subLabel)

    $btnLaunch = New-Object System.Windows.Forms.Button
    $btnLaunch.Text = "Launch CaveViewer"
    $btnLaunch.Location = New-Object System.Drawing.Point(50, 110)
    $btnLaunch.Size = New-Object System.Drawing.Size(150, 36)
    $btnLaunch.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
    $dialog.Controls.Add($btnLaunch)

    $btnClose = New-Object System.Windows.Forms.Button
    $btnClose.Text = "Close"
    $btnClose.Location = New-Object System.Drawing.Point(220, 110)
    $btnClose.Size = New-Object System.Drawing.Size(150, 36)
    $dialog.Controls.Add($btnClose)

    $btnLaunch.Add_Click({
        try {
            $pythonPath = (Get-Command python).Source
            Start-Process -FilePath $pythonPath -ArgumentList "`"$MainScript`"" -WorkingDirectory $ProjectRoot
        } catch {
            Write-Log "WARNING: Could not launch CaveViewer automatically: $($_.Exception.Message)"
            Write-Log "You can still double-click the CaveViewer icon on your Desktop."
        }
        $dialog.Close()
    })

    $btnClose.Add_Click({
        $dialog.Close()
    })

    # Treat every way this dialog can close (Launch, Close, or the
    # window's own [X] button) the same way: close the now-pointless
    # setup window behind it. Handling this once here, rather than
    # calling $form.Close() separately inside each button's own handler,
    # avoids closing $form twice over for the buttons (FormClosed fires
    # for ANY dismissal of $dialog, including a programmatic .Close()
    # call from a button handler, not just the titlebar [X]).
    $dialog.Add_FormClosed({
        $form.Close()
    })

    [void]$dialog.ShowDialog($form)
}

Write-Log "Welcome to CaveViewer Setup."
Write-Log "Click Install to set up Python, the required libraries, and a Desktop shortcut."

[void]$form.ShowDialog()
