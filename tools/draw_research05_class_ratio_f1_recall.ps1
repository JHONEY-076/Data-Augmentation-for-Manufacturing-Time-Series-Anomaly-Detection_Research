param(
    [string]$CsvPath = "data\research05\results\research05_balanced_augmentation_comparison.csv",
    [string]$OutputPath = "data\research05\figures\training_counts_f1_recall_relationship.png"
)

$rows = Import-Csv $CsvPath
Add-Type -AssemblyName System.Drawing

$w = 1700
$h = 900
$bmp = New-Object System.Drawing.Bitmap $w, $h
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g.Clear([System.Drawing.Color]::White)

$titleFont = New-Object System.Drawing.Font("Arial", 24, [System.Drawing.FontStyle]::Bold)
$subtitleFont = New-Object System.Drawing.Font("Arial", 12)
$panelTitleFont = New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)
$font = New-Object System.Drawing.Font("Arial", 12)
$smallFont = New-Object System.Drawing.Font("Arial", 10)
$axisPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(55,55,55), 2)
$gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(228,228,228), 1)
$blackBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(38,38,38))
$mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(95,95,95))

$colors = @{
    traditional = [System.Drawing.Color]::FromArgb(31,119,180)
    generative = [System.Drawing.Color]::FromArgb(214,39,40)
    none = [System.Drawing.Color]::FromArgb(44,160,44)
}
$categoryOrder = @(
    "balanced_original",
    "balanced_augmented",
    "imbalanced_augmented",
    "imbalanced_original"
)
$categoryLabels = @(
    @("Balanced original", "N=247 / A=247", "Total=494"),
    @("Balanced augmented", "N=1,247 / A=1,247", "Total=2,494"),
    @("Augmented imbalanced", "N=5,000 / A=1,247", "Total=6,247"),
    @("Original imbalanced", "N=5,000 / A=247", "Total=5,247")
)
$methods = $rows | Select-Object -ExpandProperty method -Unique | Sort-Object
$offsets = @{}
for ($i = 0; $i -lt $methods.Count; $i++) {
    $offsets[$methods[$i]] = -34 + (68 * $i / [Math]::Max(1, $methods.Count - 1))
}

function Get-CategoryKey {
    param($Row)
    $normal = [int]$Row.normal_train_used
    $anomaly = [int]$Row.total_anomaly_train_used
    $generated = [int]$Row.generated_anomaly_used
    if ($normal -eq 247 -and $anomaly -eq 247 -and $generated -eq 0) { return "balanced_original" }
    if ($normal -eq 1247 -and $anomaly -eq 1247) { return "balanced_augmented" }
    if ($normal -eq 5000 -and $anomaly -eq 1247) { return "imbalanced_augmented" }
    if ($normal -eq 5000 -and $anomaly -eq 247 -and $generated -eq 0) { return "imbalanced_original" }
    return "other"
}

function Get-CategoryIndex {
    param([string]$Key)
    for ($i = 0; $i -lt $categoryOrder.Count; $i++) {
        if ($categoryOrder[$i] -eq $Key) { return $i }
    }
    return -1
}

function Draw-Diamond {
    param(
        [System.Drawing.Graphics]$Graphics,
        [int]$X,
        [int]$Y,
        [int]$R,
        [System.Drawing.Brush]$Brush
    )
    $pts = [System.Drawing.Point[]]@(
        [System.Drawing.Point]::new($X, $Y - $R),
        [System.Drawing.Point]::new($X + $R, $Y),
        [System.Drawing.Point]::new($X, $Y + $R),
        [System.Drawing.Point]::new($X - $R, $Y)
    )
    $Graphics.FillPolygon($Brush, $pts)
}

function Draw-Panel {
    param(
        [string]$Metric,
        [string]$Title,
        [int]$X0,
        [int]$Y0,
        [int]$PanelW,
        [int]$PanelH
    )

    $yMin = 0.30
    $yMax = 0.95
    $plotLeft = $X0 + 75
    $plotTop = $Y0 + 55
    $plotW = $PanelW - 105
    $plotH = $PanelH - 150
    $tickXs = @()
    for ($i = 0; $i -lt $categoryOrder.Count; $i++) {
        $tickXs += [double]($plotLeft + 58 + (($plotW - 116) * $i / [Math]::Max(1, $categoryOrder.Count - 1)))
    }

    $g.DrawString($Title, $panelTitleFont, $blackBrush, $X0 + 75, $Y0 + 14)

    for ($yv = 0.3; $yv -le 0.951; $yv += 0.1) {
        $y = [int]($plotTop + (($yMax - $yv) / ($yMax - $yMin)) * $plotH)
        $g.DrawLine($gridPen, $plotLeft, $y, $plotLeft + $plotW, $y)
        $g.DrawString(("{0:N1}" -f $yv), $font, $blackBrush, $X0 + 22, $y - 10)
    }

    $g.DrawLine($axisPen, $plotLeft, $plotTop, $plotLeft, $plotTop + $plotH)
    $g.DrawLine($axisPen, $plotLeft, $plotTop + $plotH, $plotLeft + $plotW, $plotTop + $plotH)

    for ($i = 0; $i -lt $categoryOrder.Count; $i++) {
        $x = [int]$tickXs[$i]
        $g.DrawLine($axisPen, $x, $plotTop + $plotH, $x, $plotTop + $plotH + 7)
        for ($j = 0; $j -lt $categoryLabels[$i].Count; $j++) {
            $label = $categoryLabels[$i][$j]
            $labelFont = if ($j -eq 0) { $smallFont } else { $smallFont }
            $size = $g.MeasureString($label, $labelFont)
            $g.DrawString($label, $labelFont, $blackBrush, $x - ($size.Width / 2), $plotTop + $plotH + 15 + ($j * 17))
        }
    }

    foreach ($method in $methods) {
        $sub = $rows | Where-Object { $_.method -eq $method } | Sort-Object { Get-CategoryIndex (Get-CategoryKey $_) }
        if ($sub.Count -gt 1) {
            $lineColor = [System.Drawing.Color]::FromArgb(125, 165, 165, 165)
            $linePen = New-Object System.Drawing.Pen($lineColor, 1.4)
            $prevPoint = $null
            foreach ($r in $sub) {
                $idx = Get-CategoryIndex (Get-CategoryKey $r)
                if ($idx -lt 0) { continue }
                $x = [int]($tickXs[$idx] + $offsets[$r.method])
                $val = [double]$r.$Metric
                $y = [int]($plotTop + (($yMax - $val) / ($yMax - $yMin)) * $plotH)
                if ($null -ne $prevPoint) {
                    $g.DrawLine($linePen, $prevPoint[0], $prevPoint[1], $x, $y)
                }
                $prevPoint = @($x, $y)
            }
        }
    }

    foreach ($r in $rows) {
        $idx = Get-CategoryIndex (Get-CategoryKey $r)
        if ($idx -lt 0) { continue }
        $x = [int]($tickXs[$idx] + $offsets[$r.method])
        $val = [double]$r.$Metric
        $y = [int]($plotTop + (($yMax - $val) / ($yMax - $yMin)) * $plotH)
        $color = $colors[$r.augmentation_family]
        $brush = New-Object System.Drawing.SolidBrush($color)
        $whitePen = New-Object System.Drawing.Pen([System.Drawing.Color]::White, 1.5)
        if ($r.condition -eq "balanced") {
            $g.FillEllipse($brush, $x - 7, $y - 7, 14, 14)
            $g.DrawEllipse($whitePen, $x - 7, $y - 7, 14, 14)
        } else {
            $g.FillRectangle($brush, $x - 7, $y - 7, 14, 14)
            $g.DrawRectangle($whitePen, $x - 7, $y - 7, 14, 14)
        }
    }

    $meanPts = @()
    for ($i = 0; $i -lt $categoryOrder.Count; $i++) {
        $target = $categoryOrder[$i]
        $sub = $rows | Where-Object { (Get-CategoryKey $_) -eq $target }
        if ($sub.Count -gt 0) {
            $mean = ($sub | ForEach-Object { [double]$_.$Metric } | Measure-Object -Average).Average
            $x = [int]$tickXs[$i]
            $y = [int]($plotTop + (($yMax - $mean) / ($yMax - $yMin)) * $plotH)
            $meanPts += ,@($x, $y)
            Draw-Diamond $g $x $y 10 (New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(25,25,25)))
            $g.DrawString(("{0:N3}" -f $mean), $smallFont, $blackBrush, $x + 12, $y - 10)
        }
    }
    if ($meanPts.Count -gt 1) {
        $meanPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(25,25,25), 2)
        $meanPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
        for ($i = 0; $i -lt ($meanPts.Count - 1); $i++) {
            $g.DrawLine($meanPen, $meanPts[$i][0], $meanPts[$i][1], $meanPts[$i + 1][0], $meanPts[$i + 1][1])
        }
    }

    $best = $rows | Sort-Object { [double]$_.$Metric } -Descending | Select-Object -First 1
    $bestIdx = Get-CategoryIndex (Get-CategoryKey $best)
    $bx = [int]($tickXs[$bestIdx] + $offsets[$best.method])
    $by = [int]($plotTop + (($yMax - ([double]$best.$Metric)) / ($yMax - $yMin)) * $plotH)
    $g.DrawString(("Best: " + $best.method), $smallFont, $blackBrush, $bx + 12, $by - 22)
}

$g.DrawString("Research05: Training Counts vs F1 and Recall", $titleFont, $blackBrush, 430, 26)
$g.DrawString("Each dot is a method. Black diamonds show the mean score for each training-data composition.", $subtitleFont, $mutedBrush, 455, 66)

Draw-Panel "f1" "F1-score" 55 105 775 700
Draw-Panel "recall" "Recall" 865 105 775 700

$legendFont = New-Object System.Drawing.Font("Arial", 11)
$lx = 610
$ly = 802
$g.DrawString("Family:", $legendFont, $blackBrush, $lx, $ly)
$legendItems = @(@("Traditional","traditional"), @("Generative","generative"), @("Original only","none"))
for ($i = 0; $i -lt $legendItems.Count; $i++) {
    $name = $legendItems[$i][0]
    $key = $legendItems[$i][1]
    $x = $lx + 70 + ($i * 150)
    $b = New-Object System.Drawing.SolidBrush($colors[$key])
    $g.FillEllipse($b, $x, $ly + 2, 13, 13)
    $g.DrawString($name, $legendFont, $blackBrush, $x + 20, $ly - 1)
}
$g.DrawString("Shape:", $legendFont, $blackBrush, $lx + 610, $ly)
$shapeBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(80,80,80))
$g.FillEllipse($shapeBrush, $lx + 670, $ly + 2, 13, 13)
$g.DrawString("balanced", $legendFont, $blackBrush, $lx + 690, $ly - 1)
$g.FillRectangle($shapeBrush, $lx + 785, $ly + 2, 13, 13)
$g.DrawString("imbalanced", $legendFont, $blackBrush, $lx + 805, $ly - 1)
Draw-Diamond $g ($lx + 930) ($ly + 9) 8 (New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(25,25,25)))
$g.DrawString("mean", $legendFont, $blackBrush, $lx + 945, $ly - 1)

$outDir = Split-Path $OutputPath -Parent
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}
$bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()
Write-Output $OutputPath
