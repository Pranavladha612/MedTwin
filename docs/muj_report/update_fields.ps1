# Fills in the page numbers (PAGEREF fields) of the MUJ-format report with
# Microsoft Word, saves the .docx and exports the PDF.
#
#   powershell -ExecutionPolicy Bypass -File docs\muj_report\update_fields.ps1

param([string]$Docx = (Join-Path $PSScriptRoot "..\MedTwin_Project_Report_MUJ.docx"))
$full = (Resolve-Path $Docx).Path
$pdf = [IO.Path]::ChangeExtension($full, ".pdf")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($full)
    # Twice: the first pass can change line breaks and so page numbers
    1..2 | ForEach-Object {
        $doc.Repaginate()
        [void]$doc.Fields.Update()
    }
    $doc.Save()
    $doc.SaveAs([ref]$pdf, [ref]17)
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close([ref]0)
}
finally {
    $word.Quit()
}
"saved $pdf"
