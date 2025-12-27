# .latexmkrc
$pdf_mode = 1; 
$out_dir = '.';
$aux_dir = '.aux';
$emulate_aux = 1;

# TeX Live 2025 specific: Ensure shell-escape is on for minted
$pdflatex = 'pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Biber configuration
$biber = 'biber --input-directory .aux --output-directory .aux %O %S';