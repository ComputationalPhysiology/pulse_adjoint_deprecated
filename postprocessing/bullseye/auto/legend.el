(TeX-add-style-hook
 "legend"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("standalone" "tightpage" "26pt")))
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "pgfplots")
   (TeX-add-symbols
    "addlegendimage")
   (LaTeX-add-environments
    "customlegend")))

