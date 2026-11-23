MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex ulb-template.sty bibliography.bib
	lualatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	biber $(MAIN)
	lualatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	lualatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.bcf *.blg *.run.xml *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean