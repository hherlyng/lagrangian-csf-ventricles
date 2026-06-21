FILE=main

default:
	pdflatex $(FILE).tex

final:
	pdflatex $(FILE).tex
	bibtex $(FILE)
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

diff:
	latexdiff main_v0.tex main.tex > diff.tex
	pdflatex diff.tex
	bibtex diff
	pdflatex diff.tex
	pdflatex diff.tex

clean:
	rm *~ *.aux *.out *.toc *.blg *.bbl *.log *.tdo *.dvi
