
.PHONY: all build build-static clean mk-video build-docs

FLAGS= -ffp-contract=fast \
	-funroll-loops \
	-fno-trapping-math \
	-fno-math-errno \
	-mf16c \
	-mbmi2 \
	-DUSE_HUGEPAGES

DOCS_OUTPUT_DIR := docs/output
DOCS_PDF := $(DOCS_OUTPUT_DIR)/Documentation_Mimir-FRAMEWORK.pdf
DOCS_HTML := $(DOCS_OUTPUT_DIR)/Documentation_Mimir-FRAMEWORK.html

# Sources doc (nouvelle structure).
# Ordre: index -> getting started -> user guide -> API -> internals -> advanced -> contributing.
DOCS_SOURCES := \
	docs/00-INDEX.md \
	$(sort $(wildcard docs/01-Getting-Started/*.md)) \
	$(sort $(wildcard docs/02-User-Guide/*.md)) \
	$(sort $(wildcard docs/03-API-Reference/*.md)) \
	$(sort $(wildcard docs/04-Architecture-Internals/*.md)) \
	$(sort $(wildcard docs/05-Advanced/*.md)) \
	$(sort $(wildcard docs/06-Contributing/*.md)) \
	docs/graphs/README.md

all: build

build:
	@echo "🏗️  Build via CMake (recommandé)"
	@cmake -S . -B build
	@cmake --build build -j

build-static:
	@echo "🏗️  Build statique: mimir_static (sortie dans ./bin)"
	@cmake -S . -B build_static -DBUILD_MIMIR_STATIC=ON
	@cmake --build build_static -j --target mimir_static



mk-video:
	@ffmpeg -framerate 8 -f image2 -pattern_type glob -i "generated_epoch_*.pgm" \
       -vf "format=yuv420p" -c:v libx264 -preset veryslow -crf 0 \
       output_raw.avi

clean:
	@echo "🧹 Nettoyage des fichiers de compilation..."
	@rm -f bin/*
	@rm -f src/*.o
	@rm -f *.o
	@echo "✓ Nettoyage terminé"

build-docs:
	@echo "📚 Génération de la documentation PDF..."
	@if ! command -v pandoc >/dev/null 2>&1; then \
		echo "❌ Erreur: pandoc n'est pas installé"; \
		echo "   Installer avec: sudo apt install pandoc texlive-xetex texlive-latex-extra"; \
		exit 1; \
	fi
	@mkdir -p $(DOCS_OUTPUT_DIR)
	@set -e; \
	if pandoc $(DOCS_SOURCES) \
		-o $(DOCS_PDF) \
		--pdf-engine=xelatex \
		--include-in-header docs/pandoc/header.tex \
		--toc \
		--toc-depth=3 \
		--number-sections \
		-V geometry:a4paper \
		-V geometry:margin=2.5cm \
		-V fontsize=11pt \
		-V mainfont="DejaVu Serif" \
		-V sansfont="DejaVu Sans" \
		-V monofont="DejaVu Sans Mono" \
		-V monofontoptions=Scale=0.92 \
		-V documentclass=report \
		--highlight-style=tango \
		--title-prefix "Mimir Framework"; then \
		echo "✓ Documentation PDF générée: $(DOCS_PDF)"; \
		ls -lh $(DOCS_PDF) | awk '{print "  Taille:", $$5}'; \
		exit 0; \
	fi; \
	echo "⚠️  Échec xelatex, tentative wkhtmltopdf..."; \
	if command -v wkhtmltopdf >/dev/null 2>&1; then \
		pandoc $(DOCS_SOURCES) \
			-o $(DOCS_PDF) \
			--pdf-engine=wkhtmltopdf \
			--toc \
			--toc-depth=3; \
		echo "✓ Documentation PDF générée: $(DOCS_PDF)"; \
		ls -lh $(DOCS_PDF) | awk '{print "  Taille:", $$5}'; \
		exit 0; \
	fi; \
	echo "❌ Aucun moteur PDF disponible (xelatex/wkhtmltopdf)"; \
	echo "   Génération HTML à la place..."; \
	pandoc $(DOCS_SOURCES) \
		-o $(DOCS_HTML) \
		--toc \
		--toc-depth=3 \
		--standalone \
		--css=https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css; \
	echo "✓ Documentation HTML générée: $(DOCS_HTML)"; \
	ls -lh $(DOCS_HTML) | awk '{print "  Taille:", $$5}'
