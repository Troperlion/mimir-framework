-- Ajoute le logo du projet en tête des documents générés par Pandoc.
-- L'image reste gérée par Pandoc afin d'être embarquée en PDF et référencée
-- correctement dans la sortie HTML.

local function project_logo()
  local image = pandoc.Image(
    { pandoc.Str("Mímir Framework") },
    "logo.png",
    "Logo Mímir Framework",
    pandoc.Attr("", { "mimir-document-logo" }, { width = "30%" })
  )

  if FORMAT:match("latex") then
    return {
      pandoc.RawBlock("latex", "\\begin{center}"),
      pandoc.Para({ image }),
      pandoc.RawBlock("latex", "\\end{center}\\vspace{1em}"),
    }
  end

  if FORMAT:match("html") then
    return {
      pandoc.Div(
        { pandoc.Para({ image }) },
        pandoc.Attr("mimir-document-logo", {}, {
          style = "text-align:center;margin:1.5rem auto 2rem;",
        })
      ),
    }
  end

  return { pandoc.Para({ image }) }
end

function Pandoc(document)
  local blocks = project_logo()
  for index = #blocks, 1, -1 do
    table.insert(document.blocks, 1, blocks[index])
  end
  return document
end
