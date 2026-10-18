const vscode = require("vscode");
const stub = require("./mpk-stub.json");

function item(label, detail, kind, insertText) {
  const completion = new vscode.CompletionItem(label, kind);
  completion.detail = detail;
  if (insertText) completion.insertText = new vscode.SnippetString(insertText);
  return completion;
}

function activate(context) {
  const selector = { language: "mimir-mpk" };
  const completion = vscode.languages.registerCompletionItemProvider(
    selector,
    {
      provideCompletionItems() {
        const items = [];
        for (const [name, documentation] of Object.entries(stub.keywords)) {
          const entry = item(name, documentation, vscode.CompletionItemKind.Keyword);
          entry.documentation = new vscode.MarkdownString(documentation);
          items.push(entry);
        }
        for (const [name, documentation] of Object.entries(stub.methods)) {
          const insert = name === "set"
            ? "set(\"${1:clé}\", ${2:valeur})"
            : "append(${1:valeur})";
          const entry = item(name, documentation, vscode.CompletionItemKind.Method, insert);
          entry.documentation = new vscode.MarkdownString(documentation);
          items.push(entry);
        }
        for (const [name, documentation] of Object.entries(stub.fields)) {
          const entry = item(name, documentation, vscode.CompletionItemKind.Field);
          entry.documentation = new vscode.MarkdownString(documentation);
          items.push(entry);
        }
        for (const name of stub.layerTypes) {
          items.push(item(
            name,
            `LayerType Mímir : ${name}`,
            vscode.CompletionItemKind.EnumMember,
            `\"${name}\"`
          ));
        }
        return items;
      }
    },
    ".", "\""
  );

  const hover = vscode.languages.registerHoverProvider(selector, {
    provideHover(document, position) {
      const range = document.getWordRangeAtPosition(position);
      if (!range) return undefined;
      const word = document.getText(range);
      const documentation =
        stub.keywords[word] || stub.methods[word] || stub.fields[word];
      if (documentation) {
        return new vscode.Hover(new vscode.MarkdownString(documentation), range);
      }
      if (stub.layerTypes.includes(word)) {
        return new vscode.Hover(
          new vscode.MarkdownString(`**${word}** — LayerType reconnu par Mímir.`),
          range
        );
      }
      return undefined;
    }
  });

  context.subscriptions.push(completion, hover);
}

function deactivate() {}

module.exports = { activate, deactivate };
