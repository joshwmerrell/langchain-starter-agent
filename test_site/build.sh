#!/usr/bin/env bash
set -euo pipefail

site_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Fail on TypeScript errors before replacing the browser script. This page has
# no imports, so Node's built-in type stripper can emit JavaScript offline.
deno check --no-remote --no-config --no-lock "$site_dir/app.ts"
temporary="$(mktemp "$site_dir/.app.js.XXXXXX.mjs")"
trap 'rm -f -- "$temporary"' EXIT
node -e 'const fs = require("node:fs"); const {stripTypeScriptTypes} = require("node:module"); fs.writeFileSync(process.argv[2], stripTypeScriptTypes(fs.readFileSync(process.argv[1], "utf8"), {mode: "strip"}));' \
  "$site_dir/app.ts" "$temporary"
deno fmt --no-config --no-editorconfig "$temporary"
node --check "$temporary"
chmod 644 "$temporary"
mv -- "$temporary" "$site_dir/app.js"
trap - EXIT
