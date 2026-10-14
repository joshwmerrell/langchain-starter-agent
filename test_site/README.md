# Wallace introduction page

A small, dependency-free page describing Wallace, linking to this repository,
and demonstrating two browser interactions: capability cards and a saved
light/dark choice. Wallace itself is still a work in progress.

Build the browser JavaScript from `app.ts` with Deno and a Node release that supports `module.stripTypeScriptTypes` (tested here with Node 26):

```bash
bash test_site/build.sh
```

The build type-checks TypeScript, strips its types into `app.js`, and checks the resulting JavaScript. It runs offline and stops on errors. The HTML loads `app.js`, not the TypeScript source. To preview from the repository root:

```bash
python3 -m http.server --directory test_site 8000
```

Open `http://localhost:8000/`. Run the interaction tests with:

```bash
deno test --no-remote --no-config --no-lock test_site/app_test.ts
```
