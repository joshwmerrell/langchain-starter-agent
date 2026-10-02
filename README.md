## LangChain Starter Agent

Simple LangChain agent you can improve over time.

## Setup (one time)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Create a `.env` file in this folder with your API key:

   ```
   GOOGLE_API_KEY=your-key-here
   ```

3. From this folder, install the `agent` command:

   ```
   uv tool install --editable .
   uv tool update-shell
   ```

4. Close and reopen your terminal.

## Run it

Open any terminal and type:

```
agent
```

Type `exit` to quit.

Because the install is *editable*, changes you make to the code show up the
next time you run `agent`. There's no need to reinstall.

The agent loads LangChain documentation tools from
`https://docs.langchain.com/mcp` and also has read-only access to project files
through `list_directory` and `read_safe_file`. File access is restricted to
this project directory, and `.env` files are excluded.

The CLI shows a status indicator while it connects to documentation tools and
waits for a response. Google GenAI does not support the MCP schemas'
`additionalProperties` flag; the agent filters that specific compatibility
warning while tool input validation remains in place.

If you add a new dependency to `pyproject.toml`, run this again from this folder:

```
uv tool install --editable . --reinstall
```
