### Contributing to Summaraii

Thank you for your interest in improving Summaraii! Contributions of all kinds are welcome.

### Ways to contribute
- Open issues for bugs, docs, or feature requests
- Tackle a good first issue (see labels)
- Improve documentation and examples

### Development setup
1. Fork and clone the repo
2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run cs.py
```

### Pull request guidelines
- Create a feature branch from `main`
- Keep PRs focused and small
- Add or update docs as needed
- Ensure the app runs locally without errors

### Coding conventions
- Prefer clear, descriptive names and straightforward control flow
- Avoid unnecessary try/except blocks
- Keep comments concise and meaningful

### Commit messages
- Use imperative mood: "Add X", "Fix Y"
- Reference issues when applicable: `Fixes #123`

### Release notes
- If your change is user-facing, summarize it in the PR description

