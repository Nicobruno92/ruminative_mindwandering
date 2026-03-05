---
applyTo: "**"
---
## Git & Provenance Practices

### Rules
- **Atomic Scientism**: Do not mix scientific changes (logic/params) with cosmetic changes (formatting/typos) in the same commit.
- **The "Why" Mandate**: For commits changing scientific parameters, the commit message body must explain the **scientific rationale**.
- **No Checkout**: **It is strictly forbidden** to use `git checkout` to solve development issues or revert changes. Use standard edits and commits.
- **Large Files**: Never commit files > 10MB to Git. Use DataLad or `.gitignore`.
- **Format**: Use Conventional Commits (`feat:`, `fix:`, `docs:`, `exp:` for experiments).

### Checklist
- [ ] No binary/large files in `git status`.
- [ ] Commit messages explain *why* (scientific logic), especially for parameters.
- [ ] No `git checkout` was used.
