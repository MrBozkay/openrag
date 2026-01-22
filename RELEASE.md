# OpenRAG Release Procedures

## Release Steps

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create GitHub release
5. PyPI publish happens automatically

## Release Types

- **Patch**: Bug fixes (0.1.0 → 0.1.1)
- **Minor**: New features (0.1.0 → 0.2.0)
- **Major**: Breaking changes (0.1.0 → 1.0.0)

## Emergency Release

For critical security patches:
1. Create hotfix branch from main
2. Apply fix
3. Bump PATCH version
4. Fast-track merge to main
5. Create release immediately
