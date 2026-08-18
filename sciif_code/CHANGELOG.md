# Changelog

## [1.0.0] - 2024-01-XX

### Added
- Added `--validator_mode` command-line argument to select strict/loose validation mode
- Added `--use_multi_judge` flag to enable multi-model voting mechanism
- Added `--judge_models` argument to specify models for multi-judge voting
- Multi-model judge support for both single and multi-constraint validation
- Improved logging to show validator mode and multi-judge configuration

### Changed
- Validator mode can now be set via command-line argument (overrides environment variable)
- Multi-judge is now available as a command-line option instead of only code-level configuration

### Fixed
- Fixed parameter passing for multi-judge functionality through the evaluation pipeline

