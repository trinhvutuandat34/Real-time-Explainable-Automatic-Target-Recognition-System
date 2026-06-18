```markdown
# Real-time-Explainable-Automatic-Target-Recognition-System Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the "Real-time-Explainable-Automatic-Target-Recognition-System" Python codebase. You'll learn how to structure files, write imports and exports, follow commit message conventions, and write and run tests in alignment with the repository's established practices.

## Coding Conventions

### File Naming
- Use **snake_case** for all file and module names.
  - Example: `data_loader.py`, `model_utils.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .preprocessing import normalize_data
    from .models import TargetRecognizer
    ```

### Export Style
- Use **named exports** (explicitly define what is exported from a module).
  - Example:
    ```python
    __all__ = ['TargetRecognizer', 'explain_prediction']
    ```

### Commit Messages
- Follow **conventional commit** format.
- Use the `feat` prefix for new features.
  - Example:
    ```
    feat: add real-time data streaming to recognition pipeline
    ```

## Workflows

### Adding a New Feature
**Trigger:** When you want to introduce new functionality.
**Command:** `/add-feature`

1. Create a new module or function using snake_case naming.
2. Use relative imports to integrate with existing code.
3. Explicitly define exports using `__all__`.
4. Write a conventional commit message starting with `feat:`.
5. Add or update corresponding test files (see Testing Patterns).

### Running Tests
**Trigger:** To verify code correctness after changes.
**Command:** `/run-tests`

1. Locate test files matching the `*.test.*` pattern.
2. Run tests using your preferred Python test runner (e.g., `pytest`, `unittest`).
   - Example:
     ```bash
     pytest
     ```
3. Review test output and fix any failures.

### Refactoring Code
**Trigger:** When improving or restructuring existing code.
**Command:** `/refactor`

1. Rename files and functions to follow snake_case if needed.
2. Update relative imports as necessary.
3. Ensure all exports are explicitly defined.
4. Update or add tests to cover refactored code.
5. Use a conventional commit message (e.g., `feat: refactor data preprocessing pipeline`).

## Testing Patterns

- Test files follow the `*.test.*` naming pattern (e.g., `recognizer.test.py`).
- The specific test framework is not specified; use standard Python test runners.
- Example test file structure:
  ```python
  import unittest
  from .recognizer import TargetRecognizer

  class TestTargetRecognizer(unittest.TestCase):
      def test_prediction(self):
          model = TargetRecognizer()
          result = model.predict(sample_input)
          self.assertEqual(result, expected_output)
  ```

## Commands
| Command        | Purpose                                         |
|----------------|-------------------------------------------------|
| /add-feature   | Scaffold and document a new feature addition    |
| /run-tests     | Run all test files in the repository            |
| /refactor      | Guide through code refactoring steps            |
```