### Getting Started with Git and GitHub

1. **Install Git:**
   - **Windows:** Download and install from [git-scm.com](https://git-scm.com).
   - **macOS:** Git usually comes pre-installed. Verify by typing `git --version` in the terminal.
   - **Linux:** Install via package manager, e.g., `sudo apt install git` for Ubuntu.

2. **Set Up Git:**
   - After installation, configure Git with your name and email:
     ```bash
     git config --global user.name "Your Name"
     git config --global user.email "you@example.com"
     ```

3. **Clone the Repository Locally:**
   - To get a copy of the ISEF2425 repo, use:
     ```bash
     git clone https://github.com/tachophobia/ISEF2425.git
     cd ISEF2425
     ```

### Example Use Case of Git

Imagine you're working on a feature. You can **branch off** from the main code, work independently, and merge back only when your changes are complete. This helps keep the main codebase stable.

### Basic Git Commands

| Command                  | Description |
|--------------------------|-------------|
| `git status`             | Check status of files in repo |
| `git add .`              | Stage changes for commit |
| `git commit -m "message"`| Commit changes with a message |
| `git push`               | Push changes to GitHub |
| `git pull`               | Update local repo with GitHub changes |
| `git branch branch_name` | Create a new branch |
| `git checkout branch_name` | Switch to a branch |
| `git merge branch_name` | Merge another branch into your current branch |

### Writing Commit Messages

- **Keep it Professional**: Use clear, concise language.
- **Format**: Start with a verb in the present or past tense (e.g., "Add feature", "Fixed bug").
- **Example**: `git commit -m "Add initial data processing function"`

### Working with Branches

- **Why Branch?** Branches allow you to develop new features without affecting the main code.
- **Creating a Branch**: 
  ```bash
  git branch feature_branch
  git checkout feature_branch
  ```
- **Merging Branches**: After testing, merge back into the main branch.
  ```bash
  git checkout main
  git merge feature_branch
  ```

### Git in VS Code

1. **Source Control Tab**: View changes, stage files, and commit all within VS Code.
2. **Undoing Commits**: You can undo changes by selecting "Discard Changes" or by running `git reset --soft HEAD~1` from the command line.
3. **Syncing Changes**: Pull the latest changes from GitHub using the “Pull” option.

### Commit Frequency and When to Push

- **Commit Often**: Aim to commit every time you reach a stable point.
- **Push Only Clean Code**: Ensure code runs without errors. Use `git pull` before pushing to avoid conflicts.

### Documentation and Code Style

1. **Docstrings**: Every function and class should have docstrings explaining what they do.
   - Example:
     ```python
     def add(a: int, b: int) -> int:
         """
         Adds two numbers together.

         Parameters:
         a (int): First number.
         b (int): Second number.

         Returns:
         int: Sum of a and b.
         """
         return a + b
     ```
   - [Docstring Guide](https://www.programiz.com/python-programming/docstrings)

2. **Type Hinting**: Use type hints for better code readability. Reference the [typing cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html).

3. **Directory Management**: Organize code in modules. Example:
   - If you have functions in `functions/booth.py`, import with:
     ```python
     from functions import booth
     ```

4. **Jupyter Notebooks vs. `.py` Files**:
   - Use notebooks for **drafting and presentations**.
   - Use `.py` files for **data structures and reusable functions**.

### Command Line Basics

- Navigate with `cd folder_name`.
- List files with `ls` (or `dir` on Windows).
- Run Python scripts with `python filename.py`.

By following these practices, you’ll contribute effectively to the **ISEF2425** repository. Remember: clean, well-documented code is key!
