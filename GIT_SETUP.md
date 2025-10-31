# Git Setup Instructions

## Initial Setup (First Time)

### 1. Initialize Git Repository

```bash
git init
```

### 2. Add All Files

```bash
git add .
```

### 3. Create Initial Commit

```bash
git commit -m "Initial commit: Real-time hello keyword detection system with GUI"
```

### 4. Add Remote Repository

First, create a new repository on GitHub (don't initialize with README), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

Replace:
- `YOUR_USERNAME` with your GitHub username
- `YOUR_REPO_NAME` with your repository name (e.g., `hello-keyword-spotting`)

### 5. Push to GitHub

```bash
git branch -M main
git push -u origin main
```

## If Repository Already Exists

If you already created a repo on GitHub with a README:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Quick One-Liner (After Setup)

For future updates:

```bash
git add .
git commit -m "Update: your commit message here"
git push
```

## Common Git Commands

```bash
# Check status
git status

# View changes
git diff

# View commit history
git log --oneline

# Update .gitignore if needed
# Then:
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
```

