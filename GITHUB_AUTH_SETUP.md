# GitHub Authentication Setup

## Option 1: Personal Access Token (PAT) - Recommended

### Step 1: Create Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Direct link: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "hello-keyword-spotting"
4. Select scopes:
   - ✅ `repo` (Full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)

### Step 2: Update Remote URL with Token

```bash
git remote set-url origin https://YOUR_TOKEN@github.com/dilipnikhil/Edge-Inference-hello.git
```

Replace `YOUR_TOKEN` with the token you just created.

### Step 3: Push Again

```bash
git push -u origin main
```

When prompted for username: enter `dilipnikhil`
When prompted for password: paste your **token** (not your GitHub password)

---

## Option 2: Use SSH (More Secure, Long-term Solution)

### Step 1: Generate SSH Key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter for default location. Optionally set a passphrase.

### Step 2: Add SSH Key to GitHub

1. Copy your public key:
```bash
# Windows PowerShell
cat ~/.ssh/id_ed25519.pub
# Or Windows CMD
type %USERPROFILE%\.ssh\id_ed25519.pub
```

2. Go to GitHub → Settings → SSH and GPG keys
   - Direct link: https://github.com/settings/keys
3. Click "New SSH key"
4. Paste your public key
5. Click "Add SSH key"

### Step 3: Update Remote to SSH

```bash
git remote set-url origin git@github.com:dilipnikhil/Edge-Inference-hello.git
```

### Step 4: Push

```bash
git push -u origin main
```

---

## Option 3: Use Git Credential Manager (Easiest for Windows)

### Step 1: Install Git Credential Manager (if not already installed)

Download from: https://github.com/GitCredentialManager/git-credential-manager/releases

### Step 2: Update Remote URL

```bash
git remote set-url origin https://github.com/dilipnikhil/Edge-Inference-hello.git
```

### Step 3: Push (will prompt for authentication)

```bash
git push -u origin main
```

A browser window will open for GitHub authentication. Follow the prompts.

---

## Quick Fix (Using PAT in URL)

**Fastest way right now:**

1. Get your PAT from: https://github.com/settings/tokens
2. Run this command (replace YOUR_TOKEN):

```bash
git remote set-url origin https://YOUR_TOKEN@github.com/dilipnikhil/Edge-Inference-hello.git
git push -u origin main
```

Enter your GitHub username when prompted (token is already in URL).

---

## Alternative: Use GitHub CLI

If you have GitHub CLI installed:

```bash
gh auth login
gh repo create Edge-Inference-hello --public --source=. --remote=origin --push
```

