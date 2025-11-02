# Cloud-Native Laughter Detection and Emotion Analytics Platform

This repository contains Hashnode-ready documentation (Markdown + Mermaid diagrams) for the MSc project.

## Structure
```
hashnode-docs/
 ├── README.md
 ├── _config.yml
 └── docs/
     ├── index.md
     ├── background.md
     ├── methodology.md
     ├── system-architecture.md
     ├── implementation.md
     ├── evaluation.md
     ├── results.md
     ├── conclusion.md
     └── references.md
```

## How to publish on Hashnode (Basic)
1. Create a new **Hashnode** blog or use an existing one.
2. In **Dashboard → Integrations → GitHub**, connect your GitHub account.
3. Push this `hashnode-docs/` folder to a GitHub repository.
4. Create a **Docs** project (Hashnode) and select this repo. Set `docs/` as your docs root.
5. Enable **Mermaid** in your blog settings (Hashnode supports Mermaid in Markdown).
6. Publish!

## How to publish on GitHub Pages (Basic)
1. Create a new **GitHub repository** (public recommended).
2. Push this folder:
   ```bash
   git init
   git add .
   git commit -m "Add Hashnode docs"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
3. In **Repo → Settings → Pages**, set the branch to `main` (root or `/`).
4. Your site will be available at `https://<username>.github.io/<repo>/`.

> For advanced hosting (Docker/Nginx/AWS) you can extend later.


## Cover Image
Use the banner below for your Hashnode/README cover:

- SVG: `assets/cover.svg`
- PNG: `assets/cover.png`

![Cover](assets/cover.png)

## GitHub Pages Auto-Publish (Workflow)
This repository includes `.github/workflows/publish-docs.yml` which deploys the **docs/** folder to **GitHub Pages** on every push to `main`.

**Setup once:**
1. Go to **Repo → Settings → Pages**.
2. Under *Source*, choose **GitHub Actions**.
3. Push to `main` — your site will auto-deploy.

