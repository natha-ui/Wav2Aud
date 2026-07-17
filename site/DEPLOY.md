# Putting the wave2aud studio on the internet

The whole site is a single self-contained file: **`site/index.html`** (~1.8 MB,
audio embedded, no external requests). Everything — the sonification engine, the
3-D visual, file upload — runs in the visitor's browser, so you only need
**static hosting** (no server, no backend). Pick one:

## Option A — Netlify Drop (easiest, ~1 minute, free)
1. Go to <https://app.netlify.com/drop>.
2. Drag the **`site`** folder onto the page.
3. You instantly get a public URL like `https://random-name.netlify.app`.
   (Free Netlify account lets you rename it and add a custom domain.)

## Option B — GitHub Pages (best if you want version control, free)
This repo ships `.github/workflows/pages.yml`, which publishes the `site/`
folder automatically on every push to `main`.

1. Push the repo to GitHub.
2. Repo → **Settings → Pages** → *Build and deployment* → Source:
   **GitHub Actions**.
3. That's it — each push redeploys. It's live at
   `https://<your-username>.github.io/<repo>/`.

> Note: the "Deploy from a branch" option can only serve the repository **root**
> or a **/docs** folder — it cannot serve `/site`. That's why we use the Actions
> workflow. (If you'd rather not use Actions, copy `site/index.html` to the repo
> root and choose *Deploy from a branch → main → / (root)*.)

```bash
# from the project root
git init && git add . && git commit -m "wave2aud studio"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
# then set Settings -> Pages -> Source: GitHub Actions
```

## Option C — Cloudflare Pages / Vercel (free, fast CDN)
Connect the GitHub repo (or drag-and-drop the `site` folder). No build command;
set the output/publish directory to `site`.

## Option D — one command from your machine
```bash
npm install --global surge
cd site && surge      # gives you a *.surge.sh URL (or your own domain)
```

## Custom domain (optional)
Buy a domain (Namecheap, Cloudflare, Google Domains, …) and, in your host's
dashboard, add the domain and follow its DNS instructions (usually a `CNAME`
record). All the hosts above provide free HTTPS automatically.

## Updating the site later
Re-run the build (it regenerates `site/index.html` automatically), then
re-deploy (drag again / `git push` / `surge`):
```bash
python docs/build_showcase.py
```

## Notes
- **HTTPS is recommended** (all the hosts above give it free). Audio playback and
  file reading work fine on any https origin.
- Nothing the visitor uploads leaves their device — the page never sends data
  anywhere, so there are no privacy or storage concerns to manage.
- The `.wav`/image files a visitor drops are processed in memory only.
