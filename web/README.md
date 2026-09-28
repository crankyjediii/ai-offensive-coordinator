# Web interface

Next.js 16 (App Router, TypeScript) client for the FastAPI service. It never recomputes EPA, utility or confidence; it renders the API payload.

```bash
pnpm install
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8010 pnpm dev
```

Pages: Overview, Analyst (scouting report), Playcaller (broad-family comparison), Evaluation (model card), Data & Methodology.
Provenance is encoded by texture and text: solid = observed, hatched = predicted, dashed outline = unavailable (never a zero bar).
