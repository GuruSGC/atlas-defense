# ATLAS Frontend

Adaptive Trust Management for Long-term Agent Storage: Research Prototype UI.

Built with **Next.js 15+ (App Router)**, **TypeScript**, **Tailwind CSS**, and **Lucide Icons**.

## Features

- **Research Dashboard**: System health, active memory counts, empirical security metrics, and live activity stream.
- **Defended Chat Interface**: Real-time conversational agent with candidate memory extraction, six-signal trust score gauge, and ACCEPT/QUARANTINE decision inspection.
- **Memory Monitor**: Filterable table of persistent long-term memories and quarantined adversarial statements.
- **ATLAS Trust Analysis**: Deep-dive diagnostic tool with adjustable signal sensitivity sliders.
- **Attack Lab**: Interactive demonstration suite comparing agent vulnerability with ATLAS ON vs ATLAS OFF across multiple memory poisoning scenarios.
- **Evaluation Benchmark**: Quantitative charts and confusion matrix tables for empirical academic assessment.

## Getting Started

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment**:
   Copy `.env.local.example` to `.env.local`:
   ```bash
   cp .env.local.example .env.local
   ```
   *Note: `NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000`.*

4. **Start the development server**:
   ```bash
   npm run dev
   ```

Open [http://localhost:3000](http://localhost:3000) in your browser.
