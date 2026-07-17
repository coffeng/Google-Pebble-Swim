/**
 * Local file server for serving Pebble JSON files from disk.
 * Used when running on Windows desktop (outside of Google AI Studio).
 */
import express from 'express';
import path from 'path';
import fs from 'fs';

const app = express();
const PORT = 3001;

// Default local path for Pebble swim data
const LOCAL_DATA_PATH = process.env.PEBBLE_DATA_PATH ||
  path.join('C:', 'Users', '100014430', 'Documents', 'GitHub', 'coffeng', 'pebble-swim-tracker', 'logs', 'PebbleSwimTracker');

// CORS for local dev
app.use((_req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  if (_req.method === 'OPTIONS') { res.sendStatus(200); return; }
  next();
});

// Parse JSON bodies
app.use(express.json());

// Health check / detection endpoint
app.get('/api/local/status', (_req, res) => {
  const exists = fs.existsSync(LOCAL_DATA_PATH);
  res.json({
    available: true,
    dataPath: LOCAL_DATA_PATH,
    pathExists: exists
  });
});

// List all JSON files in the local data directory
app.get('/api/local/files', (_req, res) => {
  try {
    if (!fs.existsSync(LOCAL_DATA_PATH)) {
      return res.status(404).json({ error: 'Data path not found', path: LOCAL_DATA_PATH });
    }

    const entries = fs.readdirSync(LOCAL_DATA_PATH, { withFileTypes: true });
    const jsonFiles = entries
      .filter(e => e.isFile() && e.name.endsWith('.json'))
      .map(e => {
        const filePath = path.join(LOCAL_DATA_PATH, e.name);
        const stats = fs.statSync(filePath);
        return {
          id: e.name, // use filename as ID for local files
          name: e.name,
          size: stats.size,
          modifiedTime: stats.mtime.toISOString()
        };
      })
      .sort((a, b) => b.size - a.size); // Sort by size descending (longest recording first)

    res.json({ files: jsonFiles });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// Download/read a specific JSON file by name
app.get('/api/local/files/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    // Prevent path traversal
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }

    const filePath = path.join(LOCAL_DATA_PATH, filename);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const json = JSON.parse(content);
    res.json(json);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// --- Annotations API ---
// Annotations are stored as sidecar files: swim-20260716-152708.annotations.json

function annotationPath(filename: string): string {
  const base = filename.replace(/\.json$/, '');
  return path.join(LOCAL_DATA_PATH, `${base}.annotations.json`);
}

// GET annotations for a file
app.get('/api/local/annotations/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }
    const annoPath = annotationPath(filename);
    if (!fs.existsSync(annoPath)) {
      return res.json({ turns: [] });
    }
    const content = fs.readFileSync(annoPath, 'utf-8');
    res.json(JSON.parse(content));
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// PUT (save) annotations for a file
app.put('/api/local/annotations/:filename', (req, res) => {
  try {
    const filename = req.params.filename;
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }
    const annoPath = annotationPath(filename);
    const data = req.body;
    fs.writeFileSync(annoPath, JSON.stringify(data, null, 2), 'utf-8');
    res.json({ ok: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`\n  Local file server running at http://localhost:${PORT}`);
  console.log(`  Serving files from: ${LOCAL_DATA_PATH}\n`);
});
