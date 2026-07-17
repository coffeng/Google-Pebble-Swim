/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DriveFile } from '../types';

const LOCAL_API_BASE = 'http://localhost:3001';

/**
 * Check if the local file server is available.
 * Returns true when running on desktop with the local server active.
 */
export async function isLocalServerAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${LOCAL_API_BASE}/api/local/status`, {
      signal: AbortSignal.timeout(1500)
    });
    if (!response.ok) return false;
    const data = await response.json();
    return data.available === true && data.pathExists === true;
  } catch {
    return false;
  }
}

/**
 * List JSON files from the local data directory.
 */
export async function listLocalFiles(): Promise<DriveFile[]> {
  const response = await fetch(`${LOCAL_API_BASE}/api/local/files`);
  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Failed to list local files: ${err}`);
  }
  const data = await response.json();
  const files: DriveFile[] = (data.files || []).map((f: any) => {
    const sizeBytes = typeof f.size === 'number' ? f.size : 0;
    let sizeString = '0 B';
    if (sizeBytes > 0) {
      if (sizeBytes < 1024) sizeString = `${sizeBytes} B`;
      else if (sizeBytes < 1024 * 1024) sizeString = `${(sizeBytes / 1024).toFixed(1)} KB`;
      else sizeString = `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return {
      id: f.id,
      name: f.name,
      size: sizeBytes,
      modifiedTime: f.modifiedTime,
      sizeString
    };
  });
  return files;
}

/**
 * Download and parse a local JSON file by filename (used as ID).
 */
export async function downloadLocalFile(filename: string): Promise<any> {
  const response = await fetch(`${LOCAL_API_BASE}/api/local/files/${encodeURIComponent(filename)}`);
  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Failed to download local file: ${err}`);
  }
  return response.json();
}


/**
 * Annotation data structure stored per file.
 */
export interface AnnotationData {
  turns: number[]; // Array of timeOffsetMs values where turns are annotated
}

/**
 * Load annotations for a file.
 */
export async function loadAnnotations(filename: string): Promise<AnnotationData> {
  try {
    const response = await fetch(`${LOCAL_API_BASE}/api/local/annotations/${encodeURIComponent(filename)}`);
    if (!response.ok) return { turns: [] };
    return await response.json();
  } catch {
    return { turns: [] };
  }
}

/**
 * Save annotations for a file.
 */
export async function saveAnnotations(filename: string, data: AnnotationData): Promise<void> {
  await fetch(`${LOCAL_API_BASE}/api/local/annotations/${encodeURIComponent(filename)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
}
