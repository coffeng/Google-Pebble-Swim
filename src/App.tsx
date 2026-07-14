/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Activity, 
  Compass, 
  FileJson, 
  Folder, 
  User, 
  LogOut, 
  RefreshCw, 
  Sliders, 
  Database, 
  AlertTriangle, 
  TrendingUp, 
  Clock, 
  Info,
  ChevronRight,
  ArrowUp,
  Download,
  Flame
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  AreaChart, 
  Area,
  ReferenceLine
} from 'recharts';
import { User as FirebaseUser } from 'firebase/auth';

import { DriveFile, DriveFolder, PebbleReading, ParseResult } from './types';
import { 
  initAuth, 
  googleSignIn, 
  logout, 
  searchPebbleFolders, 
  listPebbleJsonFiles, 
  downloadFileContent 
} from './lib/googleDrive';
import { parsePebbleJson, formatTimeOffset } from './lib/pebbleParser';

// --- Swim Stroke Classifier ported from pebble-swim-tracker ---
enum StrokeType {
  UNKNOWN = 0,
  FREESTYLE = 1,
  BREASTSTROKE = 2,
  BACKSTROKE = 3,
  BUTTERFLY = 4,
}

const BACKSTROKE_GZ = 250.0;
const BUTTERFLY_EZ_EY = 1.20;
const BREAST_MAX_FREQ = 0.475;
const FREESTYLE_MAX_FREQ = 0.52;

const strokeColors: Record<StrokeType, string> = {
  [StrokeType.UNKNOWN]: '#94a3b8', // slate-400
  [StrokeType.FREESTYLE]: '#ec4899', // pink-500
  [StrokeType.BREASTSTROKE]: '#f59e0b', // amber-500
  [StrokeType.BACKSTROKE]: '#10b981', // emerald-500
  [StrokeType.BUTTERFLY]: '#8b5cf6', // purple-500
};

const strokeNames: Record<StrokeType, string> = {
  [StrokeType.UNKNOWN]: 'Unknown',
  [StrokeType.FREESTYLE]: 'Freestyle',
  [StrokeType.BREASTSTROKE]: 'Breaststroke',
  [StrokeType.BACKSTROKE]: 'Backstroke',
  [StrokeType.BUTTERFLY]: 'Butterfly',
};

interface StrokeFeatures {
  n: number;
  gz_static: number;
  ey: number;
  ez: number;
  ez_ey: number;
  freq: number;
}

function mean_i16(v: number[]): number {
  if (v.length === 0) return 0.0;
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i];
  return sum / v.length;
}

function energy_i16(v: number[]): number {
  if (v.length === 0) return 0.0;
  const m = mean_i16(v);
  let acc = 0.0;
  for (let i = 0; i < v.length; i++) acc += Math.abs(v[i] - m);
  return acc / v.length;
}

function lowpass_mean_i16(v: number[], alpha: number): number {
  if (v.length === 0) return 0.0;
  let ema = v[0];
  let acc = ema;
  for (let i = 1; i < v.length; i++) {
    ema = alpha * v[i] + (1.0 - alpha) * ema;
    acc += ema;
  }
  return acc / v.length;
}

function dominant_frequency(sig: number[], fs: number): number {
  const n = sig.length;
  if (n < 4) return 0.0;
  let min_lag = Math.floor(fs / 2.0);
  if (min_lag < 1) min_lag = 1;
  let max_lag = Math.floor(fs / 0.3);
  if (max_lag > n - 1) max_lag = n - 1;
  
  let best_lag = 0;
  let best_val = 0.0;
  
  for (let lag = min_lag; lag <= max_lag; lag++) {
    let acc = 0.0;
    for (let i = 0; i < n - lag; i++) {
      acc += sig[i] * sig[i + lag];
    }
    if (acc > best_val) {
      best_val = acc;
      best_lag = lag;
    }
  }
  
  if (best_lag === 0) return 0.0;
  return fs / best_lag;
}

function stroke_extract_features(xs: number[], ys: number[], zs: number[], fs_hz: number): StrokeFeatures {
  const n = xs.length;
  const f: StrokeFeatures = { n, gz_static: 0, ey: 0, ez: 0, ez_ey: 0, freq: 0 };
  if (n < 8) return f;
  
  const limit = Math.min(n, 700);
  let mmag_acc = 0.0;
  const sig: number[] = new Array(limit);
  for (let i = 0; i < limit; i++) {
    const x = xs[i], y = ys[i], z = zs[i];
    const mag = Math.sqrt(x * x + y * y + z * z);
    sig[i] = mag;
    mmag_acc += mag;
  }
  const mmag = mmag_acc / limit;
  for (let i = 0; i < limit; i++) {
    sig[i] -= mmag;
  }
  
  const ys_sub = ys.slice(0, limit);
  const zs_sub = zs.slice(0, limit);
  
  f.gz_static = lowpass_mean_i16(zs_sub, 0.1);
  f.ey = energy_i16(ys_sub);
  f.ez = energy_i16(zs_sub);
  f.ez_ey = (f.ey > 1e-6) ? (f.ez / f.ey) : 0.0;
  f.freq = dominant_frequency(sig, fs_hz);
  return f;
}

function stroke_classify_features(f: StrokeFeatures, prev: StrokeType): StrokeType {
  if (f.n < 8) return StrokeType.UNKNOWN;
  if (f.gz_static < BACKSTROKE_GZ) return StrokeType.BACKSTROKE;
  if (f.ez_ey >= BUTTERFLY_EZ_EY) return StrokeType.BUTTERFLY;
  if (f.freq > 0.0 && f.freq <= BREAST_MAX_FREQ) return StrokeType.BREASTSTROKE;
  if (prev === StrokeType.FREESTYLE && f.freq <= FREESTYLE_MAX_FREQ) return prev;
  return StrokeType.FREESTYLE;
}

export default function App() {
  // Authentication states
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [needsAuth, setNeedsAuth] = useState<boolean>(true);
  const [isLoadingAuth, setIsLoadingAuth] = useState<boolean>(true);
  const [isLoggingIn, setIsLoggingIn] = useState<boolean>(false);

  // Google Drive states
  const [folders, setFolders] = useState<DriveFolder[]>([]);
  const [files, setFiles] = useState<DriveFile[]>([]);
  const [isLoadingDrive, setIsLoadingDrive] = useState<boolean>(false);
  const [driveError, setDriveError] = useState<string | null>(null);

  // Active File states
  const [selectedFile, setSelectedFile] = useState<DriveFile | null>(null);
  const [isLoadingFile, setIsLoadingFile] = useState<boolean>(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [rawJson, setRawJson] = useState<any>(null);
  const [parseResult, setParseResult] = useState<ParseResult | null>(null);

  // UI Control states
  const [timeWindow, setTimeWindow] = useState<'first-minute' | 'full'>('first-minute');
  const [hoverReading, setHoverReading] = useState<PebbleReading | null>(null);
  const [activeTab, setActiveTab] = useState<'charts' | 'stats' | 'raw'>('charts');

  // Sync auth state on load
  useEffect(() => {
    const unsubscribe = initAuth(
      (currentUser, accessToken) => {
        setUser(currentUser);
        setToken(accessToken);
        setNeedsAuth(false);
        setIsLoadingAuth(false);
      },
      () => {
        setUser(null);
        setToken(null);
        setNeedsAuth(true);
        setIsLoadingAuth(false);
      }
    );
    return () => unsubscribe();
  }, []);

  // Fetch folders and files once authenticated
  const loadDriveData = async (accessToken: string) => {
    setIsLoadingDrive(true);
    setDriveError(null);
    try {
      // 1. Search for folders containing "pebble"
      const detectedFolders = await searchPebbleFolders(accessToken);
      setFolders(detectedFolders);

      // 2. Search for JSON files (in folders or globally)
      const detectedFiles = await listPebbleJsonFiles(accessToken, detectedFolders);
      
      // Sort files by size descending so the "longest" file is first
      const sortedFiles = [...detectedFiles].sort((a, b) => {
        const sizeA = typeof a.size === 'number' ? a.size : 0;
        const sizeB = typeof b.size === 'number' ? b.size : 0;
        return sizeB - sizeA;
      });

      setFiles(sortedFiles);

      // 3. Automatically select and download the longest file (first in the list) if available
      if (sortedFiles.length > 0) {
        handleSelectFile(sortedFiles[0], accessToken);
      }
    } catch (err: any) {
      console.error('Error loading Drive directories:', err);
      setDriveError(err.message || 'Failed to retrieve files from Google Drive.');
    } finally {
      setIsLoadingDrive(false);
    }
  };

  // Re-fetch files manually
  const handleRefreshDrive = () => {
    if (token) {
      loadDriveData(token);
    }
  };

  // Trigger loading drive data once token changes
  useEffect(() => {
    if (token) {
      loadDriveData(token);
    }
  }, [token]);

  // Handle Sign-In click
  const handleLogin = async () => {
    setIsLoggingIn(true);
    try {
      const result = await googleSignIn();
      if (result) {
        setToken(result.accessToken);
        setUser(result.user);
        setNeedsAuth(false);
      }
    } catch (err) {
      console.error('Login process halted:', err);
    } finally {
      setIsLoggingIn(false);
    }
  };

  // Handle Logout
  const handleLogout = async () => {
    try {
      await logout();
      setUser(null);
      setToken(null);
      setFolders([]);
      setFiles([]);
      setSelectedFile(null);
      setParseResult(null);
      setRawJson(null);
      setNeedsAuth(true);
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  // Fetch and parse file content
  const handleSelectFile = async (file: DriveFile, currentToken?: string) => {
    const activeToken = currentToken || token;
    if (!activeToken) return;

    setSelectedFile(file);
    setIsLoadingFile(true);
    setFileError(null);
    setHoverReading(null);
    
    try {
      const data = await downloadFileContent(activeToken, file.id);
      setRawJson(data);
      const parsed = parsePebbleJson(data);
      setParseResult(parsed);
      
      // Default to first-minute view
      setTimeWindow('first-minute');
    } catch (err: any) {
      console.error('Error downloading/parsing file:', err);
      setFileError(err.message || 'Could not download or parse this JSON file. Ensure it is valid sensor data.');
      setParseResult(null);
      setRawJson(null);
    } finally {
      setIsLoadingFile(false);
    }
  };

  // Process data for graphing based on active window and downsampling
  const graphData = useMemo(() => {
    if (!parseResult) return [];

    let filtered = [...parseResult.readings];

    // Filter to first minute (60,000 ms)
    if (timeWindow === 'first-minute') {
      filtered = filtered.filter(r => r.timeOffset <= 60000);
    }

    // Downsample full dataset if it is too massive to prevent browser/Recharts SVG rendering lag
    // Recharts handles up to 1000 points extremely smoothly
    if (filtered.length > 1000) {
      const maxPoints = 1000;
      const step = Math.ceil(filtered.length / maxPoints);
      const downsampled: PebbleReading[] = [];
      for (let i = 0; i < filtered.length; i += step) {
        downsampled.push(filtered[i]);
      }
      return downsampled;
    }

    return filtered;
  }, [parseResult, timeWindow]);

  // Pre-calculate statistics for the current filtered data window
  const computedStats = useMemo(() => {
    if (!graphData || graphData.length === 0) return null;

    let minX = Infinity, maxX = -Infinity, sumX = 0;
    let minY = Infinity, maxY = -Infinity, sumY = 0;
    let minZ = Infinity, maxZ = -Infinity, sumZ = 0;
    let minMag = Infinity, maxMag = -Infinity, sumMag = 0;
    let compassSum = 0, compassCount = 0;
    let headings: number[] = [];

    graphData.forEach(r => {
      if (r.x < minX) minX = r.x;
      if (r.x > maxX) maxX = r.x;
      sumX += r.x;

      if (r.y < minY) minY = r.y;
      if (r.y > maxY) maxY = r.y;
      sumY += r.y;

      if (r.z < minZ) minZ = r.z;
      if (r.z > maxZ) maxZ = r.z;
      sumZ += r.z;

      if (r.magnitude < minMag) minMag = r.magnitude;
      if (r.magnitude > maxMag) maxMag = r.magnitude;
      sumMag += r.magnitude;

      if (r.compass !== undefined) {
        compassSum += r.compass;
        compassCount++;
        headings.push(r.compass);
      }
    });

    const len = graphData.length;
    return {
      accel: {
        x: { min: minX, max: maxX, avg: Math.round(sumX / len) },
        y: { min: minY, max: maxY, avg: Math.round(sumY / len) },
        z: { min: minZ, max: maxZ, avg: Math.round(sumZ / len) },
        mag: { min: minMag, max: maxMag, avg: Math.round(sumMag / len) }
      },
      compass: compassCount > 0 ? {
        min: Math.min(...headings),
        max: Math.max(...headings),
        avg: Math.round(compassSum / compassCount),
        count: compassCount
      } : null
    };
  }, [graphData]);

  // Algorithm to detect individual swim strokes across the full length of telemetry
  const detectedStrokes = useMemo(() => {
    if (!parseResult || parseResult.readings.length === 0) return [];

    const fullData = parseResult.readings;

    // 1. Apply a moving average filter to smooth out high-frequency noise/jitter
    const smoothed: { reading: PebbleReading; smoothMag: number }[] = [];
    const windowRadius = 4; // Total window = 9 samples
    
    for (let i = 0; i < fullData.length; i++) {
      let sum = 0;
      let count = 0;
      for (let j = Math.max(0, i - windowRadius); j <= Math.min(fullData.length - 1, i + windowRadius); j++) {
        sum += fullData[j].magnitude;
        count++;
      }
      smoothed.push({
        reading: fullData[i],
        smoothMag: sum / count
      });
    }

    // 2. Compute dynamic thresholds based on the smoothed kinetic energy (magnitude)
    const smoothMags = smoothed.map(s => s.smoothMag);
    const avgMag = smoothMags.reduce((a, b) => a + b, 0) / smoothMags.length;
    const maxMag = Math.max(...smoothMags);
    
    // Swimming strokes involve peak power. A peak should be clearly above standard gravity (1000mG)
    // and also above the average level of the smoothed data.
    const threshold = Math.max(1120, avgMag + (maxMag - avgMag) * 0.22);

    // 3. Peak detection with temporal distance constraints
    // Swimmers complete strokes separated by at least 700ms to 2.5s.
    const strokes: PebbleReading[] = [];
    let lastStrokeTime = -Infinity;

    for (let i = 1; i < smoothed.length - 1; i++) {
      const prev = smoothed[i - 1].smoothMag;
      const curr = smoothed[i].smoothMag;
      const next = smoothed[i + 1].smoothMag;

      // Local maximum check
      if (curr > prev && curr > next && curr > threshold) {
        const timeOffset = smoothed[i].reading.timeOffset;
        // Require at least 700ms separation between successive strokes
        if (timeOffset - lastStrokeTime >= 700) {
          strokes.push(smoothed[i].reading);
          lastStrokeTime = timeOffset;
        }
      }
    }

    return strokes;
  }, [parseResult]);

  // Algorithm to detect swim turn events across the entire session
  const detectedTurns = useMemo(() => {
    if (!parseResult || parseResult.readings.length === 0) return [];

    const readings = parseResult.readings;
    const turns: PebbleReading[] = [];

    if (parseResult.hasCompass) {
      // --- Compass-based Turn Detection (Highly precise) ---
      // We will compute rolling average direction before and after each point,
      // and look for local maxima of angular difference near 180 degrees.
      const diffs: { reading: PebbleReading; diff: number }[] = [];
      let beforeStartIdx = 0;
      let beforeEndIdx = 0;
      let afterStartIdx = 0;
      let afterEndIdx = 0;

      for (let i = 0; i < readings.length; i++) {
        const currTime = readings[i].timeOffset;

        // Advance beforeStartIdx to first index >= currTime - 14000
        while (beforeStartIdx < readings.length && readings[beforeStartIdx].timeOffset < currTime - 14000) {
          beforeStartIdx++;
        }
        // Advance beforeEndIdx to first index > currTime - 3000
        while (beforeEndIdx < readings.length && readings[beforeEndIdx].timeOffset <= currTime - 3000) {
          beforeEndIdx++;
        }

        // Advance afterStartIdx to first index >= currTime + 3000
        while (afterStartIdx < readings.length && readings[afterStartIdx].timeOffset < currTime + 3000) {
          afterStartIdx++;
        }
        // Advance afterEndIdx to first index > currTime + 14000
        while (afterEndIdx < readings.length && readings[afterEndIdx].timeOffset <= currTime + 14000) {
          afterEndIdx++;
        }

        const beforeCount = beforeEndIdx - beforeStartIdx;
        const afterCount = afterEndIdx - afterStartIdx;

        if (beforeCount >= 8 && afterCount >= 8) {
          // Circular average for before window
          let sumCosB = 0;
          let sumSinB = 0;
          let countB = 0;
          for (let k = beforeStartIdx; k < beforeEndIdx; k++) {
            const comp = readings[k].compass;
            if (comp !== undefined) {
              const rad = (comp * Math.PI) / 180;
              sumCosB += Math.cos(rad);
              sumSinB += Math.sin(rad);
              countB++;
            }
          }

          // Circular average for after window
          let sumCosA = 0;
          let sumSinA = 0;
          let countA = 0;
          for (let k = afterStartIdx; k < afterEndIdx; k++) {
            const comp = readings[k].compass;
            if (comp !== undefined) {
              const rad = (comp * Math.PI) / 180;
              sumCosA += Math.cos(rad);
              sumSinA += Math.sin(rad);
              countA++;
            }
          }

          if (countB > 0 && countA > 0) {
            const avgRadB = Math.atan2(sumSinB / countB, sumCosB / countB);
            let avgDegB = (avgRadB * 180) / Math.PI;
            if (avgDegB < 0) avgDegB += 360;

            const avgRadA = Math.atan2(sumSinA / countA, sumCosA / countA);
            let avgDegA = (avgRadA * 180) / Math.PI;
            if (avgDegA < 0) avgDegA += 360;

            const angularDiff = 180 - Math.abs(180 - Math.abs(avgDegB - avgDegA));
            diffs.push({ reading: readings[i], diff: angularDiff });
          }
        }
      }

      // Find local maxima in diffs
      let lastTurnTime = -Infinity;
      const minTurnInterval = 35000; // Minimum 35s between turns (covers standard pool lap times)

      for (let j = 1; j < diffs.length - 1; j++) {
        const prev = diffs[j - 1].diff;
        const curr = diffs[j].diff;
        const next = diffs[j + 1].diff;

        if (curr > prev && curr > next && curr > 135) {
          const time = diffs[j].reading.timeOffset;
          if (time - lastTurnTime >= minTurnInterval) {
            turns.push(diffs[j].reading);
            lastTurnTime = time;
          }
        }
      }
    } else {
      // --- Fallback: Accelerometer-based Turn Detection ---
      // We look for quiet streamline glide periods with very low standard deviation in magnitude
      const stdDevs: { reading: PebbleReading; stdDev: number }[] = [];
      const halfWin = 15; // 3 seconds window around point (at 10Hz)

      for (let i = halfWin; i < readings.length - halfWin; i++) {
        const windowReadings = readings.slice(i - halfWin, i + halfWin);
        if (windowReadings.length > 0) {
          const mags = windowReadings.map(r => r.magnitude);
          const mean = mags.reduce((a, b) => a + b, 0) / mags.length;
          const variance = mags.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / mags.length;
          const stdDev = Math.sqrt(variance);
          stdDevs.push({ reading: readings[i], stdDev });
        }
      }

      let lastTurnTime = -Infinity;
      const minTurnInterval = 35000;

      for (let j = 1; j < stdDevs.length - 1; j++) {
        const prev = stdDevs[j - 1].stdDev;
        const curr = stdDevs[j].stdDev;
        const next = stdDevs[j + 1].stdDev;

        if (curr < prev && curr < next && curr < 115) {
          const time = stdDevs[j].reading.timeOffset;
          if (time > 15000 && time < readings[readings.length - 1].timeOffset - 15000) {
            if (time - lastTurnTime >= minTurnInterval) {
              turns.push(stdDevs[j].reading);
              lastTurnTime = time;
            }
          }
        }
      }
    }

    return turns;
  }, [parseResult]);

  // Find the closest point in graphData for each stroke to ensure Recharts can render the ReferenceLines
  const strokeMarkers = useMemo(() => {
    if (detectedStrokes.length === 0 || graphData.length === 0) return [];
    
    const minTime = graphData[0].timeOffset;
    const maxTime = graphData[graphData.length - 1].timeOffset;

    const markers: { reading: PebbleReading; originalIndex: number }[] = [];

    detectedStrokes.forEach((stroke, idx) => {
      if (stroke.timeOffset >= minTime && stroke.timeOffset <= maxTime) {
        let closest = graphData[0];
        let minDiff = Math.abs(graphData[0].timeOffset - stroke.timeOffset);
        
        for (let i = 1; i < graphData.length; i++) {
          const diff = Math.abs(graphData[i].timeOffset - stroke.timeOffset);
          if (diff < minDiff) {
            minDiff = diff;
            closest = graphData[i];
          }
        }
        markers.push({
          reading: closest,
          originalIndex: idx
        });
      }
    });

    return markers;
  }, [detectedStrokes, graphData]);

  // Find the closest point in graphData for each turn to ensure Recharts can render the ReferenceLines
  const turnMarkers = useMemo(() => {
    if (detectedTurns.length === 0 || graphData.length === 0) return [];
    
    const minTime = graphData[0].timeOffset;
    const maxTime = graphData[graphData.length - 1].timeOffset;

    const markers: { reading: PebbleReading; originalIndex: number }[] = [];

    detectedTurns.forEach((turn, idx) => {
      if (turn.timeOffset >= minTime && turn.timeOffset <= maxTime) {
        let closest = graphData[0];
        let minDiff = Math.abs(graphData[0].timeOffset - turn.timeOffset);
        
        for (let i = 1; i < graphData.length; i++) {
          const diff = Math.abs(graphData[i].timeOffset - turn.timeOffset);
          if (diff < minDiff) {
            minDiff = diff;
            closest = graphData[i];
          }
        }
        markers.push({
          reading: closest,
          originalIndex: idx
        });
      }
    });

    return markers;
  }, [detectedTurns, graphData]);

  // Classify strokes for each lap based on the C classifier logic
  const lapStrokeTypes = useMemo(() => {
    if (!parseResult || parseResult.readings.length === 0) return [];
    const readings = parseResult.readings;
    const turns = detectedTurns;
    const fs_hz = parseResult.sampleRateHz ?? 25;

    const laps: { startIndex: number; endIndex: number; strokeType: StrokeType }[] = [];
    let prevStroke = StrokeType.FREESTYLE;

    for (let i = 0; i <= turns.length; i++) {
      const startIdx = i === 0 ? 0 : readings.findIndex(r => r.timeOffset === turns[i-1].timeOffset);
      const endIdx = i === turns.length ? readings.length - 1 : readings.findIndex(r => r.timeOffset === turns[i].timeOffset);
      
      if (startIdx === -1 || endIdx === -1 || startIdx >= endIdx) {
        laps.push({ startIndex: Math.max(0, startIdx), endIndex: Math.max(0, endIdx), strokeType: StrokeType.FREESTYLE });
        continue;
      }

      const lapReadings = readings.slice(startIdx, endIdx + 1);
      const xs = lapReadings.map(r => r.x);
      const ys = lapReadings.map(r => r.y);
      const zs = lapReadings.map(r => r.z);

      const f = stroke_extract_features(xs, ys, zs, fs_hz);
      const type = stroke_classify_features(f, prevStroke);
      laps.push({ startIndex: startIdx, endIndex: endIdx, strokeType: type });
      prevStroke = type;
    }

    return laps;
  }, [parseResult, detectedTurns]);

  // Map each stroke marker to its classified StrokeType based on which lap it falls into
  const strokeMarkersWithTypes = useMemo(() => {
    if (strokeMarkers.length === 0) return [];
    return strokeMarkers.map(marker => {
      const idx = marker.reading.index;
      let strokeType = StrokeType.UNKNOWN;
      for (const lap of lapStrokeTypes) {
        if (idx >= lap.startIndex && idx <= lap.endIndex) {
          strokeType = lap.strokeType;
          break;
        }
      }
      if (strokeType === StrokeType.UNKNOWN) strokeType = StrokeType.FREESTYLE;
      return {
        ...marker,
        strokeType,
      };
    });
  }, [strokeMarkers, lapStrokeTypes]);

  // Compute stroke type distribution counts
  const strokeCounts = useMemo(() => {
    const counts: Record<StrokeType, number> = {
      [StrokeType.UNKNOWN]: 0,
      [StrokeType.FREESTYLE]: 0,
      [StrokeType.BREASTSTROKE]: 0,
      [StrokeType.BACKSTROKE]: 0,
      [StrokeType.BUTTERFLY]: 0,
    };
    strokeMarkersWithTypes.forEach(marker => {
      counts[marker.strokeType]++;
    });
    return counts;
  }, [strokeMarkersWithTypes]);

  // Set the first reading as the default hover state so elements are pre-rendered beautifully
  useEffect(() => {
    if (graphData && graphData.length > 0) {
      setHoverReading(graphData[0]);
    } else {
      setHoverReading(null);
    }
  }, [graphData]);

  // 3D Tilt calculation (Pitch/Roll) for the device model visualizer
  const deviceOrientation = useMemo(() => {
    if (!hoverReading) return { pitch: 0, roll: 0 };
    const { x, y, z } = hoverReading;
    // Calculate pitch (forward/backward tilt) and roll (sideways tilt) in degrees
    // We assume the accelerometer values are relative to Earth's gravity
    const pitch = Math.atan2(-x, Math.sqrt(y * y + z * z)) * (180 / Math.PI);
    const roll = Math.atan2(y, z) * (180 / Math.PI);
    return { 
      pitch: parseFloat(pitch.toFixed(1)), 
      roll: parseFloat(roll.toFixed(1)) 
    };
  }, [hoverReading]);

  // Helper for formatting file size in KB
  const totalDurationString = useMemo(() => {
    if (!parseResult) return '0s';
    const totalSecs = Math.round(parseResult.totalDurationMs / 1000);
    const mins = Math.floor(totalSecs / 60);
    const secs = totalSecs % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  }, [parseResult]);

  return (
    <div id="app_root" className="min-h-screen bg-slate-950 text-slate-200 font-sans flex flex-col antialiased selection:bg-blue-600 selection:text-white">
      {/* Upper Navigation Header */}
      <header id="header_nav" className="h-14 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900 shrink-0">
        <div className="flex items-center space-x-4">
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center">
            <Activity className="w-5 h-5 text-white animate-pulse" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight uppercase text-slate-100">Pebble Data Explorer</h1>
            <p className="text-[10px] text-slate-500 font-mono uppercase truncate max-w-[200px] sm:max-w-xs md:max-w-md">
              {selectedFile ? `File: /drive/pebble/${selectedFile.name}` : 'No active file'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-6">
          {parseResult && (
            <>
              <div className="text-right hidden sm:block">
                <span className="block text-[10px] text-slate-500 uppercase">Sampling Rate</span>
                <span className="text-xs font-mono">{parseResult.sampleRateHz ? `${parseResult.sampleRateHz} Hz` : 'N/A'}</span>
              </div>
              <div className="text-right hidden sm:block">
                <span className="block text-[10px] text-slate-500 uppercase">Duration</span>
                <span className="text-xs font-mono">{timeWindow === 'first-minute' ? '00:01:00' : totalDurationString}</span>
              </div>
            </>
          )}

          {user ? (
            <div className="flex items-center gap-2.5 bg-slate-950/80 pl-2.5 pr-1 py-1 rounded-full border border-slate-800">
              <div className="text-right hidden md:block">
                <p className="text-[10px] font-semibold text-slate-200 leading-none">{user.displayName || 'User'}</p>
              </div>
              {user.photoURL ? (
                <img src={user.photoURL} alt={user.displayName || 'Avatar'} className="h-6 w-6 rounded-full border border-slate-700" referrerPolicy="no-referrer" />
              ) : (
                <div className="h-6 w-6 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700">
                  <User className="h-3 w-3 text-slate-400" />
                </div>
              )}
              <button
                id="btn_logout"
                onClick={handleLogout}
                title="Sign Out of Google Account"
                className="p-1 hover:bg-red-500/10 text-slate-400 hover:text-red-400 rounded-full transition-colors"
              >
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : null}
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 p-4 lg:p-6 flex flex-col gap-6 max-w-7xl w-full mx-auto">
        
        {isLoadingAuth ? (
          <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4">
            <div className="h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-slate-400 text-xs font-mono">Initializing connection profile...</p>
          </div>
        ) : needsAuth ? (
          /* Authentication Screen */
          <motion.div 
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex-1 max-w-xl mx-auto w-full flex flex-col justify-center py-12 px-4"
          >
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 shadow-2xl text-center flex flex-col gap-6 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-blue-600"></div>
              
              <div className="mx-auto p-3 bg-blue-600/10 text-blue-400 rounded border border-blue-500/20 w-fit">
                <Database className="h-8 w-8" />
              </div>

              <div>
                <h2 className="text-xs font-bold tracking-wider uppercase text-slate-100">Connect Google Drive</h2>
                <p className="text-slate-400 text-xs mt-2 leading-relaxed">
                  To analyze your Pebble smartwatch sensor readings, this application needs secure read-only permission to retrieve files from your Google Drive folder.
                </p>
              </div>

              <div className="bg-slate-950 border border-slate-800 rounded p-4 text-left flex flex-col gap-3">
                <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Expected Folder Layout</h4>
                <div className="flex gap-3 items-start text-xs text-slate-400">
                  <Folder className="h-4 w-4 text-blue-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-mono text-blue-300">/pebble/</span> folder containing one or multiple <span className="font-mono text-blue-300">.json</span> logs.
                  </div>
                </div>
                <div className="flex gap-3 items-start text-xs text-slate-400">
                  <FileJson className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    Files should contain arrays or structures mapping <span className="font-mono text-rose-300">x, y, z</span> accelerometer vectors and compass heading data.
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-3 items-center mt-2">
                <button
                  id="btn_gsi_signin"
                  onClick={handleLogin}
                  disabled={isLoggingIn}
                  className="gsi-material-button w-full shadow-lg"
                >
                  <div className="gsi-material-button-state"></div>
                  <div className="gsi-material-button-content-wrapper">
                    <div className="gsi-material-button-icon">
                      <svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" style={{ display: 'block' }}>
                        <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"></path>
                        <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"></path>
                        <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"></path>
                        <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"></path>
                        <path fill="none" d="M0 0h48v48H0z"></path>
                      </svg>
                    </div>
                    <span className="gsi-material-button-contents">
                      {isLoggingIn ? 'Authenticating Access...' : 'Connect to Google Drive'}
                    </span>
                  </div>
                </button>
                <p className="text-[10px] text-slate-500 font-mono mt-1">Authorized via Google Identity Services and Firebase</p>
              </div>

            </div>
          </motion.div>
        ) : (
          /* Main Application Grid */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start w-full">
            
            {/* Sidebar Column: File Browser (Takes 3 cols out of 12 on lg) */}
            <section id="sidebar_file_browser" className="lg:col-span-3 flex flex-col gap-4">
              
              {/* Session Overview */}
              <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Session Overview</h3>
                  <button
                    onClick={handleRefreshDrive}
                    disabled={isLoadingDrive}
                    className="p-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 rounded transition-all disabled:opacity-50"
                    title="Refresh file listings"
                  >
                    <RefreshCw className={`h-3 w-3 ${isLoadingDrive ? 'animate-spin' : ''}`} />
                  </button>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-end border-b border-slate-800 pb-2">
                    <span className="text-xs text-slate-400 font-light italic">Folders Found</span>
                    <span className="text-sm font-mono text-blue-400">{folders.length}</span>
                  </div>
                  <div className="flex justify-between items-end border-b border-slate-800 pb-2">
                    <span className="text-xs text-slate-400 font-light italic">Logs Detected</span>
                    <span className="text-sm font-mono text-rose-400">{files.length}</span>
                  </div>
                  {folders.length > 0 && (
                    <div className="text-[9px] text-slate-500 font-mono flex flex-wrap gap-1">
                      <span className="text-slate-400 font-medium">Folders:</span>
                      {folders.map(f => f.name).join(', ')}
                    </div>
                  )}
                </div>
              </div>

              {/* File Selection Box */}
              <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-3 min-h-[300px]">
                <div className="flex items-center space-x-2 border-b border-slate-800 pb-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]"></div>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">JSON Logs List</span>
                </div>

                {isLoadingDrive ? (
                  <div className="flex-1 flex flex-col items-center justify-center py-10 gap-2">
                    <div className="h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-slate-500 text-[10px] font-mono">Scanning directories...</p>
                  </div>
                ) : driveError ? (
                  <div className="flex-1 flex flex-col items-center justify-center p-2 text-center gap-2">
                    <AlertTriangle className="h-6 w-6 text-amber-500 opacity-80" />
                    <p className="text-slate-300 text-xs font-medium">Access Error</p>
                    <p className="text-slate-500 text-[9px] leading-relaxed max-w-[180px] font-mono">{driveError}</p>
                    <button 
                      onClick={handleRefreshDrive}
                      className="mt-1 text-[10px] text-blue-400 hover:text-blue-300 underline font-mono uppercase"
                    >
                      Retry
                    </button>
                  </div>
                ) : files.length === 0 ? (
                  <div className="flex-1 flex flex-col items-center justify-center p-3 text-center gap-2 border border-dashed border-slate-800 rounded bg-slate-950/10">
                    <div className="p-2 bg-slate-950 rounded text-slate-500 border border-slate-800">
                      <Folder className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">No Files Found</p>
                      <p className="text-[9px] text-slate-500 mt-1 leading-normal max-w-[180px] mx-auto">
                        We searched for folders named <span className="font-mono text-blue-400">"pebble"</span> but couldn't detect any <span className="font-mono text-rose-400">.json</span> files.
                      </p>
                    </div>
                  </div>
                ) : (
                  /* File List */
                  <div className="space-y-1.5 overflow-y-auto max-h-[360px] pr-1">
                    {files.map((file, idx) => {
                      const isSelected = selectedFile?.id === file.id;
                      const isLongest = idx === 0; // First index is longest due to sorted order
                      return (
                        <button
                          key={file.id}
                          id={`file_item_${file.id}`}
                          onClick={() => handleSelectFile(file)}
                          className={`w-full text-left p-2 rounded border text-xs transition-all relative overflow-hidden flex items-start justify-between gap-2 ${
                            isSelected 
                              ? 'bg-blue-600/10 border-blue-500 hover:bg-blue-600/15' 
                              : 'bg-slate-850/50 border-slate-800/80 hover:bg-slate-800 hover:border-slate-700'
                          }`}
                        >
                          {/* Selected Active glow line */}
                          {isSelected && (
                            <div className="absolute top-0 left-0 w-0.5 h-full bg-blue-500"></div>
                          )}

                          <div className="space-y-0.5 min-w-0 flex-1">
                            <div className="font-medium text-slate-200 truncate pr-2 font-mono text-[10px]" title={file.name}>
                              {file.name}
                            </div>
                            <div className="flex items-center gap-2 text-[9px] text-slate-500 font-mono">
                              <span className="text-slate-400 font-semibold">
                                {file.sizeString}
                              </span>
                              {file.modifiedTime && (
                                <span className="truncate">
                                  {new Date(file.modifiedTime).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Longest badge / selection indicator */}
                          <div className="flex flex-col items-end gap-1 shrink-0">
                            {isLongest ? (
                              <span className="px-1 py-0.2 bg-amber-500/10 border border-amber-500/20 text-[8px] text-amber-400 rounded font-bold font-mono uppercase flex items-center gap-0.5 shadow-sm">
                                <Flame className="h-2 w-2" /> Longest
                              </span>
                            ) : null}
                            {isSelected && (
                              <span className="h-1 w-1 bg-blue-400 rounded-full animate-ping"></span>
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Data Specifications Guide Card */}
              <div className="bg-slate-900/30 border border-slate-800/60 rounded p-4 text-xs text-slate-400 flex flex-col gap-2">
                <div className="flex items-center gap-1.5 text-slate-400 font-bold uppercase tracking-widest text-[9px]">
                  <Info className="h-3 w-3 text-blue-400" />
                  <span>How "Longest" is Computed</span>
                </div>
                <p className="leading-relaxed text-[11px]">
                  The files listed are arranged by <strong>byte size descending</strong>. Pebble sensor logs with high sample counts and multiple telemetry axes translate to larger file footprints, identifying the first record as the longest log.
                </p>
              </div>

            </section>

            {/* Dashboard Visualizer Panel (Takes 8 cols out of 12 on lg) */}
            <section id="dashboard_view" className="lg:col-span-8 flex flex-col gap-6">
              
              {/* File Loading overlay */}
              <AnimatePresence mode="wait">
                {isLoadingFile ? (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="bg-slate-900/50 border border-slate-800 rounded-lg p-16 flex flex-col items-center justify-center gap-4 text-center min-h-[450px]"
                  >
                    <div className="relative">
                      <div className="h-10 w-10 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
                      <Download className="h-4 w-4 text-blue-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-bounce" />
                    </div>
                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-widest text-white">Downloading Pebble Log</h4>
                      <p className="text-[11px] text-slate-500 mt-1 font-mono">Fetching raw sensor JSON array from Google Drive...</p>
                    </div>
                  </motion.div>
                ) : fileError ? (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="bg-slate-900/50 border border-slate-800 rounded-lg p-12 flex flex-col items-center justify-center gap-3 text-center min-h-[450px]"
                  >
                    <AlertTriangle className="h-10 w-10 text-rose-500 animate-bounce" />
                    <h4 className="text-xs font-bold uppercase tracking-widest text-white">Parsing Failure</h4>
                    <p className="text-[11px] text-slate-400 leading-relaxed max-w-md font-mono">{fileError}</p>
                    <button
                      onClick={() => selectedFile && handleSelectFile(selectedFile)}
                      className="mt-4 px-3 py-1.5 bg-slate-800 border border-slate-700 hover:bg-slate-700 rounded text-[10px] font-bold uppercase tracking-wider text-slate-200 transition-all"
                    >
                      Reload File
                    </button>
                  </motion.div>
                ) : parseResult ? (
                  /* Live Interactive File Dashboard */
                  <motion.div
                    key={selectedFile?.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-4"
                  >
                    
                    {/* Active File Title and Quick Stats Strip */}
                    <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-4">
                      
                      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                        <div className="space-y-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="h-1.5 w-1.5 bg-emerald-500 rounded-full animate-pulse shrink-0"></span>
                            <span className="text-[9px] font-bold tracking-widest text-emerald-400 font-mono uppercase">Telemetry Loaded</span>
                          </div>
                          <h2 className="text-sm font-bold text-slate-100 truncate font-mono" title={selectedFile?.name}>
                            {selectedFile?.name}
                          </h2>
                        </div>

                        {/* Chart range Toggle (First Minute vs Full) */}
                        <div className="flex bg-slate-950 p-1 border border-slate-800 rounded self-start md:self-auto">
                          <button
                            id="btn_view_first_minute"
                            onClick={() => setTimeWindow('first-minute')}
                            className={`px-3 py-1 rounded text-[10px] font-bold uppercase tracking-wider transition-all flex items-center gap-1.5 ${
                              timeWindow === 'first-minute'
                                ? 'bg-blue-600 text-white shadow-md'
                                : 'text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            <Clock className="h-3 w-3" /> First Minute
                          </button>
                          <button
                            id="btn_view_full"
                            onClick={() => setTimeWindow('full')}
                            className={`px-3 py-1 rounded text-[10px] font-bold uppercase tracking-wider transition-all flex items-center gap-1.5 ${
                              timeWindow === 'full'
                                ? 'bg-blue-600 text-white shadow-md'
                                : 'text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            <Sliders className="h-3 w-3" /> Full Session ({totalDurationString})
                          </button>
                        </div>
                      </div>

                      {/* Info grid */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        <div className="bg-slate-950 p-3 rounded border border-slate-850">
                          <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono">Sample Count</div>
                          <div className="text-sm font-bold text-blue-400 mt-1 font-mono">
                            {parseResult.readings.length.toLocaleString()}
                          </div>
                        </div>

                        <div className="bg-slate-950 p-3 rounded border border-slate-850">
                          <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono">Sample Rate</div>
                          <div className="text-sm font-bold text-rose-400 mt-1 font-mono">
                            {parseResult.sampleRateHz ? `${parseResult.sampleRateHz} Hz` : 'N/A'}
                          </div>
                        </div>

                        <div className="bg-slate-950 p-3 rounded border border-slate-850">
                          <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono">Sensors</div>
                          <div className="text-xs font-bold text-slate-300 mt-1 flex flex-wrap gap-1">
                            {parseResult.hasAccelerometer && (
                              <span className="px-1 bg-blue-500/10 border border-blue-500/20 text-blue-400 rounded text-[8px] font-mono uppercase">XYZ Accel</span>
                            )}
                            {parseResult.hasCompass && (
                              <span className="px-1 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded text-[8px] font-mono uppercase">Compass</span>
                            )}
                          </div>
                        </div>

                        <div className="bg-slate-950 p-3 rounded border border-slate-850">
                          <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono">Total Duration</div>
                          <div className="text-sm font-bold text-amber-500 mt-1 font-mono">
                            {totalDurationString}
                          </div>
                        </div>
                      </div>

                    </div>

                    {/* Navigation Tab Bar */}
                    <div className="flex border-b border-slate-800 font-mono text-[10px] uppercase tracking-wider">
                      <button
                        onClick={() => setActiveTab('charts')}
                        className={`px-4 py-2 border-b transition-all ${
                          activeTab === 'charts'
                            ? 'border-blue-500 text-blue-400 font-bold'
                            : 'border-transparent text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Visual Charts
                      </button>
                      <button
                        onClick={() => setActiveTab('stats')}
                        className={`px-4 py-2 border-b transition-all ${
                          activeTab === 'stats'
                            ? 'border-blue-500 text-blue-400 font-bold'
                            : 'border-transparent text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Sensor Stats
                      </button>
                      <button
                        onClick={() => setActiveTab('raw')}
                        className={`px-4 py-2 border-b transition-all ${
                          activeTab === 'raw'
                            ? 'border-blue-500 text-blue-400 font-bold'
                            : 'border-transparent text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Raw JSON Browser
                      </button>
                    </div>

                    {/* Tab 1: Interactive Charts */}
                    {activeTab === 'charts' && (
                      <div className="space-y-4">
                        
                        {/* Swim Stroke Rate Metric Dashboard if strokes are detected */}
                        {detectedStrokes.length > 0 && (
                          <div className="bg-slate-900/45 border border-slate-800/80 rounded-lg p-4 flex flex-col gap-3">
                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                              <div className="flex items-center gap-3">
                                <div className="p-2 bg-pink-500/10 text-pink-400 rounded-lg border border-pink-500/20">
                                  <Activity className="h-5 w-5 animate-pulse" />
                                </div>
                                <div>
                                  <h4 className="text-xs font-bold uppercase tracking-widest text-slate-100">🏊 Swim Telemetry Analyzer</h4>
                                  <p className="text-[10px] text-slate-400 mt-0.5">
                                    Full-length session analysis identifies individual strokes and turning points across all laps.
                                  </p>
                                </div>
                              </div>

                              <div className="flex flex-wrap gap-2">
                                <div className="bg-slate-950 px-3 py-1.5 rounded border border-slate-800/60 font-mono text-center min-w-[110px]">
                                  <span className="block text-[8px] text-slate-500 uppercase tracking-wider">Strokes Detected</span>
                                  <span className="text-sm font-bold text-pink-400">{detectedStrokes.length}</span>
                                </div>
                                <div className="bg-slate-950 px-3 py-1.5 rounded border border-slate-800/60 font-mono text-center min-w-[110px]">
                                  <span className="block text-[8px] text-slate-500 uppercase tracking-wider">Avg Stroke Rate</span>
                                  <span className="text-sm font-bold text-pink-400">
                                    {(() => {
                                      const maxOffset = Math.max(...parseResult.readings.map(r => r.timeOffset), 0);
                                      const sessionSecs = maxOffset / 1000;
                                      return sessionSecs > 0 ? Math.round((detectedStrokes.length / sessionSecs) * 60) : 0;
                                    })()} <span className="text-[9px] text-slate-400">SPM</span>
                                  </span>
                                </div>
                                <div className="bg-slate-950 px-3 py-1.5 rounded border border-slate-800/60 font-mono text-center min-w-[110px]">
                                  <span className="block text-[8px] text-slate-500 uppercase tracking-wider">Laps Covered</span>
                                  <span className="text-sm font-bold text-orange-400">
                                    {detectedTurns.length + 1} <span className="text-[9px] text-slate-500 font-normal">({detectedTurns.length} turns)</span>
                                  </span>
                                </div>
                                <div className="bg-slate-950 px-3 py-1.5 rounded border border-slate-800/60 font-mono text-center min-w-[110px]">
                                  <span className="block text-[8px] text-slate-500 uppercase tracking-wider">Analysis Phase</span>
                                  <span className="text-xs font-bold text-blue-400 flex items-center justify-center gap-1 mt-0.5">
                                    <Clock className="h-3 w-3" /> {timeWindow === 'first-minute' ? 'First 60s' : 'Full Session'}
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* Stroke Type Distribution Legend */}
                            <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-slate-800/60 w-full text-[10px] font-sans">
                              <span className="text-[9px] font-bold uppercase tracking-wider text-slate-400 font-mono mr-1">Stroke Types (Pebble Classifier):</span>
                              {([StrokeType.FREESTYLE, StrokeType.BREASTSTROKE, StrokeType.BACKSTROKE, StrokeType.BUTTERFLY] as StrokeType[]).map(t => {
                                const count = strokeCounts[t] || 0;
                                const color = strokeColors[t];
                                const name = strokeNames[t];
                                return (
                                  <div key={t} className="flex items-center gap-2 bg-slate-950/60 px-3 py-1 rounded-md border border-slate-850/80 hover:bg-slate-950 transition-all">
                                    <span className="w-2.5 h-2.5 rounded-full shrink-0 animate-pulse" style={{ backgroundColor: color }}></span>
                                    <span className="text-slate-300 font-medium">{name}</span>
                                    <span className="font-mono font-bold px-1.5 py-0.5 rounded bg-slate-900 text-[10px] ml-1 border border-slate-800" style={{ color }}>
                                      {count}
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Core Plot Grid */}
                        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
                          
                          {/* Stacked Telemetry Column (Takes 8 cols) */}
                          <div className="xl:col-span-8 flex flex-col gap-4">
                            
                            {/* Stacked Graph 1: Accelerometer (XYZ only) */}
                            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-3">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <Activity className="h-3.5 w-3.5 text-blue-400 animate-pulse" />
                                  <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100">Accelerometer (XYZ Forces)</h3>
                                </div>
                                <span className="text-[10px] font-mono text-slate-400">
                                  {timeWindow === 'first-minute' ? 'First 60s (Full Resolution)' : 'Full Log (Downsampled)'}
                                </span>
                              </div>

                              <div className="h-[210px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                  <LineChart
                                    data={graphData}
                                    syncId="swimSync"
                                    onMouseMove={(state: any) => {
                                      if (state && state.activePayload && state.activePayload.length > 0) {
                                        setHoverReading(state.activePayload[0].payload as PebbleReading);
                                      }
                                    }}
                                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                                  >
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis 
                                      dataKey="timeStr" 
                                      stroke="#475569" 
                                      fontSize={10} 
                                      fontFamily="monospace"
                                      tickLine={false}
                                    />
                                    <YAxis 
                                      stroke="#475569" 
                                      fontSize={10} 
                                      fontFamily="monospace"
                                      tickLine={false}
                                      label={{ value: 'mG Force', angle: -90, position: 'insideLeft', style: { fill: '#64748b', fontSize: 10, fontFamily: 'sans-serif' }, offset: 10 }}
                                    />
                                    <Tooltip
                                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                                      labelStyle={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: '11px', fontWeight: 'bold' }}
                                      itemStyle={{ fontSize: '11px', padding: '1px 0' }}
                                    />
                                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} />
                                    
                                    {/* Render Detected Swim Strokes as Vertical Lines */}
                                    {strokeMarkersWithTypes.map(({ reading, originalIndex, strokeType }) => {
                                      const isEvery10th = (originalIndex + 1) % 10 === 0;
                                      const color = strokeColors[strokeType];
                                      return (
                                        <ReferenceLine
                                          key={`stroke-line-xyz-${originalIndex}`}
                                          x={reading.timeStr}
                                          stroke={color}
                                          strokeWidth={isEvery10th ? 2 : 1}
                                          strokeDasharray={isEvery10th ? "0" : "3 3"}
                                          label={isEvery10th ? { value: `#${originalIndex + 1}`, fill: color, fontSize: 9, fontWeight: 'bold', position: 'top' } : undefined}
                                        />
                                      );
                                    })}

                                    {/* Render Detected Turning Points as Thick Vertical Lines */}
                                    {turnMarkers.map(({ reading, originalIndex }) => (
                                      <ReferenceLine
                                        key={`turn-line-xyz-${originalIndex}`}
                                        x={reading.timeStr}
                                        stroke="#f97316"
                                        strokeWidth={3}
                                        label={{ value: `Turn ${originalIndex + 1}`, fill: '#f97316', fontSize: 10, fontWeight: 'bold', position: 'top' }}
                                      />
                                    ))}

                                    <Line 
                                      type="monotone" 
                                      dataKey="x" 
                                      name="X Axis" 
                                      stroke="#ef4444" 
                                      strokeWidth={1.5} 
                                      dot={false}
                                      activeDot={{ r: 4 }}
                                    />
                                    <Line 
                                      type="monotone" 
                                      dataKey="y" 
                                      name="Y Axis" 
                                      stroke="#10b981" 
                                      strokeWidth={1.5} 
                                      dot={false}
                                      activeDot={{ r: 4 }}
                                    />
                                    <Line 
                                      type="monotone" 
                                      dataKey="z" 
                                      name="Z Axis" 
                                      stroke="#3b82f6" 
                                      strokeWidth={1.5} 
                                      dot={false}
                                      activeDot={{ r: 4 }}
                                    />
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                            </div>

                            {/* Stacked Graph 2: Overall Movement / Magnitude */}
                            <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-3">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <TrendingUp className="h-3.5 w-3.5 text-amber-500" />
                                  <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100 font-sans">Overall Movement (Magnitude)</h3>
                                </div>
                                <span className="text-[10px] font-mono text-slate-400">Total displacement G-force stress vectors</span>
                              </div>

                              <div className="h-[140px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                  <AreaChart
                                    data={graphData}
                                    syncId="swimSync"
                                    onMouseMove={(state: any) => {
                                      if (state && state.activePayload && state.activePayload.length > 0) {
                                        setHoverReading(state.activePayload[0].payload as PebbleReading);
                                      }
                                    }}
                                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                                  >
                                    <defs>
                                      <linearGradient id="colorMagnitude" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.25}/>
                                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                                      </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis 
                                      dataKey="timeStr" 
                                      stroke="#475569" 
                                      fontSize={10} 
                                      fontFamily="monospace"
                                      tickLine={false}
                                    />
                                    <YAxis 
                                      stroke="#475569" 
                                      fontSize={10} 
                                      fontFamily="monospace"
                                      tickLine={false}
                                      label={{ value: 'mG', angle: -90, position: 'insideLeft', style: { fill: '#64748b', fontSize: 10, fontFamily: 'sans-serif' }, offset: 10 }}
                                    />
                                    <Tooltip
                                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                                      labelStyle={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: '11px', fontWeight: 'bold' }}
                                      itemStyle={{ fontSize: '11px', padding: '1px 0' }}
                                    />
                                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} />
                                    
                                    {/* Render Detected Swim Strokes as Vertical Lines */}
                                    {strokeMarkersWithTypes.map(({ reading, originalIndex, strokeType }) => {
                                      const isEvery10th = (originalIndex + 1) % 10 === 0;
                                      const color = strokeColors[strokeType];
                                      return (
                                        <ReferenceLine
                                          key={`stroke-line-mag-${originalIndex}`}
                                          x={reading.timeStr}
                                          stroke={color}
                                          strokeWidth={isEvery10th ? 2 : 1}
                                          strokeDasharray={isEvery10th ? "0" : "3 3"}
                                          label={isEvery10th ? { value: `#${originalIndex + 1}`, fill: color, fontSize: 9, fontWeight: 'bold', position: 'top' } : undefined}
                                        />
                                      );
                                    })}

                                    {/* Render Detected Turning Points as Thick Vertical Lines */}
                                    {turnMarkers.map(({ reading, originalIndex }) => (
                                      <ReferenceLine
                                        key={`turn-line-mag-${originalIndex}`}
                                        x={reading.timeStr}
                                        stroke="#f97316"
                                        strokeWidth={3}
                                        label={{ value: `Turn ${originalIndex + 1}`, fill: '#f97316', fontSize: 10, fontWeight: 'bold', position: 'top' }}
                                      />
                                    ))}

                                    <Area 
                                      type="monotone" 
                                      dataKey="magnitude" 
                                      name="Kinetic Energy" 
                                      stroke="#f59e0b" 
                                      fillOpacity={1} 
                                      fill="url(#colorMagnitude)" 
                                      strokeWidth={1.5}
                                      activeDot={{ r: 4 }}
                                    />
                                  </AreaChart>
                                </ResponsiveContainer>
                              </div>
                            </div>

                            {/* Stacked Graph 3: Compass Heading (Only if present in file) */}
                            {parseResult.hasCompass && (
                              <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-3">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <Compass className="h-3.5 w-3.5 text-rose-400" />
                                    <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100 font-sans">Compass Heading</h3>
                                  </div>
                                  <span className="text-[10px] font-mono text-slate-400">Heading degrees (0° - 360°)</span>
                                </div>

                                <div className="h-[140px] w-full">
                                  <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart
                                      data={graphData}
                                      syncId="swimSync"
                                      onMouseMove={(state: any) => {
                                        if (state && state.activePayload && state.activePayload.length > 0) {
                                          setHoverReading(state.activePayload[0].payload as PebbleReading);
                                        }
                                      }}
                                      margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                                    >
                                      <defs>
                                        <linearGradient id="colorCompass" x1="0" y1="0" x2="0" y2="1">
                                          <stop offset="5%" stopColor="#ec4899" stopOpacity={0.25}/>
                                          <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                                        </linearGradient>
                                      </defs>
                                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                      <XAxis 
                                        dataKey="timeStr" 
                                        stroke="#475569" 
                                        fontSize={10} 
                                        fontFamily="monospace"
                                        tickLine={false}
                                      />
                                      <YAxis 
                                        domain={[0, 360]} 
                                        stroke="#475569" 
                                        fontSize={10} 
                                        fontFamily="monospace"
                                        tickLine={false}
                                        ticks={[0, 90, 180, 270, 360]}
                                        label={{ value: 'Degrees', angle: -90, position: 'insideLeft', style: { fill: '#64748b', fontSize: 10, fontFamily: 'sans-serif' }, offset: 10 }}
                                      />
                                      <Tooltip
                                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                                        labelStyle={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: '11px', fontWeight: 'bold' }}
                                        itemStyle={{ fontSize: '11px', padding: '1px 0' }}
                                      />
                                      <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} />
                                      
                                      {/* Render Detected Swim Strokes as Vertical Lines */}
                                      {strokeMarkersWithTypes.map(({ reading, originalIndex, strokeType }) => {
                                        const isEvery10th = (originalIndex + 1) % 10 === 0;
                                        const color = strokeColors[strokeType];
                                        return (
                                          <ReferenceLine
                                            key={`stroke-line-compass-${originalIndex}`}
                                            x={reading.timeStr}
                                            stroke={color}
                                            strokeWidth={isEvery10th ? 2 : 1}
                                            strokeDasharray={isEvery10th ? "0" : "3 3"}
                                            label={isEvery10th ? { value: `#${originalIndex + 1}`, fill: color, fontSize: 9, fontWeight: 'bold', position: 'top' } : undefined}
                                          />
                                        );
                                      })}

                                      {/* Render Detected Turning Points as Thick Vertical Lines */}
                                      {turnMarkers.map(({ reading, originalIndex }) => (
                                        <ReferenceLine
                                          key={`turn-line-compass-${originalIndex}`}
                                          x={reading.timeStr}
                                          stroke="#f97316"
                                          strokeWidth={3}
                                          label={{ value: `Turn ${originalIndex + 1}`, fill: '#f97316', fontSize: 10, fontWeight: 'bold', position: 'top' }}
                                        />
                                      ))}

                                      {/* Draw Compass Angle */}
                                      {parseResult.detectedFields.includes('heading') && (
                                        <Area 
                                          type="monotone" 
                                          dataKey="compass" 
                                          name="Heading Direction" 
                                          stroke="#ec4899" 
                                          fillOpacity={1} 
                                          fill="url(#colorCompass)" 
                                          strokeWidth={1.5}
                                          activeDot={{ r: 4 }}
                                        />
                                      )}

                                      {/* Draw optional Raw Mag data if compass key wasn't named exactly 'compass' */}
                                      {!parseResult.detectedFields.includes('heading') && parseResult.detectedFields.includes('magX') && (
                                        <Area 
                                          type="monotone" 
                                          dataKey="magX" 
                                          name="Magnetic X Axis" 
                                          stroke="#db2777" 
                                          fillOpacity={0.1}
                                          strokeWidth={1.5}
                                          activeDot={{ r: 4 }}
                                        />
                                      )}
                                    </AreaChart>
                                  </ResponsiveContainer>
                                </div>
                              </div>
                            )}

                          </div>

                          {/* SVG Compass & Spatial 3D Orientation (Takes 4 cols) */}
                          <div className="xl:col-span-4 bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-4">
                            <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
                              <Compass className="h-3.5 w-3.5 text-rose-400" />
                              <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100 font-sans">Orientation Status</h3>
                            </div>

                            {/* Inspection Info Block */}
                            {hoverReading ? (
                              <div className="flex-1 flex flex-col justify-between gap-4">
                                
                                {/* 3D CSS Tilt Device Visualizer */}
                                <div className="flex flex-col items-center justify-center bg-slate-950 rounded py-4 border border-slate-850 relative overflow-hidden h-[120px]">
                                  <div className="absolute top-2 left-3 text-[8px] font-mono text-slate-500 uppercase tracking-wider">3D Device Wireframe</div>
                                  
                                  {/* Pebble smart watch mockup with CSS 3D Rotation */}
                                  <div 
                                    className="w-12 h-14 bg-slate-800 rounded border border-slate-600 shadow-2xl relative flex flex-col items-center justify-center transition-all duration-75"
                                    style={{
                                      transform: `perspective(200px) rotateX(${deviceOrientation.pitch}deg) rotateY(${deviceOrientation.roll}deg)`,
                                      boxShadow: `${-deviceOrientation.roll/2}px ${deviceOrientation.pitch/2}px 15px rgba(0,0,0,0.5)`
                                    }}
                                  >
                                    {/* Straps */}
                                    <div className="absolute -top-3 w-6 h-3 bg-slate-900 rounded-t border-t border-x border-slate-700"></div>
                                    <div className="absolute -bottom-3 w-6 h-3 bg-slate-900 rounded-b border-b border-x border-slate-700"></div>
                                    
                                    {/* Smartwatch Screen content */}
                                    <div className="w-9 h-9 bg-slate-950 rounded border border-slate-700 flex flex-col items-center justify-center p-1 text-[7px] font-mono text-blue-400 select-none">
                                      <div className="text-[6px] text-slate-500">Pebble</div>
                                      <div className="font-bold text-slate-300 text-[8px] leading-tight mt-0.5">X: {hoverReading.x}</div>
                                      <div className="text-slate-400 leading-none">Y: {hoverReading.y}</div>
                                    </div>
                                    
                                    {/* Physical button */}
                                    <div className="absolute right-[-2px] top-1/2 -translate-y-1/2 w-0.5 h-2.5 bg-slate-600 rounded-r"></div>
                                  </div>

                                  {/* Numeric Pitch/Roll stats */}
                                  <div className="absolute bottom-2 right-3 left-3 flex justify-between text-[8px] font-mono text-slate-500">
                                    <span>Pitch: <strong className="text-rose-400">{deviceOrientation.pitch}°</strong></span>
                                    <span>Roll: <strong className="text-emerald-400">{deviceOrientation.roll}°</strong></span>
                                  </div>
                                </div>

                                {/* Active Circular Compass dial */}
                                <div className="flex items-center justify-around gap-2 bg-slate-950 p-2.5 rounded border border-slate-850">
                                  
                                  {/* Compass Dial */}
                                  <div className="relative w-14 h-14 shrink-0 bg-slate-900 border border-slate-800 rounded-full flex items-center justify-center">
                                    {/* Card directions */}
                                    <span className="absolute top-0.5 text-[7px] font-bold text-rose-500 font-mono">N</span>
                                    <span className="absolute bottom-0.5 text-[7px] font-bold text-slate-500 font-mono">S</span>
                                    <span className="absolute left-1 text-[7px] font-bold text-slate-500 font-mono">W</span>
                                    <span className="absolute right-1 text-[7px] font-bold text-slate-500 font-mono">E</span>
                                    
                                    {/* Dial Needle */}
                                    <div 
                                      className="absolute w-0.5 h-10 flex flex-col justify-between transition-transform duration-100 ease-out"
                                      style={{ transform: `rotate(${hoverReading.compass ?? 0}deg)` }}
                                    >
                                      <div className="w-0 h-0 border-l-[1.5px] border-l-transparent border-r-[1.5px] border-r-transparent border-b-[15px] border-b-rose-500"></div>
                                      <div className="w-0 h-0 border-l-[1.5px] border-l-transparent border-r-[1.5px] border-r-transparent border-t-[15px] border-t-slate-500"></div>
                                    </div>
                                    <div className="w-1.5 h-1.5 bg-white border border-slate-800 rounded-full z-10"></div>
                                  </div>

                                  <div className="flex-1 space-y-0.5">
                                    <div className="text-[8px] font-bold text-slate-500 uppercase tracking-widest font-mono">Compass Heading</div>
                                    <div className="text-sm font-bold text-white font-mono flex items-baseline gap-1">
                                      {hoverReading.compass !== undefined ? (
                                        <>
                                          <span>{Math.round(hoverReading.compass)}°</span>
                                          <span className="text-[9px] text-rose-400">
                                            {hoverReading.compass >= 337.5 || hoverReading.compass < 22.5 ? 'N' :
                                             hoverReading.compass >= 22.5 && hoverReading.compass < 67.5 ? 'NE' :
                                             hoverReading.compass >= 67.5 && hoverReading.compass < 112.5 ? 'E' :
                                             hoverReading.compass >= 112.5 && hoverReading.compass < 157.5 ? 'SE' :
                                             hoverReading.compass >= 157.5 && hoverReading.compass < 202.5 ? 'S' :
                                             hoverReading.compass >= 202.5 && hoverReading.compass < 247.5 ? 'SW' :
                                             hoverReading.compass >= 247.5 && hoverReading.compass < 292.5 ? 'W' : 'NW'}
                                          </span>
                                        </>
                                      ) : (
                                        <span className="text-[10px] text-slate-500 italic">N/A</span>
                                      )}
                                    </div>
                                    <div className="text-[8px] text-slate-500 font-mono">
                                      At <span className="text-blue-400 font-semibold">{hoverReading.timeStr}</span>
                                    </div>
                                  </div>

                                </div>

                              </div>
                            ) : (
                              <div className="flex-1 flex items-center justify-center text-center p-6 border border-dashed border-slate-800 rounded text-slate-500 text-[11px]">
                                Hover over the graphs on the left to see dynamic orientation updates!
                              </div>
                            )}

                          </div>

                        </div>

                        {/* Interactive Inspection Table */}
                        <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-3">
                          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100">Visual Data Inspection Grid</h3>
                          <div className="overflow-x-auto rounded border border-slate-800">
                            <table className="w-full text-left border-collapse text-[10px] font-mono">
                              <thead>
                                <tr className="bg-slate-950 text-slate-400 border-b border-slate-800">
                                  <th className="p-2">Time Stamp</th>
                                  <th className="p-2 text-rose-400">Accel X</th>
                                  <th className="p-2 text-emerald-400">Accel Y</th>
                                  <th className="p-2 text-blue-400">Accel Z</th>
                                  <th className="p-2 text-amber-400">Magnitude</th>
                                  <th className="p-2 text-rose-400">Compass</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                                {graphData.slice(0, 8).map((r, i) => (
                                  <tr 
                                    key={i} 
                                    className={`hover:bg-blue-600/5 transition-colors cursor-pointer ${
                                      hoverReading?.index === r.index ? 'bg-blue-600/10' : ''
                                    }`}
                                    onMouseEnter={() => setHoverReading(r)}
                                  >
                                    <td className="p-2 text-slate-200">{r.timeStr}</td>
                                    <td className="p-2">{r.x}</td>
                                    <td className="p-2">{r.y}</td>
                                    <td className="p-2">{r.z}</td>
                                    <td className="p-2 font-bold">{r.magnitude}</td>
                                    <td className="p-2">
                                      {r.compass !== undefined ? `${Math.round(r.compass)}°` : '—'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <p className="text-[10px] text-slate-500 italic">Showing first {Math.min(8, graphData.length)} samples of current selection. Hover over lines to inspect complete telemetry.</p>
                        </div>

                      </div>
                    )}

                    {/* Tab 2: Detailed Stats Analysis */}
                    {activeTab === 'stats' && computedStats && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        
                        {/* Accelerometer analysis */}
                        <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg space-y-4">
                          <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
                            <Activity className="h-3.5 w-3.5 text-blue-400" />
                            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100">Accelerometer Statistical Variance</h3>
                          </div>

                          <div className="space-y-3">
                            {/* X Stats */}
                            <div className="bg-slate-950 p-3 rounded border border-slate-850">
                              <h4 className="text-[10px] font-bold text-rose-400 font-mono uppercase tracking-wider">X Axis Telemetry</h4>
                              <div className="grid grid-cols-3 gap-2 mt-2 text-center text-xs font-mono">
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Min Force</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.x.min} mG</div>
                                </div>
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Max Force</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.x.max} mG</div>
                                </div>
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Average</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.x.avg} mG</div>
                                </div>
                              </div>
                            </div>

                            {/* Y Stats */}
                            <div className="bg-slate-950 p-3 rounded border border-slate-850">
                              <h4 className="text-[10px] font-bold text-emerald-400 font-mono uppercase tracking-wider">Y Axis Telemetry</h4>
                              <div className="grid grid-cols-3 gap-2 mt-2 text-center text-xs font-mono">
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Min Force</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.y.min} mG</div>
                                </div>
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Max Force</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.y.max} mG</div>
                                </div>
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Average</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.y.avg} mG</div>
                                </div>
                              </div>
                            </div>

                            {/* Z Stats */}
                            <div className="bg-slate-950 p-3 rounded border border-slate-850">
                              <h4 className="text-[10px] font-bold text-blue-400 font-mono uppercase tracking-wider">Z Axis Telemetry</h4>
                              <div className="grid grid-cols-3 gap-2 mt-2 text-center text-xs font-mono">
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Min Force</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.z.min} mG</div>
                                </div>
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Max Force</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.z.max} mG</div>
                                </div>
                                <div>
                                  <div className="text-[8px] text-slate-500 uppercase font-bold">Average</div>
                                  <div className="font-bold text-slate-300">{computedStats.accel.z.avg} mG</div>
                                </div>
                              </div>
                            </div>

                          </div>
                        </div>

                        {/* Compass and Magnitude Analysis */}
                        <div className="space-y-4">
                          
                          {/* Compass Stats */}
                          <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg space-y-4">
                            <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
                              <Compass className="h-3.5 w-3.5 text-rose-400" />
                              <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100">Compass Orientation Summary</h3>
                            </div>

                            {computedStats.compass ? (
                              <div className="space-y-2 font-mono text-xs">
                                <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-850">
                                  <span className="text-slate-500">Minimum Angle:</span>
                                  <span className="text-slate-300 font-bold">{computedStats.compass.min}°</span>
                                </div>
                                <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-850">
                                  <span className="text-slate-500">Maximum Angle:</span>
                                  <span className="text-slate-300 font-bold">{computedStats.compass.max}°</span>
                                </div>
                                <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-850">
                                  <span className="text-slate-500">Average Heading:</span>
                                  <span className="text-blue-400 font-bold">{computedStats.compass.avg}°</span>
                                </div>
                              </div>
                            ) : (
                              <p className="text-slate-500 italic text-xs">Compass metrics are not present in this file.</p>
                            )}
                          </div>

                          {/* Movement Severity (Magnitude) */}
                          <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg space-y-4">
                            <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
                              <TrendingUp className="h-3.5 w-3.5 text-amber-500" />
                              <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100">Movement Magnitude Dynamics</h3>
                            </div>

                            <div className="space-y-2 font-mono text-xs">
                              <p className="text-slate-500 font-sans text-[11px] leading-relaxed pb-1">
                                Acceleration magnitude tracks overall watch displacement (G-force stress vectors). Static rest corresponds to ~1000mG (standard Earth gravity).
                              </p>
                              <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-850">
                                <span className="text-slate-500">Minimum Acceleration:</span>
                                <span className="text-slate-300 font-bold">{computedStats.accel.mag.min} mG</span>
                              </div>
                              <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-850">
                                <span className="text-slate-500">Peak G-Force Spike:</span>
                                <span className="text-rose-400 font-bold">{computedStats.accel.mag.max} mG</span>
                              </div>
                              <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-850">
                                <span className="text-slate-500">Average Kinetic Energy:</span>
                                <span className="text-amber-400 font-bold">{computedStats.accel.mag.avg} mG</span>
                              </div>
                            </div>
                          </div>

                        </div>

                      </div>
                    )}

                    {/* Tab 3: Raw JSON Data Browser */}
                    {activeTab === 'raw' && (
                      <div className="bg-slate-900/50 border border-slate-800 p-4 rounded-lg flex flex-col gap-3">
                        <div className="flex items-center justify-between">
                          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-100">Raw JSON Schema & Records</h3>
                          <span className="text-[9px] font-mono text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded uppercase font-bold">
                            Object Schema
                          </span>
                        </div>
                        <p className="text-xs text-slate-400">
                          Direct representation of the JSON telemetry format fetched from Google Drive parent folders.
                        </p>

                        <div className="bg-slate-950 p-4 rounded border border-slate-850 overflow-auto max-h-[380px] font-mono text-[10px] text-slate-300 leading-relaxed shadow-inner">
                          <pre>{JSON.stringify(rawJson, null, 2)}</pre>
                        </div>
                      </div>
                    )}

                  </motion.div>
                ) : (
                  /* Initial Logged-in screen (No file loaded yet) */
                  <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-16 flex flex-col items-center justify-center text-center gap-4 min-h-[450px]">
                    <div className="p-4 bg-blue-500/10 text-blue-400 rounded border border-blue-500/20">
                      <FileJson className="h-8 w-8" />
                    </div>
                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-widest text-white">No Telemetry File Loaded</h3>
                      <p className="text-xs text-slate-400 mt-2 max-w-sm mx-auto leading-relaxed">
                        Select a Pebble JSON log from the file panel on the left to display its interactive graphs. The longest recording is loaded automatically once discovered.
                      </p>
                    </div>
                  </div>
                )}
              </AnimatePresence>

            </section>

          </div>
        )}

      </main>

      {/* Footer bar */}
      <footer className="border-t border-slate-800 py-6 mt-12 bg-slate-900 text-center text-[10px] text-slate-500 font-mono uppercase tracking-widest">
        <div className="max-w-7xl mx-auto px-4">
          <p>© 2026 Pebble Data Explorer • Sandbox Connection Protocol Active</p>
        </div>
      </footer>
    </div>
  );
}
