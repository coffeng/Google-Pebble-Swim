/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { PebbleReading, ParseResult, RawEvent } from '../types';

/**
 * Parse a time string like "14:25.520" or "19:0.800" into milliseconds.
 */
function parseTimeString(timeStr: string): number {
  // Format: "MM:SS.mmm" or "M:S.m"
  const match = timeStr.match(/^(\d+):(\d+)\.(\d+)$/);
  if (!match) return 0;
  const minutes = parseInt(match[1], 10);
  const seconds = parseInt(match[2], 10);
  // Pad or trim milliseconds to 3 digits
  let msStr = match[3];
  if (msStr.length < 3) msStr = msStr.padEnd(3, '0');
  else if (msStr.length > 3) msStr = msStr.slice(0, 3);
  const ms = parseInt(msStr, 10);
  return minutes * 60000 + seconds * 1000 + ms;
}

/**
 * Robustly parses Pebble JSON sensor data.
 * Handles arrays of objects, nested data fields, and detects various accelerometer & compass property names.
 * Also handles the v19+ format where `raw` contains interleaved data arrays and event objects.
 */
export function parsePebbleJson(json: any): ParseResult {
  let rawList: any[] = [];
  const rawEvents: RawEvent[] = [];
  let explicitSampleRate: number | undefined = undefined;
  // 1. Identify the list of readings from the JSON object
  if (Array.isArray(json)) {
    rawList = json;
  } else if (json && typeof json === 'object') {
    // Check for explicit sample rate
    if (typeof json.raw_hz === 'number') {
      explicitSampleRate = json.raw_hz;
    }

    // Check for `raw` array (v19+ pebble-swim-tracker format)
    // This array can contain both data arrays and inline event objects
    if (Array.isArray(json.raw)) {
      for (const item of json.raw) {
        if (Array.isArray(item)) {
          // Data sample: [x, y, z, compass?]
          rawList.push(item);
        } else if (item && typeof item === 'object' && item.type) {
          // Inline event: {"type": "pause", "time": "14:25.520"}
          rawEvents.push({
            type: item.type,
            time: item.time || '',
            timeOffsetMs: item.time ? parseTimeString(item.time) : 0
          });
        }
      }
    }

    // If no raw array found, look for other nested arrays
    if (rawList.length === 0) {
      const possibleKeys = ['data', 'readings', 'samples', 'accel', 'sensor', 'sensors', 'logs', 'records', 'values'];
      for (const key of possibleKeys) {
        if (Array.isArray(json[key])) {
          rawList = json[key];
          break;
        }
      }
    }
    
    // If no explicit list found, find any key that contains an array of arrays, objects, or primitive lists
    if (rawList.length === 0) {
      for (const key in json) {
        if (key === 'events' || key === 'laps') continue; // Skip log/metadata arrays
        if (Array.isArray(json[key]) && json[key].length > 0) {
          rawList = json[key];
          break;
        }
      }
    }
  }

  const readings: PebbleReading[] = [];
  let hasAccelerometer = false;
  let hasCompass = false;
  const detectedFields: Set<string> = new Set();

  if (rawList.length === 0) {
    return {
      readings: [],
      hasAccelerometer: false,
      hasCompass: false,
      detectedFields: [],
      totalDurationMs: 0,
      rawEvents
    };
  }

  // Helper to search for properties in a case-insensitive way
  const findValue = (obj: any, keys: string[]): number | undefined => {
    if (!obj || typeof obj !== 'object') return undefined;
    for (const key of keys) {
      // Direct match
      if (obj[key] !== undefined && typeof obj[key] === 'number') {
        return obj[key];
      }
      // Case-insensitive match
      const lowerKey = key.toLowerCase();
      for (const objKey in obj) {
        if (objKey.toLowerCase() === lowerKey && typeof obj[objKey] === 'number') {
          return obj[objKey];
        }
      }
    }
    return undefined;
  };

  // Pre-analyze the first few items to detect property names
  const testItem = rawList[0];
  
  // Accelerometer keys: x, y, z or accel_x, accel.x, etc.
  const xKeys = ['x', 'accelX', 'accel_x', 'ax', 'accelerometer_x', 'acceleration_x'];
  const yKeys = ['y', 'accelY', 'accel_y', 'ay', 'accelerometer_y', 'acceleration_y'];
  const zKeys = ['z', 'accelZ', 'accel_z', 'az', 'accelerometer_z', 'acceleration_z'];

  // Compass keys
  const compassKeys = ['compass', 'heading', 'bearing', 'direction', 'yaw', 'azimuth', 'angle'];
  const magXKeys = ['magX', 'mag_x', 'mx', 'magnetometer_x', 'magnetic_x'];
  const magYKeys = ['magY', 'mag_y', 'my', 'magnetometer_y', 'magnetic_y'];
  const magZKeys = ['magZ', 'mag_z', 'mz', 'magnetometer_z', 'magnetic_z'];

  // Time/Timestamp keys
  const timeKeys = ['time', 'timestamp', 'timestampMs', 'ms', 't', 'date', 'epoch', 'time_ms'];

  // 2. Extract readings
  let minTime = Infinity;
  let maxTime = -Infinity;
  let hasValidTimestamps = false;

  const intermediateReadings: {
    rawTime?: number;
    x: number;
    y: number;
    z: number;
    compass?: number;
    magX?: number;
    magY?: number;
    magZ?: number;
  }[] = [];

  for (const item of rawList) {
    let x = 0;
    let y = 0;
    let z = 0;
    let compass: number | undefined = undefined;
    let magX: number | undefined = undefined;
    let magY: number | undefined = undefined;
    let magZ: number | undefined = undefined;
    let timeVal: number | undefined = undefined;

    if (Array.isArray(item)) {
      // Array of values: e.g., [x, y, z, heading] or [x, y, z]
      if (item.length >= 3) {
        x = typeof item[0] === 'number' ? item[0] : parseFloat(item[0]) || 0;
        y = typeof item[1] === 'number' ? item[1] : parseFloat(item[1]) || 0;
        z = typeof item[2] === 'number' ? item[2] : parseFloat(item[2]) || 0;
        hasAccelerometer = true;
      }
      if (item.length >= 4) {
        const parsedCompass = typeof item[3] === 'number' ? item[3] : parseFloat(item[3]);
        if (parsedCompass !== undefined && !isNaN(parsedCompass)) {
          compass = parsedCompass;
          hasCompass = true;
          detectedFields.add('heading');
        }
      }
    } else if (item && typeof item === 'object') {
      // Look for nested coordinate objects (e.g. {accel: {x,y,z}, compass: {heading}})
      let accelSource = item;
      if (item.accel && typeof item.accel === 'object') accelSource = item.accel;
      else if (item.accelerometer && typeof item.accelerometer === 'object') accelSource = item.accelerometer;

      x = findValue(accelSource, xKeys) ?? 0;
      y = findValue(accelSource, yKeys) ?? 0;
      z = findValue(accelSource, zKeys) ?? 0;

      if (findValue(accelSource, xKeys) !== undefined) hasAccelerometer = true;

      // Check compass source
      let compassSource = item;
      if (item.compass && typeof item.compass === 'object') compassSource = item.compass;
      else if (item.magnetometer && typeof item.magnetometer === 'object') compassSource = item.magnetometer;

      compass = findValue(compassSource, compassKeys);
      if (compass !== undefined) {
        hasCompass = true;
        detectedFields.add('heading');
      }

      magX = findValue(compassSource, magXKeys);
      magY = findValue(compassSource, magYKeys);
      magZ = findValue(compassSource, magZKeys);

      if (magX !== undefined || magY !== undefined || magZ !== undefined) {
        hasCompass = true; // Magnetometer data serves as compass information
        if (magX !== undefined) detectedFields.add('magX');
        if (magY !== undefined) detectedFields.add('magY');
        if (magZ !== undefined) detectedFields.add('magZ');
      }

      timeVal = findValue(item, timeKeys);
      if (timeVal !== undefined) {
        hasValidTimestamps = true;
        if (timeVal < minTime) minTime = timeVal;
        if (timeVal > maxTime) maxTime = timeVal;
      }
    } else if (typeof item === 'number') {
      x = item;
    }

    intermediateReadings.push({
      rawTime: timeVal,
      x,
      y,
      z,
      compass,
      magX,
      magY,
      magZ
    });
  }

  // 3. Normalize timestamps and generate final readings
  // If no valid timestamps, we assume a standard 25Hz sample rate (40ms spacing) or estimate based on typical values
  const defaultSampleSpacingMs = 40; // 25 Hz
  let sampleRateHz: number | undefined = undefined;

  if (hasValidTimestamps && minTime !== Infinity && maxTime !== Infinity && maxTime > minTime) {
    const totalDuration = maxTime - minTime;
    // Determine if timestamp is in seconds vs milliseconds
    // If maximum difference is small (e.g. < 5000) but sample size is large, it might be in seconds!
    const isSeconds = totalDuration < 10000 && rawList.length > 50;
    const multiplier = isSeconds ? 1000 : 1;

    minTime *= multiplier;
    maxTime *= multiplier;

    const actualDurationMs = maxTime - minTime;
    sampleRateHz = parseFloat((rawList.length / (actualDurationMs / 1000)).toFixed(1));

    intermediateReadings.forEach((reading, index) => {
      let t = reading.rawTime !== undefined ? reading.rawTime * multiplier : minTime + index * defaultSampleSpacingMs;
      const timeOffset = Math.max(0, t - minTime);
      const magnitude = parseFloat(Math.sqrt(reading.x * reading.x + reading.y * reading.y + reading.z * reading.z).toFixed(2));

      readings.push({
        index,
        time: t,
        timeOffset,
        timeStr: formatTimeOffset(timeOffset),
        x: reading.x,
        y: reading.y,
        z: reading.z,
        magnitude,
        compass: reading.compass,
        magX: reading.magX,
        magY: reading.magY,
        magZ: reading.magZ
      });
    });
  } else {
    // Generate sequential artificial time offsets
    sampleRateHz = explicitSampleRate || 25; // Use explicit rate from JSON or assume 25Hz
    const sampleSpacingMs = 1000 / sampleRateHz;
    intermediateReadings.forEach((reading, index) => {
      const timeOffset = index * sampleSpacingMs;
      const magnitude = parseFloat(Math.sqrt(reading.x * reading.x + reading.y * reading.y + reading.z * reading.z).toFixed(2));

      readings.push({
        index,
        time: timeOffset,
        timeOffset,
        timeStr: formatTimeOffset(timeOffset),
        x: reading.x,
        y: reading.y,
        z: reading.z,
        magnitude,
        compass: reading.compass,
        magX: reading.magX,
        magY: reading.magY,
        magZ: reading.magZ
      });
    });
  }

  // Add x, y, z to detected fields
  if (hasAccelerometer) {
    detectedFields.add('accelX');
    detectedFields.add('accelY');
    detectedFields.add('accelZ');
  }

  const totalDurationMs = readings.length > 0 ? readings[readings.length - 1].timeOffset : 0;

  return {
    readings,
    hasAccelerometer,
    hasCompass,
    sampleRateHz: explicitSampleRate || sampleRateHz,
    detectedFields: Array.from(detectedFields),
    totalDurationMs,
    rawEvents
  };
}

/**
 * Formats milliseconds elapsed since start into M:SS.mmm
 */
export function formatTimeOffset(offsetMs: number): string {
  const minutes = Math.floor(offsetMs / 60000);
  const seconds = Math.floor((offsetMs % 60000) / 1000);
  const ms = Math.floor(offsetMs % 1000);
  return `${minutes}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}
