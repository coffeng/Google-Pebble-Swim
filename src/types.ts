/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface DriveFolder {
  id: string;
  name: string;
}

export interface DriveFile {
  id: string;
  name: string;
  size?: string | number;
  modifiedTime?: string;
  sizeString?: string;
}

export interface PebbleReading {
  index: number;
  time: number;          // raw numeric timestamp or elapsed ms
  timeOffset: number;    // ms since start of recording
  timeStr: string;       // formatted time (e.g., "0:12.4")
  x: number;             // Accelerometer X
  y: number;             // Accelerometer Y
  z: number;             // Accelerometer Z
  magnitude: number;     // sqrt(x^2 + y^2 + z^2)
  compass?: number;      // Compass heading (degrees, 0-360)
  magX?: number;         // Optional raw magnetometer X
  magY?: number;         // Optional raw magnetometer Y
  magZ?: number;         // Optional raw magnetometer Z
}

export interface ParseResult {
  readings: PebbleReading[];
  hasAccelerometer: boolean;
  hasCompass: boolean;
  sampleRateHz?: number;
  detectedFields: string[];
  totalDurationMs: number;
}
