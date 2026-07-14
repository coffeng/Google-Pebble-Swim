/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { initializeApp } from 'firebase/app';
import { getAuth, signInWithPopup, GoogleAuthProvider, onAuthStateChanged, User } from 'firebase/auth';
import { DriveFile, DriveFolder } from '../types';
import firebaseConfig from '../../firebase-applet-config.json';

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

const provider = new GoogleAuthProvider();
// Request Google Drive Read-only scope
provider.addScope('https://www.googleapis.com/auth/drive.readonly');

let isSigningIn = false;
let cachedAccessToken: string | null = null;

// Initialize auth state listener
export const initAuth = (
  onAuthSuccess?: (user: User, token: string) => void,
  onAuthFailure?: () => void
) => {
  return onAuthStateChanged(auth, async (user: User | null) => {
    if (user) {
      if (cachedAccessToken) {
        if (onAuthSuccess) onAuthSuccess(user, cachedAccessToken);
      } else if (!isSigningIn) {
        // If we have a user but no cached token (e.g. page reload),
        // we will need the user to click sign-in to refresh the OAuth credential,
        // since Firebase Auth's idToken does not contain the third-party provider access token.
        cachedAccessToken = null;
        if (onAuthFailure) onAuthFailure();
      }
    } else {
      cachedAccessToken = null;
      if (onAuthFailure) onAuthFailure();
    }
  });
};

// Sign in with Google (Popup)
export const googleSignIn = async (): Promise<{ user: User; accessToken: string } | null> => {
  try {
    isSigningIn = true;
    const result = await signInWithPopup(auth, provider);
    const credential = GoogleAuthProvider.credentialFromResult(result);
    if (!credential?.accessToken) {
      throw new Error('Failed to get Google Drive access token from authentication');
    }

    cachedAccessToken = credential.accessToken;
    return { user: result.user, accessToken: cachedAccessToken };
  } catch (error: any) {
    console.error('Google Sign-In Error:', error);
    throw error;
  } finally {
    isSigningIn = false;
  }
};

export const getAccessToken = (): string | null => {
  return cachedAccessToken;
};

export const logout = async () => {
  await auth.signOut();
  cachedAccessToken = null;
};

// Helper for Google API requests
async function googleFetch(url: string, accessToken: string, options: RequestInit = {}) {
  const headers = {
    ...options.headers,
    Authorization: `Bearer ${accessToken}`,
  };
  const response = await fetch(url, { ...options, headers });
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Google API request failed: ${response.status} - ${errText}`);
  }
  return response;
}

// 1. Search for Pebble Folders
export const searchPebbleFolders = async (accessToken: string): Promise<DriveFolder[]> => {
  const query = encodeURIComponent("mimeType = 'application/vnd.google-apps.folder' and (name contains 'pebble' or name contains 'Pebble') and trashed = false");
  const url = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id, name)`;
  
  const response = await googleFetch(url, accessToken);
  const data = await response.json();
  return data.files || [];
};

// 2. Search for JSON files inside folders, or globally if no folders or no files inside folders are found
export const listPebbleJsonFiles = async (accessToken: string, folders: DriveFolder[]): Promise<DriveFile[]> => {
  let files: DriveFile[] = [];

  // Try to list files inside Pebble folders
  if (folders.length > 0) {
    const parentQueries = folders.map(f => `'${f.id}' in parents`).join(' or ');
    const query = encodeURIComponent(`(${parentQueries}) and mimeType = 'application/json' and trashed = false`);
    const url = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id, name, size, modifiedTime)&orderBy=name`;
    
    try {
      const response = await googleFetch(url, accessToken);
      const data = await response.json();
      files = data.files || [];
    } catch (e) {
      console.warn('Error fetching files inside Pebble folders, trying fallback:', e);
    }
  }

  // Fallback / Broad search: if no folders or no files found inside those folders,
  // search globally for any JSON files containing "pebble" in their name.
  if (files.length === 0) {
    const query = encodeURIComponent("mimeType = 'application/json' and (name contains 'pebble' or name contains 'Pebble') and trashed = false");
    const url = `https://www.googleapis.com/drive/v3/files?q=${query}&fields=files(id, name, size, modifiedTime)&orderBy=name`;
    const response = await googleFetch(url, accessToken);
    const data = await response.json();
    files = data.files || [];
  }

  // Format size for display (bytes to KB or MB)
  return files.map(file => {
    const sizeBytes = typeof file.size === 'string' ? parseInt(file.size, 10) : (file.size || 0);
    let sizeString = '0 B';
    if (sizeBytes > 0) {
      if (sizeBytes < 1024) sizeString = `${sizeBytes} B`;
      else if (sizeBytes < 1024 * 1024) sizeString = `${(sizeBytes / 1024).toFixed(1)} KB`;
      else sizeString = `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return {
      ...file,
      size: sizeBytes,
      sizeString
    };
  });
};

// 3. Download raw JSON file content
export const downloadFileContent = async (accessToken: string, fileId: string): Promise<any> => {
  const url = `https://www.googleapis.com/drive/v3/files/${fileId}?alt=media`;
  const response = await googleFetch(url, accessToken);
  return response.json();
};
