# Smart Agriculture Mobile App (React Native / Expo)

The Phase 1 mobile application client allows farmers and agronomists to capture or upload soil and plant images from their mobile device, record soil parameters and environmental measurements, and send them directly to the FastAPI backend for validation and preprocessing.

## Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- [Expo Go app](https://expo.dev/client) installed on Android / iOS device, or an Android/iOS emulator

## Quickstart

1. Navigate to the mobile directory:
   ```bash
   cd mobile
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Expo development server:
   ```bash
   npx expo start
   ```

4. Scan the QR code with the **Expo Go** app on your phone, or press `a` for Android Emulator / `i` for iOS Simulator / `w` for Web preview.

## Connecting to FastAPI

- When running on an **Android Emulator**, the app defaults to `http://10.0.2.2:8000`.
- When running on a **Physical Device (via Expo Go)**, tap the "Change" link at the top of the app and set your computer's local Wi-Fi IP address (e.g. `http://192.168.1.50:8000`).
- Ensure FastAPI is started with host `0.0.0.0`:
  ```bash
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
