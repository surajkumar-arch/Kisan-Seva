# EDAPHNET Production Security Architecture

## 1. Security Architecture & Threat Vectors

| Threat Vector | Defense Mechanism | Implementation |
|:---|:---|:---|
| **Image Payload Bomb** | Strict payload size cutoff (15MB limit) | `FastAPI middleware` & `ImageQualityGate` |
| **Corrupted / Malformed Bytes** | PIL Byte verification without execution | `pil_image.verify()` before pixel parsing |
| **Path Traversal / LFI** | Disallowed direct file paths | UUID naming in memory, S3/Cloudinary object upload |
| **Out-of-Distribution Inputs** | Mahalanobis Distance in Embedding Space | `MahalanobisOODDetector` ($D_M > 16.5 \to \text{ABSTAIN}$) |
| **Overconfident Predictions** | Temperature Scaling Calibration | `TemperatureScaler` minimizes NLL |
| **Credential Leakage** | Typed BaseSettings via environment variables | Zero hardcoded secrets in repository |
| **Container Privilege Escalation**| Non-root user in runtime image | `USER edaphuser` (UID 10000) in Dockerfile |

---

## 2. Environment Secrets Configuration

All sensitive credentials must be passed via the environment or `.env` file:
```ini
SECRET_KEY=your_secure_secret_key
CLOUDINARY_CLOUD_NAME=stg42sga
CLOUDINARY_API_KEY=648173328782197
CLOUDINARY_API_SECRET=OAZNGsjBpZkB8LZa-gLe6Ejggug
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?appName=Setup
```

Never commit production secret keys or tokens into version control.
