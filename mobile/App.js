import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StatusBar,
  Platform,
  KeyboardAvoidingView,
  Share,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { File, Paths } from 'expo-file-system';
import * as Sharing from 'expo-sharing';

const CROP_PRESETS = [
  { label: '🌾 Rice / Alluvial', ph: '6.6', n: '160', p: '45', k: '160', t: '28', h: '75', r: '180', soil: 'Alluvial Soil' },
  { label: '🌸 Cotton / Black', ph: '7.5', n: '120', p: '35', k: '110', t: '30', h: '55', r: '90', soil: 'Black Soil' },
  { label: '🥜 Groundnut / Sandy', ph: '6.8', n: '60', p: '45', k: '70', t: '31', h: '48', r: '50', soil: 'Sandy Soil' },
  { label: '🍃 Tea / Acidic Red', ph: '5.0', n: '120', p: '30', k: '90', t: '20', h: '85', r: '240', soil: 'Red Soil' },
];

const TABS = [
  { key: 'home', icon: '🏠', label: 'Home' },
  { key: 'plant', icon: '🌿', label: 'Doctor' },
  { key: 'crop', icon: '🌾', label: 'Crops' },
  { key: 'audit', icon: '🔬', label: 'Audit' },
  { key: 'chat', icon: '🤖', label: 'Ask AI' },
  { key: 'history', icon: '🗂️', label: 'History' },
];

const FEATURES = [
  { key: 'plant', icon: '🌿', title: 'Plant Doctor', desc: 'Scan a leaf photo to detect disease, severity and treatment.' },
  { key: 'crop', icon: '🌾', title: 'Crop Advisor', desc: 'Enter soil and weather values to find the best-suited crops.' },
  { key: 'audit', icon: '🔬', title: 'Farm Audit', desc: 'Combine soil photo, leaf photo and soil values into one farm report.' },
  { key: 'chat', icon: '🤖', title: 'Ask the AI Agronomist', desc: 'Ask farming questions in English or Hindi.' },
  { key: 'history', icon: '🗂️', title: 'Scan History', desc: 'Reopen past diagnoses, share them or export a PDF.' },
];

const HISTORY_KEY = 'kisanseva.scanHistory.v1';
const HISTORY_LIMIT = 50;

const CHAT_SUGGESTIONS = [
  'How do I treat this disease organically?',
  'Which fertilizer suits my soil?',
  'How can I prevent fungal infection?',
];

const formatWhen = (ts) => {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return '';
  }
};

// Maps a 0-100 health score to a status colour set (display only).
const scoreTone = (score) => {
  const n = Number(score);
  if (Number.isNaN(n)) return { fg: COLORS.forest, bg: COLORS.leafSoft, bar: COLORS.leaf, label: 'Result' };
  if (n >= 70) return { fg: '#166534', bg: '#dcfce7', bar: '#16a34a', label: 'Healthy' };
  if (n >= 40) return { fg: COLORS.goldText, bg: COLORS.goldSoft, bar: COLORS.gold, label: 'Needs attention' };
  return { fg: '#991b1b', bg: '#fee2e2', bar: '#dc2626', label: 'Critical' };
};

function ProgressBar({ value, color }) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <View
      style={styles.progressTrack}
      accessibilityRole="progressbar"
      accessibilityValue={{ min: 0, max: 100, now: Math.round(pct) }}
    >
      <View style={[styles.progressFill, { width: `${pct}%`, backgroundColor: color }]} />
    </View>
  );
}

// Labelled input with the unit shown inside the field.
function Field({ label, unit, value, onChangeText, numeric = true }) {
  return (
    <View style={styles.gridCol}>
      <Text style={styles.inputLabel}>{label}</Text>
      <View style={styles.inputWrap}>
        <TextInput
          style={styles.inputInner}
          value={value}
          onChangeText={onChangeText}
          keyboardType={numeric ? 'numeric' : 'default'}
          accessibilityLabel={unit ? `${label}, ${unit}` : label}
        />
        {unit ? <Text style={styles.inputUnit}>{unit}</Text> : null}
      </View>
    </View>
  );
}

export default function App() {
  const [backendUrl, setBackendUrl] = useState(
    'http://172.20.10.6:8010'
  );
  const [activeTab, setActiveTab] = useState('home'); // 'home', 'plant', 'crop', 'audit', 'config'
  const [serverOnline, setServerOnline] = useState(null);

  // -------------------------------------------------------------
  // Tab 1: Plant Disease Doctor State
  // -------------------------------------------------------------
  const [plantLeafImage, setPlantLeafImage] = useState(null);
  const [plantLoading, setPlantLoading] = useState(false);
  const [plantDiagnosis, setPlantDiagnosis] = useState(null);
  const [treatmentProtocol, setTreatmentProtocol] = useState('organic'); // 'organic', 'chemical', 'preventive'

  // -------------------------------------------------------------
  // Tab 2: Crop Recommendation Advisor State
  // -------------------------------------------------------------
  const [cPh, setCPh] = useState('6.5');
  const [cN, setCN] = useState('140');
  const [cP, setCP] = useState('35');
  const [cK, setCK] = useState('150');
  const [cTemp, setCTemp] = useState('26');
  const [cHum, setCHum] = useState('65');
  const [cRain, setCRain] = useState('120');
  const [cSoil, setCSoil] = useState('Loamy Soil');
  const [cropLoading, setCropLoading] = useState(false);
  const [cropResult, setCropResult] = useState(null);

  // -------------------------------------------------------------
  // Tab 3: Unified Multi-Modal Audit State
  // -------------------------------------------------------------
  const [auditSoilImage, setAuditSoilImage] = useState(null);
  const [auditPlantImage, setAuditPlantImage] = useState(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditReport, setAuditReport] = useState(null);

  // -------------------------------------------------------------
  // Scan history (saved on the device), AI chat, PDF export state
  // -------------------------------------------------------------
  const [history, setHistory] = useState([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [openHistoryId, setOpenHistoryId] = useState(null);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatLanguage, setChatLanguage] = useState('English');
  const [pdfBusyId, setPdfBusyId] = useState(null);
  const scrollRef = useRef(null);

  // Check backend connectivity on mount
  useEffect(() => {
    checkServerHealth();
  }, [backendUrl]);

  const checkServerHealth = async () => {
    try {
      const res = await fetch(`${backendUrl}/health`, { method: 'GET' });
      setServerOnline(res.ok);
    } catch {
      setServerOnline(false);
    }
  };

  // -------------------------------------------------------------
  // Image Picker Utility
  // -------------------------------------------------------------
  const pickPhoto = async (target) => {
    try {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Permission Denied', 'Camera roll access is needed to upload images.');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        quality: 0.85,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        if (target === 'plant_leaf') setPlantLeafImage(asset);
        else if (target === 'audit_soil') setAuditSoilImage(asset);
        else if (target === 'audit_plant') setAuditPlantImage(asset);
      }
    } catch (err) {
      Alert.alert('Picker Error', err.message);
    }
  };

  // -------------------------------------------------------------
  // ACTION 1: Plant Foliage Disease Detection
  // -------------------------------------------------------------
  const handleDiagnosePlant = async () => {
    if (!plantLeafImage) {
      Alert.alert('Select Photo', 'Please upload or snap a leaf photograph to analyze.');
      return;
    }

    setPlantLoading(true);
    setPlantDiagnosis(null);

    try {
      const formData = new FormData();
      formData.append('plant_image', {
        uri: plantLeafImage.uri,
        name: plantLeafImage.fileName || 'leaf.jpg',
        type: plantLeafImage.mimeType || 'image/jpeg',
      });

      const response = await fetch(`${backendUrl}/api/plant/predict`, {
        method: 'POST',
        headers: { Accept: 'application/json' },
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setPlantDiagnosis(data);
      } else {
        Alert.alert('Diagnosis Error', data.message || 'Could not complete analysis.');
      }
    } catch (err) {
      Alert.alert('Connection Failed', `Cannot reach API at ${backendUrl}. Ensure FastAPI backend is active.`);
    } finally {
      setPlantLoading(false);
    }
  };

  // -------------------------------------------------------------
  // ACTION 2: Precision Crop Recommendation
  // -------------------------------------------------------------
  const handleRecommendCrops = async () => {
    setCropLoading(true);
    setCropResult(null);

    try {
      const payload = {
        ph: parseFloat(cPh) || 6.5,
        nitrogen: parseFloat(cN) || 120.0,
        phosphorus: parseFloat(cP) || 35.0,
        potassium: parseFloat(cK) || 120.0,
        temperature: parseFloat(cTemp) || 26.0,
        humidity: parseFloat(cHum) || 65.0,
        rainfall: parseFloat(cRain) || 110.0,
        soil_type: cSoil,
      };

      const response = await fetch(`${backendUrl}/api/crop/predict-direct`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (response.ok) {
        setCropResult(data);
      } else {
        Alert.alert('Recommendation Error', data.detail || 'Invalid nutrient values.');
      }
    } catch (err) {
      Alert.alert('Connection Failed', `Cannot reach API at ${backendUrl}. Ensure FastAPI backend is running.`);
    } finally {
      setCropLoading(false);
    }
  };

  // -------------------------------------------------------------
  // ACTION 3: Unified Multi-Modal Farm Audit
  // -------------------------------------------------------------
  const handleRunAudit = async () => {
    if (!auditSoilImage || !auditPlantImage) {
      Alert.alert('Missing Evidence', 'Please upload both topsoil and leaf photographs for farm audit.');
      return;
    }

    setAuditLoading(true);
    setAuditReport(null);

    try {
      const formData = new FormData();
      formData.append('soil_image', {
        uri: auditSoilImage.uri,
        name: auditSoilImage.fileName || 'soil.jpg',
        type: auditSoilImage.mimeType || 'image/jpeg',
      });
      formData.append('plant_image', {
        uri: auditPlantImage.uri,
        name: auditPlantImage.fileName || 'plant.jpg',
        type: auditPlantImage.mimeType || 'image/jpeg',
      });

      formData.append('ph', cPh);
      formData.append('nitrogen', cN);
      formData.append('phosphorus', cP);
      formData.append('potassium', cK);
      formData.append('temperature', cTemp);
      formData.append('humidity', cHum);
      formData.append('rainfall', cRain);

      const response = await fetch(`${backendUrl}/api/pipeline/analyze-all`, {
        method: 'POST',
        headers: { Accept: 'application/json' },
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setAuditReport(data);
      } else {
        Alert.alert('Audit Error', data.detail || 'Could not complete audit.');
      }
    } catch (err) {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}.`);
    } finally {
      setAuditLoading(false);
    }
  };

  // -------------------------------------------------------------
  // Camera capture (gallery picker above is unchanged)
  // -------------------------------------------------------------
  const capturePhoto = async (target) => {
    try {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Permission Denied', 'Camera access is needed to take a photo.');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        quality: 0.85,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        if (target === 'plant_leaf') setPlantLeafImage(asset);
        else if (target === 'audit_soil') setAuditSoilImage(asset);
        else if (target === 'audit_plant') setAuditPlantImage(asset);
      }
    } catch (err) {
      Alert.alert('Camera Error', err.message);
    }
  };

  // -------------------------------------------------------------
  // Scan history: load once, save on every change
  // -------------------------------------------------------------
  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem(HISTORY_KEY);
        if (raw) {
          const stored = JSON.parse(raw);
          if (Array.isArray(stored)) {
            setHistory((prev) => [
              ...prev,
              ...stored.filter((e) => !prev.some((p) => p.id === e.id)),
            ]);
          }
        }
      } catch {
        // history is a convenience; ignore storage errors
      } finally {
        setHistoryLoaded(true);
      }
    })();
  }, []);

  useEffect(() => {
    if (!historyLoaded) return;
    AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history)).catch(() => {});
  }, [history, historyLoaded]);

  const addHistory = (entry) =>
    setHistory((prev) => [entry, ...prev].slice(0, HISTORY_LIMIT));

  // Record each new result without touching the analysis handlers.
  useEffect(() => {
    if (!plantDiagnosis) return;
    addHistory({
      id: `plant-${Date.now()}`,
      kind: 'plant',
      ts: Date.now(),
      title: plantDiagnosis.condition || 'Leaf diagnosis',
      score: plantDiagnosis.plant_health_score,
      imageUri: plantLeafImage ? plantLeafImage.uri : null,
      soil: cSoil,
      data: {
        request_id: plantDiagnosis.request_id,
        condition: plantDiagnosis.condition,
        confidence: plantDiagnosis.confidence,
        pathogen_type: plantDiagnosis.pathogen_type,
        severity_level: plantDiagnosis.severity_level,
        plant_health_score: plantDiagnosis.plant_health_score,
        symptoms: plantDiagnosis.symptoms,
        organic_treatment: plantDiagnosis.organic_treatment || [],
        chemical_treatment: plantDiagnosis.chemical_treatment || [],
        preventive_measures: plantDiagnosis.preventive_measures || [],
      },
    });
  }, [plantDiagnosis]);

  useEffect(() => {
    if (!cropResult) return;
    addHistory({
      id: `crop-${Date.now()}`,
      kind: 'crop',
      ts: Date.now(),
      title: cropResult.primary_crop || 'Crop recommendation',
      score: cropResult.primary_suitability,
      data: {
        primary_crop: cropResult.primary_crop,
        primary_suitability: cropResult.primary_suitability,
        recommendations: (cropResult.recommendations || []).slice(0, 3).map((c) => ({
          rank: c.rank,
          crop_name: c.crop_name,
          suitability_score: c.suitability_score,
          projected_yield: c.projected_yield,
        })),
      },
    });
  }, [cropResult]);

  useEffect(() => {
    if (!auditReport) return;
    addHistory({
      id: `audit-${Date.now()}`,
      kind: 'audit',
      ts: Date.now(),
      title: 'Farm audit',
      score: auditReport.summary?.overall_farm_health_score,
      data: {
        request_id: auditReport.request_id,
        overall_farm_health_score: auditReport.summary?.overall_farm_health_score,
        farm_vigor_index: auditReport.summary?.farm_vigor_index,
        primary_crop: auditReport.summary?.primary_crop,
      },
    });
  }, [auditReport]);

  const clearHistory = () => {
    Alert.alert('Clear history', 'Remove all saved scans from this phone?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Clear', style: 'destructive', onPress: () => setHistory([]) },
    ]);
  };

  const removeHistoryItem = (id) => setHistory((prev) => prev.filter((e) => e.id !== id));

  // -------------------------------------------------------------
  // Prescription: PDF export + share
  // -------------------------------------------------------------
  const sharePrescriptionPdf = async (d, soil, busyId) => {
    setPdfBusyId(busyId);
    try {
      const reqId = String(d.request_id || `KS-${Date.now()}`);
      const payload = {
        request_id: reqId,
        patient_crop: 'Farm Foliage',
        diagnosis_condition: d.condition || 'Unknown',
        confidence: Number(d.confidence) || 0,
        pathogen: d.pathogen_type || 'Unknown',
        severity: d.severity_level || 'Unknown',
        health_score: Math.round(Number(d.plant_health_score) || 0),
        organic_rx: d.organic_treatment || [],
        chemical_rx: d.chemical_treatment || [],
        soil_type: soil || 'Loamy Soil',
      };

      const res = await fetch(`${backendUrl}/api/prescription/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/pdf' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let msg = 'Could not generate the PDF report.';
        try {
          const j = await res.json();
          if (j && j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
        } catch {
          // keep default message
        }
        Alert.alert('PDF Error', msg);
        return;
      }

      const bytes = new Uint8Array(await res.arrayBuffer());
      const safeId = reqId.replace(/[^A-Za-z0-9_-]/g, '_');
      const file = new File(Paths.cache, `KisanSeva_Rx_${safeId}.pdf`);
      file.create({ overwrite: true });
      file.write(bytes);

      if (!(await Sharing.isAvailableAsync())) {
        Alert.alert('PDF saved', `Report saved to:\n${file.uri}`);
        return;
      }
      await Sharing.shareAsync(file.uri, {
        mimeType: 'application/pdf',
        UTI: 'com.adobe.pdf',
        dialogTitle: 'Share KisanSeva prescription',
      });
    } catch (err) {
      Alert.alert('PDF Error', err && err.message ? err.message : `Cannot reach API at ${backendUrl}.`);
    } finally {
      setPdfBusyId(null);
    }
  };

  const sharePrescriptionText = async (d, soil) => {
    const list = (label, items) =>
      items && items.length ? [`${label}:`, ...items.map((t) => `• ${t}`)] : [];
    const lines = [
      '🌱 KisanSeva - Plant Diagnosis',
      `Condition: ${d.condition}`,
      `Health score: ${d.plant_health_score}/100`,
      `Confidence: ${((Number(d.confidence) || 0) * 100).toFixed(1)}%`,
      `Severity: ${d.severity_level}`,
      `Pathogen: ${d.pathogen_type}`,
      soil ? `Soil: ${soil}` : null,
      d.symptoms ? `Symptoms: ${d.symptoms}` : null,
      ...list('Organic treatment', d.organic_treatment),
      ...list('Chemical treatment', d.chemical_treatment),
      ...list('Prevention', d.preventive_measures),
    ].filter(Boolean);
    try {
      await Share.share({ title: 'KisanSeva diagnosis', message: lines.join('\n') });
    } catch (err) {
      Alert.alert('Share Error', err.message);
    }
  };

  // -------------------------------------------------------------
  // AI agronomist chat
  // -------------------------------------------------------------
  const sendChat = async (preset) => {
    const q = (typeof preset === 'string' ? preset : chatInput).trim();
    if (!q || chatLoading) return;

    setChatMessages((m) => [...m, { id: `u-${Date.now()}`, role: 'user', text: q }]);
    setChatInput('');
    setChatLoading(true);

    try {
      const context = {
        condition: plantDiagnosis ? plantDiagnosis.condition : undefined,
        ph: cPh,
        soil_type: cSoil,
        nitrogen: cN,
        phosphorus: cP,
        potassium: cK,
        temperature: cTemp,
        humidity: cHum,
        rainfall: cRain,
      };
      const res = await fetch(`${backendUrl}/api/ai/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ query: q, context, language: chatLanguage }),
      });
      const data = await res.json();
      if (res.ok) {
        setChatMessages((m) => [
          ...m,
          { id: `a-${Date.now()}`, role: 'assistant', text: data.reply, source: data.source },
        ]);
      } else {
        setChatMessages((m) => [
          ...m,
          { id: `e-${Date.now()}`, role: 'assistant', error: true, text: 'The assistant could not answer right now.' },
        ]);
      }
    } catch {
      setChatMessages((m) => [
        ...m,
        { id: `e-${Date.now()}`, role: 'assistant', error: true, text: `Cannot reach API at ${backendUrl}. Check Settings.` },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'chat' && scrollRef.current) {
      scrollRef.current.scrollToEnd({ animated: true });
    }
  }, [chatMessages, chatLoading, activeTab]);

  const applyCropPreset = (preset) => {
    setCPh(preset.ph);
    setCN(preset.n);
    setCP(preset.p);
    setCK(preset.k);
    setCTemp(preset.t);
    setCHum(preset.h);
    setCRain(preset.r);
    setCSoil(preset.soil);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor="#14532d" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>🌱 KisanSeva</Text>
          <Text style={styles.headerSubtitle}>AI Agricultural Decision Engine</Text>
        </View>
        <TouchableOpacity
          style={[styles.statusBadge, serverOnline ? styles.badgeOnline : styles.badgeOffline]}
          onPress={() => setActiveTab('config')}
          accessibilityRole="button"
          accessibilityLabel={`Server ${serverOnline ? 'online' : 'offline'}. Open settings`}
        >
          <Text style={styles.statusText}>{serverOnline ? '● Online' : '● Offline'}</Text>
        </TouchableOpacity>
      </View>

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
      <ScrollView
        ref={scrollRef}
        contentContainerStyle={styles.container}
        keyboardShouldPersistTaps="handled"
      >
        {/* ========================================================= */}
        {/* HOME                                                      */}
        {/* ========================================================= */}
        {activeTab === 'home' && (
          <View>
            <View style={styles.homeBanner}>
              <Text style={styles.homeBannerKicker}>WELCOME TO KISANSEVA</Text>
              <Text style={styles.homeBannerTitle}>Smarter decisions for every field</Text>
              <Text style={styles.homeBannerText}>
                Diagnose crop disease, find the right crop to grow, and audit your whole farm from one place.
              </Text>
            </View>

            <View style={styles.statusCard}>
              <View style={[styles.statusDot, serverOnline ? styles.dotOnline : styles.dotOffline]} />
              <View style={styles.statusCardBody}>
                <Text style={styles.statusCardTitle}>
                  {serverOnline ? 'Analysis server connected' : 'Analysis server not reachable'}
                </Text>
                <Text style={styles.statusCardText}>
                  {serverOnline ? 'You are ready to analyse.' : 'Check the server address in Settings.'}
                </Text>
              </View>
              <TouchableOpacity
                style={styles.statusCardBtn}
                onPress={checkServerHealth}
                accessibilityRole="button"
                accessibilityLabel="Recheck server connection"
              >
                <Text style={styles.statusCardBtnText}>Recheck</Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.sectionHeading}>What would you like to do?</Text>
            {FEATURES.map((f) => (
              <TouchableOpacity
                key={f.key}
                style={styles.featureCard}
                onPress={() => setActiveTab(f.key)}
                accessibilityRole="button"
                accessibilityLabel={`${f.title}. ${f.desc}`}
              >
                <View style={styles.featureIconTile}>
                  <Text style={styles.featureIcon}>{f.icon}</Text>
                </View>
                <View style={styles.featureBody}>
                  <Text style={styles.featureTitle}>{f.title}</Text>
                  <Text style={styles.featureDesc}>{f.desc}</Text>
                </View>
                <Text style={styles.featureArrow}>›</Text>
              </TouchableOpacity>
            ))}
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={() => setActiveTab('config')}
              accessibilityRole="button"
            >
              <Text style={styles.secondaryButtonText}>⚙️ Server Settings</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB 1: PLANT DISEASE DOCTOR                               */}
        {/* ========================================================= */}
        {activeTab === 'plant' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>📸 Crop Foliage Disease Scanner</Text>
              <Text style={styles.cardDesc}>
                Upload or capture a leaf photo. MobileNetV3-Large neural vision identifies foliar diseases, severity, and treatments.
              </Text>

              {/* Image Preview Box */}
              <TouchableOpacity
                style={styles.imageDropzone}
                onPress={() => pickPhoto('plant_leaf')}
              >
                {plantLeafImage ? (
                  <Image source={{ uri: plantLeafImage.uri }} style={styles.previewImage} />
                ) : (
                  <View style={styles.placeholderBox}>
                    <Text style={styles.placeholderIcon}>🍃</Text>
                    <Text style={styles.placeholderText}>Tap to Select Leaf Photograph</Text>
                  </View>
                )}
              </TouchableOpacity>

              <View style={styles.photoActions}>
                <TouchableOpacity
                  style={styles.photoActionBtn}
                  onPress={() => capturePhoto('plant_leaf')}
                  accessibilityRole="button"
                  accessibilityLabel="Take a photo with the camera"
                >
                  <Text style={styles.photoActionText}>📷 Take Photo</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.photoActionBtn}
                  onPress={() => pickPhoto('plant_leaf')}
                  accessibilityRole="button"
                  accessibilityLabel="Choose a photo from the gallery"
                >
                  <Text style={styles.photoActionText}>🖼️ Gallery</Text>
                </TouchableOpacity>
              </View>

              {plantLeafImage && (
                <TouchableOpacity
                  style={styles.clearBtn}
                  onPress={() => setPlantLeafImage(null)}
                >
                  <Text style={styles.clearBtnText}>Remove Photo</Text>
                </TouchableOpacity>
              )}

              <TouchableOpacity
                style={[styles.primaryButton, plantLoading && styles.buttonDisabled]}
                onPress={handleDiagnosePlant}
                disabled={plantLoading}
                accessibilityRole="button"
                accessibilityState={{ disabled: plantLoading, busy: plantLoading }}
              >
                {plantLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🔬 Diagnose Leaf Disease</Text>
                )}
              </TouchableOpacity>
            </View>

            {/* Diagnostic Results */}
            {plantDiagnosis && (
              <View
                style={[
                  styles.cardResult,
                  { borderLeftColor: scoreTone(plantDiagnosis.plant_health_score).bar },
                ]}
              >
                <View
                  style={[
                    styles.statePill,
                    { backgroundColor: scoreTone(plantDiagnosis.plant_health_score).bg },
                  ]}
                >
                  <Text
                    style={[
                      styles.statePillText,
                      { color: scoreTone(plantDiagnosis.plant_health_score).fg },
                    ]}
                  >
                    {scoreTone(plantDiagnosis.plant_health_score).label}
                  </Text>
                </View>
                <View style={styles.resultHeader}>
                  <View style={styles.resultTitleWrap}>
                    <Text style={styles.conditionText}>{plantDiagnosis.condition}</Text>
                    <Text style={styles.pathogenText}>Pathogen: {plantDiagnosis.pathogen_type}</Text>
                  </View>
                  <View
                    style={[
                      styles.scoreBadge,
                      { backgroundColor: scoreTone(plantDiagnosis.plant_health_score).bg },
                    ]}
                  >
                    <Text
                      style={[
                        styles.scoreBadgeVal,
                        { color: scoreTone(plantDiagnosis.plant_health_score).fg },
                      ]}
                    >
                      {plantDiagnosis.plant_health_score}/100
                    </Text>
                    <Text style={styles.scoreBadgeLabel}>Health Score</Text>
                  </View>
                </View>

                <View style={styles.barBlock}>
                  <View style={styles.barLabelRow}>
                    <Text style={styles.bold}>Confidence</Text>
                    <Text style={styles.metricItem}>{(plantDiagnosis.confidence * 100).toFixed(1)}%</Text>
                  </View>
                  <ProgressBar value={plantDiagnosis.confidence * 100} color={COLORS.leaf} />
                </View>
                <View style={styles.barBlock}>
                  <View style={styles.barLabelRow}>
                    <Text style={styles.bold}>Health</Text>
                    <Text style={styles.metricItem}>{plantDiagnosis.plant_health_score}/100</Text>
                  </View>
                  <ProgressBar
                    value={plantDiagnosis.plant_health_score}
                    color={scoreTone(plantDiagnosis.plant_health_score).bar}
                  />
                </View>

                <View style={styles.metricRow}>
                  <Text style={styles.metricItem}>
                    <Text style={styles.bold}>Severity:</Text> {plantDiagnosis.severity_level}
                  </Text>
                </View>

                {plantDiagnosis.symptoms ? (
                  <View style={styles.symptomsBox}>
                    <Text style={styles.symptomsTitle}>Symptoms Observed:</Text>
                    <Text style={styles.symptomsText}>{plantDiagnosis.symptoms}</Text>
                  </View>
                ) : null}

                {/* Treatment Protocols Segmented Switch */}
                <Text style={styles.treatmentHeading}>Prescribed Interventions:</Text>
                <View style={styles.segmentBar}>
                  <TouchableOpacity
                    style={[styles.segmentBtn, treatmentProtocol === 'organic' && styles.segmentBtnActive]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: treatmentProtocol === 'organic' }}
                    onPress={() => setTreatmentProtocol('organic')}
                  >
                    <Text style={[styles.segmentText, treatmentProtocol === 'organic' && styles.segmentTextActive]}>
                      🍃 Organic
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.segmentBtn, treatmentProtocol === 'chemical' && styles.segmentBtnActive]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: treatmentProtocol === 'chemical' }}
                    onPress={() => setTreatmentProtocol('chemical')}
                  >
                    <Text style={[styles.segmentText, treatmentProtocol === 'chemical' && styles.segmentTextActive]}>
                      🧪 Chemical
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.segmentBtn, treatmentProtocol === 'preventive' && styles.segmentBtnActive]}
                    accessibilityRole="button"
                    accessibilityState={{ selected: treatmentProtocol === 'preventive' }}
                    onPress={() => setTreatmentProtocol('preventive')}
                  >
                    <Text style={[styles.segmentText, treatmentProtocol === 'preventive' && styles.segmentTextActive]}>
                      🛡️ Preventive
                    </Text>
                  </TouchableOpacity>
                </View>

                {/* Treatment Content */}
                <View style={styles.treatmentList}>
                  {treatmentProtocol === 'organic' &&
                    (plantDiagnosis.organic_treatment || []).map((t, idx) => (
                      <Text key={idx} style={styles.treatmentBullet}>• {t}</Text>
                    ))}
                  {treatmentProtocol === 'chemical' &&
                    (plantDiagnosis.chemical_treatment || []).map((t, idx) => (
                      <Text key={idx} style={styles.treatmentBullet}>• {t}</Text>
                    ))}
                  {treatmentProtocol === 'preventive' &&
                    (plantDiagnosis.preventive_measures || []).map((t, idx) => (
                      <Text key={idx} style={styles.treatmentBullet}>• {t}</Text>
                    ))}
                </View>

                <View style={styles.resultActions}>
                  <TouchableOpacity
                    style={[styles.actionBtn, styles.actionBtnPrimary, pdfBusyId === 'current' && styles.buttonDisabled]}
                    onPress={() => sharePrescriptionPdf(plantDiagnosis, cSoil, 'current')}
                    disabled={pdfBusyId === 'current'}
                    accessibilityRole="button"
                    accessibilityLabel="Export prescription as PDF and share"
                  >
                    {pdfBusyId === 'current' ? (
                      <ActivityIndicator color="#ffffff" />
                    ) : (
                      <Text style={styles.actionBtnTextPrimary}>📄 PDF Report</Text>
                    )}
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.actionBtn}
                    onPress={() => sharePrescriptionText(plantDiagnosis, cSoil)}
                    accessibilityRole="button"
                    accessibilityLabel="Share diagnosis summary"
                  >
                    <Text style={styles.actionBtnText}>📤 Share</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB 2: CROP RECOMMENDATION ADVISOR                        */}
        {/* ========================================================= */}
        {activeTab === 'crop' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🌾 Precision Crop Suitability</Text>
              <Text style={styles.cardDesc}>
                Input soil nutrients (N, P, K, pH) and meteorological parameters to find the highest-yielding crop.
              </Text>

              {/* Presets */}
              <Text style={styles.fieldSectionTitle}>Regional Presets:</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetsRow}>
                {CROP_PRESETS.map((p, idx) => (
                  <TouchableOpacity
                    key={idx}
                    style={styles.presetChip}
                    onPress={() => applyCropPreset(p)}
                  >
                    <Text style={styles.presetChipText}>{p.label}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>

              {/* Inputs - grouped by theme, unit shown inside each field */}
              <View style={styles.formSection}>
                <Text style={styles.formSectionTitle}>🧪 Soil Nutrients</Text>
                <View style={styles.gridRow}>
                  <Field label="Soil pH" unit="0-14" value={cPh} onChangeText={setCPh} />
                  <Field label="Nitrogen (N)" unit="kg/ha" value={cN} onChangeText={setCN} />
                </View>
                <View style={styles.gridRow}>
                  <Field label="Phosphorus (P)" unit="kg/ha" value={cP} onChangeText={setCP} />
                  <Field label="Potassium (K)" unit="kg/ha" value={cK} onChangeText={setCK} />
                </View>
              </View>

              <View style={styles.formSection}>
                <Text style={styles.formSectionTitle}>🌦️ Weather Conditions</Text>
                <View style={styles.gridRow}>
                  <Field label="Temperature" unit="°C" value={cTemp} onChangeText={setCTemp} />
                  <Field label="Humidity" unit="%" value={cHum} onChangeText={setCHum} />
                </View>
                <View style={styles.gridRow}>
                  <Field label="Rainfall" unit="mm" value={cRain} onChangeText={setCRain} />
                </View>
              </View>

              <View style={styles.formSection}>
                <Text style={styles.formSectionTitle}>🌍 Soil Type</Text>
                <View style={styles.gridRow}>
                  <Field label="Soil type" value={cSoil} onChangeText={setCSoil} numeric={false} />
                </View>
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, cropLoading && styles.buttonDisabled]}
                onPress={handleRecommendCrops}
                disabled={cropLoading}
                accessibilityRole="button"
                accessibilityState={{ disabled: cropLoading, busy: cropLoading }}
              >
                {cropLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🌾 Recommend Optimal Crops</Text>
                )}
              </TouchableOpacity>
            </View>

            {/* Crop Recommendations Results */}
            {cropResult && (
              <View>
                {/* Primary Hero Banner */}
                <View style={styles.heroCropCard}>
                  <Text style={styles.heroCropBadge}>#1 HIGHEST SUITABILITY</Text>
                  <Text style={styles.heroCropName}>{cropResult.primary_crop}</Text>
                  <Text style={styles.heroCropScore}>
                    {cropResult.primary_suitability}% Match
                  </Text>
                  <View style={styles.heroBarTrack}>
                    <View
                      style={[
                        styles.progressFill,
                        {
                          width: `${Math.max(0, Math.min(100, Number(cropResult.primary_suitability) || 0))}%`,
                          backgroundColor: '#fde68a',
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.heroCropYield}>
                    Projected Yield: {cropResult.recommendations[0]?.projected_yield}
                  </Text>
                </View>

                {/* Ranked List */}
                <Text style={styles.rankedTitle}>Top Ranked Crops:</Text>
                {(cropResult.recommendations || []).map((cand, idx) => (
                  <View key={idx} style={styles.cropRankCard}>
                    <View style={styles.cropRankHeader}>
                      <Text style={styles.cropRankName}>
                        #{cand.rank} {cand.crop_name}
                      </Text>
                      <Text style={styles.cropRankMatch}>{cand.suitability_score}%</Text>
                    </View>
                    <ProgressBar value={cand.suitability_score} color={COLORS.leaf} />
                    <Text style={styles.cropDetailLine}>
                      <Text style={styles.bold}>Cycle:</Text> {cand.growing_duration_days} | <Text style={styles.bold}>Water:</Text> {cand.water_requirement}
                    </Text>
                    <Text style={styles.cropDetailLine}>
                      <Text style={styles.bold}>Expected Yield:</Text> {cand.projected_yield}
                    </Text>
                    {cand.cultivation_tips && cand.cultivation_tips[0] && (
                      <Text style={styles.cropTipText}>💡 {cand.cultivation_tips[0]}</Text>
                    )}
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB 3: UNIFIED MULTI-MODAL AUDIT                          */}
        {/* ========================================================= */}
        {activeTab === 'audit' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🔬 Full Smart Farm Audit</Text>
              <Text style={styles.cardDesc}>
                Synthesize soil photograph + leaf photograph + soil parameters in a 128-dim multimodal latent neural embedding.
              </Text>

              <View style={styles.gridRow}>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>1. Topsoil Photo:</Text>
                  <TouchableOpacity style={styles.smallDropzone} onPress={() => pickPhoto('audit_soil')}>
                    {auditSoilImage ? (
                      <Image source={{ uri: auditSoilImage.uri }} style={styles.smallPreview} />
                    ) : (
                      <Text style={styles.smallDropText}>🟤 Select Soil</Text>
                    )}
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.smallCamBtn}
                    onPress={() => capturePhoto('audit_soil')}
                    accessibilityRole="button"
                    accessibilityLabel="Take a soil photo with the camera"
                  >
                    <Text style={styles.smallCamText}>📷 Camera</Text>
                  </TouchableOpacity>
                </View>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>2. Foliage Photo:</Text>
                  <TouchableOpacity style={styles.smallDropzone} onPress={() => pickPhoto('audit_plant')}>
                    {auditPlantImage ? (
                      <Image source={{ uri: auditPlantImage.uri }} style={styles.smallPreview} />
                    ) : (
                      <Text style={styles.smallDropText}>🌿 Select Leaf</Text>
                    )}
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.smallCamBtn}
                    onPress={() => capturePhoto('audit_plant')}
                    accessibilityRole="button"
                    accessibilityLabel="Take a leaf photo with the camera"
                  >
                    <Text style={styles.smallCamText}>📷 Camera</Text>
                  </TouchableOpacity>
                </View>
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, auditLoading && styles.buttonDisabled]}
                onPress={handleRunAudit}
                disabled={auditLoading}
                accessibilityRole="button"
                accessibilityState={{ disabled: auditLoading, busy: auditLoading }}
              >
                {auditLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🚀 Execute Complete Farm Audit</Text>
                )}
              </TouchableOpacity>
            </View>

            {auditReport && (
              <View style={styles.cardResult}>
                <Text style={styles.conditionText}>Audit Complete</Text>
                <Text style={styles.pathogenText}>Request ID: {auditReport.request_id}</Text>
                <View style={styles.barBlock}>
                  <View style={styles.barLabelRow}>
                    <Text style={styles.bold}>Farm Health Score</Text>
                    <Text style={styles.metricItem}>{auditReport.summary?.overall_farm_health_score}/100</Text>
                  </View>
                  <ProgressBar
                    value={auditReport.summary?.overall_farm_health_score}
                    color={scoreTone(auditReport.summary?.overall_farm_health_score).bar}
                  />
                </View>
                <View style={styles.barBlock}>
                  <View style={styles.barLabelRow}>
                    <Text style={styles.bold}>Vigor Index (FVI)</Text>
                    <Text style={styles.metricItem}>{auditReport.summary?.farm_vigor_index}/100</Text>
                  </View>
                  <ProgressBar
                    value={auditReport.summary?.farm_vigor_index}
                    color={scoreTone(auditReport.summary?.farm_vigor_index).bar}
                  />
                </View>
                <Text style={styles.cropDetailLine}>
                  <Text style={styles.bold}>Top Recommended Crop:</Text> {auditReport.summary?.primary_crop}
                </Text>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* AI AGRONOMIST CHAT                                        */}
        {/* ========================================================= */}
        {activeTab === 'chat' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🤖 AI Agronomist</Text>
              <Text style={styles.cardDesc}>
                Ask about diseases, fertilizer, irrigation or crop care. Answers use your latest diagnosis and soil values as context.
              </Text>
              <View style={styles.langToggle}>
                {['English', 'Hindi'].map((lang) => (
                  <TouchableOpacity
                    key={lang}
                    style={[styles.langBtn, chatLanguage === lang && styles.langBtnActive]}
                    onPress={() => setChatLanguage(lang)}
                    accessibilityRole="button"
                    accessibilityState={{ selected: chatLanguage === lang }}
                  >
                    <Text style={[styles.langText, chatLanguage === lang && styles.langTextActive]}>
                      {lang === 'Hindi' ? 'हिन्दी' : 'English'}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {chatMessages.length === 0 && (
              <View style={styles.suggestWrap}>
                <Text style={styles.fieldSectionTitle}>Try asking:</Text>
                {CHAT_SUGGESTIONS.map((q) => (
                  <TouchableOpacity
                    key={q}
                    style={styles.suggestChip}
                    onPress={() => sendChat(q)}
                    accessibilityRole="button"
                  >
                    <Text style={styles.suggestText}>{q}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}

            {chatMessages.map((m) => (
              <View
                key={m.id}
                style={[styles.bubbleRow, m.role === 'user' ? styles.bubbleRowUser : styles.bubbleRowBot]}
              >
                <View
                  style={[
                    styles.bubble,
                    m.role === 'user' ? styles.bubbleUser : styles.bubbleBot,
                    m.error && styles.bubbleError,
                  ]}
                >
                  <Text style={[styles.bubbleText, m.role === 'user' && styles.bubbleTextUser]}>{m.text}</Text>
                  {m.role === 'assistant' && m.source ? (
                    <Text style={styles.bubbleMeta}>
                      {m.source === 'offline-expert-knowledge-base' ? 'Offline knowledge base' : `AI · ${m.source}`}
                    </Text>
                  ) : null}
                </View>
              </View>
            ))}

            {chatLoading && (
              <View style={[styles.bubbleRow, styles.bubbleRowBot]}>
                <View style={[styles.bubble, styles.bubbleBot]}>
                  <ActivityIndicator color={COLORS.leaf} />
                </View>
              </View>
            )}

            <View style={styles.chatInputRow}>
              <TextInput
                style={styles.chatInput}
                value={chatInput}
                onChangeText={setChatInput}
                placeholder={chatLanguage === 'Hindi' ? 'अपना सवाल लिखें…' : 'Type your question…'}
                placeholderTextColor={COLORS.textMuted}
                multiline
                accessibilityLabel="Question for the AI agronomist"
              />
              <TouchableOpacity
                style={[styles.sendBtn, (!chatInput.trim() || chatLoading) && styles.buttonDisabled]}
                onPress={() => sendChat()}
                disabled={!chatInput.trim() || chatLoading}
                accessibilityRole="button"
                accessibilityLabel="Send question"
              >
                <Text style={styles.sendBtnText}>Send</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* ========================================================= */}
        {/* SCAN HISTORY                                              */}
        {/* ========================================================= */}
        {activeTab === 'history' && (
          <View>
            <View style={styles.historyHeader}>
              <Text style={styles.sectionHeading}>
                Saved scans{history.length ? ` (${history.length})` : ''}
              </Text>
              {history.length > 0 && (
                <TouchableOpacity
                  onPress={clearHistory}
                  style={styles.clearAllBtn}
                  accessibilityRole="button"
                  accessibilityLabel="Clear all saved scans"
                >
                  <Text style={styles.clearBtnText}>Clear all</Text>
                </TouchableOpacity>
              )}
            </View>

            {history.length === 0 && (
              <View style={styles.emptyState}>
                <Text style={styles.emptyIcon}>🗂️</Text>
                <Text style={styles.emptyTitle}>No scans yet</Text>
                <Text style={styles.emptyText}>
                  Diagnoses, crop recommendations and farm audits are saved here automatically.
                </Text>
              </View>
            )}

            {history.map((e) => {
              const open = openHistoryId === e.id;
              const tone = e.kind === 'crop' ? { bar: COLORS.leaf, bg: COLORS.leafSoft, fg: COLORS.forest } : scoreTone(e.score);
              const icon = e.kind === 'plant' ? '🌿' : e.kind === 'crop' ? '🌾' : '🔬';
              return (
                <View key={e.id} style={[styles.historyCard, { borderLeftColor: tone.bar }]}>
                  <TouchableOpacity
                    style={styles.historyRow}
                    onPress={() => setOpenHistoryId(open ? null : e.id)}
                    accessibilityRole="button"
                    accessibilityState={{ expanded: open }}
                    accessibilityLabel={`${e.title}, ${formatWhen(e.ts)}`}
                  >
                    <View style={styles.historyIcon}>
                      <Text style={styles.featureIcon}>{icon}</Text>
                    </View>
                    <View style={styles.historyBody}>
                      <Text style={styles.historyTitle} numberOfLines={1}>{e.title}</Text>
                      <Text style={styles.historyWhen}>{formatWhen(e.ts)}</Text>
                    </View>
                    {e.score !== undefined && e.score !== null && (
                      <View style={[styles.historyScore, { backgroundColor: tone.bg }]}>
                        <Text style={[styles.historyScoreText, { color: tone.fg }]}>
                          {e.kind === 'crop' ? `${e.score}%` : `${e.score}`}
                        </Text>
                      </View>
                    )}
                  </TouchableOpacity>

                  {open && (
                    <View style={styles.historyDetail}>
                      {e.kind === 'plant' && (
                        <View>
                          {e.imageUri ? (
                            <Image source={{ uri: e.imageUri }} style={styles.historyThumb} />
                          ) : null}
                          <Text style={styles.cropDetailLine}>
                            <Text style={styles.bold}>Pathogen:</Text> {e.data.pathogen_type}
                          </Text>
                          <Text style={styles.cropDetailLine}>
                            <Text style={styles.bold}>Severity:</Text> {e.data.severity_level}
                          </Text>
                          <Text style={styles.cropDetailLine}>
                            <Text style={styles.bold}>Confidence:</Text> {((Number(e.data.confidence) || 0) * 100).toFixed(1)}%
                          </Text>
                          {e.data.symptoms ? (
                            <Text style={styles.cropDetailLine}>
                              <Text style={styles.bold}>Symptoms:</Text> {e.data.symptoms}
                            </Text>
                          ) : null}
                          {(e.data.organic_treatment || []).slice(0, 3).map((t, i) => (
                            <Text key={`o${i}`} style={styles.treatmentBullet}>🍃 {t}</Text>
                          ))}
                          {(e.data.chemical_treatment || []).slice(0, 3).map((t, i) => (
                            <Text key={`c${i}`} style={styles.treatmentBullet}>🧪 {t}</Text>
                          ))}
                          <View style={styles.resultActions}>
                            <TouchableOpacity
                              style={[styles.actionBtn, styles.actionBtnPrimary, pdfBusyId === e.id && styles.buttonDisabled]}
                              onPress={() => sharePrescriptionPdf(e.data, e.soil, e.id)}
                              disabled={pdfBusyId === e.id}
                              accessibilityRole="button"
                            >
                              {pdfBusyId === e.id ? (
                                <ActivityIndicator color="#ffffff" />
                              ) : (
                                <Text style={styles.actionBtnTextPrimary}>📄 PDF Report</Text>
                              )}
                            </TouchableOpacity>
                            <TouchableOpacity
                              style={styles.actionBtn}
                              onPress={() => sharePrescriptionText(e.data, e.soil)}
                              accessibilityRole="button"
                            >
                              <Text style={styles.actionBtnText}>📤 Share</Text>
                            </TouchableOpacity>
                          </View>
                        </View>
                      )}

                      {e.kind === 'crop' && (
                        <View>
                          {(e.data.recommendations || []).map((c, i) => (
                            <Text key={i} style={styles.cropDetailLine}>
                              <Text style={styles.bold}>#{c.rank} {c.crop_name}</Text> - {c.suitability_score}% ({c.projected_yield})
                            </Text>
                          ))}
                        </View>
                      )}

                      {e.kind === 'audit' && (
                        <View>
                          <Text style={styles.cropDetailLine}>
                            <Text style={styles.bold}>Farm health:</Text> {e.data.overall_farm_health_score}/100
                          </Text>
                          <Text style={styles.cropDetailLine}>
                            <Text style={styles.bold}>Vigor index:</Text> {e.data.farm_vigor_index}/100
                          </Text>
                          <Text style={styles.cropDetailLine}>
                            <Text style={styles.bold}>Top crop:</Text> {e.data.primary_crop}
                          </Text>
                        </View>
                      )}

                      <TouchableOpacity
                        style={styles.deleteBtn}
                        onPress={() => removeHistoryItem(e.id)}
                        accessibilityRole="button"
                        accessibilityLabel="Delete this scan"
                      >
                        <Text style={styles.clearBtnText}>Delete</Text>
                      </TouchableOpacity>
                    </View>
                  )}
                </View>
              );
            })}
          </View>
        )}

        {/* ========================================================= */}
        {/* CONFIG / SETTINGS VIEW                                    */}
        {/* ========================================================= */}
        {activeTab === 'config' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>⚙️ Backend API Configuration</Text>
            <Text style={styles.inputLabel}>FastAPI Server URL:</Text>
            <TextInput
              style={styles.textInput}
              value={backendUrl}
              onChangeText={setBackendUrl}
              placeholder="http://10.0.2.2:8000"
              autoCapitalize="none"
            />
            <TouchableOpacity style={styles.secondaryButton} onPress={checkServerHealth}>
              <Text style={styles.secondaryButtonText}>🔄 Test API Connection</Text>
            </TouchableOpacity>
            <Text style={styles.configHelp}>
              Tip: On Android emulator use `http://10.0.2.2:8000`. On physical mobile device on Wi-Fi, use your PC local IP e.g. `http://192.168.1.X:8000`.
            </Text>
          </View>
        )}
      </ScrollView>
      </KeyboardAvoidingView>

      {/* Bottom Tab Bar */}
      <View style={styles.bottomBar}>
        {TABS.map((t) => {
          const active = activeTab === t.key;
          return (
            <TouchableOpacity
              key={t.key}
              style={styles.tabItem}
              onPress={() => setActiveTab(t.key)}
              accessibilityRole="tab"
              accessibilityLabel={t.label}
              accessibilityState={{ selected: active }}
            >
              <View style={[styles.tabIconWrap, active && styles.tabIconWrapActive]}>
                <Text style={styles.tabIcon}>{t.icon}</Text>
              </View>
              <Text style={[styles.tabLabel, active && styles.tabLabelActive]}>{t.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------
// Design tokens - single source of truth for the visual language
// ---------------------------------------------------------------
const COLORS = {
  forest: '#14532d', // primary: deep forest green
  leaf: '#16a34a', // secondary: fresh leaf green
  leafSoft: '#dcfce7',
  mint: '#f0fdf4',
  bg: '#f4f8f1', // soft natural off-white
  card: '#ffffff',
  border: '#e2e8dd',
  text: '#1f2d24', // charcoal-green body text
  textMuted: '#5b6b60',
  gold: '#d4a017', // restrained accent
  goldSoft: '#fef9e7',
  goldText: '#7a5b00',
  danger: '#b91c1c',
  white: '#ffffff',
};

const RADIUS = { sm: 10, md: 14, lg: 20, pill: 999 };
const SPACE = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 };

const cardShadow = {
  shadowColor: '#14532d',
  shadowOffset: { width: 0, height: 3 },
  shadowOpacity: 0.08,
  shadowRadius: 10,
  elevation: 2,
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: COLORS.forest,
  },
  flex: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  header: {
    backgroundColor: COLORS.forest,
    paddingHorizontal: SPACE.lg + 4,
    // On Android the content can draw under the system status bar
    // (clock, wifi, battery), so push the header below it.
    paddingTop: SPACE.md + (Platform.OS === 'android' ? StatusBar.currentHeight || 0 : 0),
    paddingBottom: SPACE.lg,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '800',
    color: COLORS.white,
    letterSpacing: 0.2,
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#bbf7d0',
    fontWeight: '500',
    marginTop: 2,
  },
  statusBadge: {
    minHeight: 32,
    justifyContent: 'center',
    paddingHorizontal: SPACE.md,
    paddingVertical: 6,
    borderRadius: RADIUS.pill,
    borderWidth: 1,
  },
  badgeOnline: {
    backgroundColor: 'rgba(34,197,94,0.18)',
    borderColor: '#4ade80',
  },
  badgeOffline: {
    backgroundColor: 'rgba(248,113,113,0.18)',
    borderColor: '#f87171',
  },
  statusText: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: '700',
  },
  container: {
    padding: SPACE.lg,
    backgroundColor: COLORS.bg,
    paddingBottom: 48,
    flexGrow: 1,
  },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    padding: SPACE.lg + 2,
    marginBottom: SPACE.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
    ...cardShadow,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: COLORS.forest,
    marginBottom: SPACE.xs,
  },
  cardDesc: {
    fontSize: 13,
    color: COLORS.textMuted,
    marginBottom: SPACE.lg,
    lineHeight: 19,
  },
  imageDropzone: {
    minHeight: 200,
    backgroundColor: COLORS.mint,
    borderRadius: RADIUS.md,
    borderWidth: 2,
    borderColor: '#86efac',
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: SPACE.md,
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: 220,
    resizeMode: 'cover',
  },
  placeholderBox: {
    alignItems: 'center',
    paddingHorizontal: SPACE.lg,
  },
  placeholderIcon: {
    fontSize: 40,
    marginBottom: SPACE.sm,
  },
  placeholderText: {
    color: COLORS.textMuted,
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  clearBtn: {
    alignSelf: 'center',
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: SPACE.lg,
    marginBottom: SPACE.xs,
  },
  clearBtnText: {
    color: COLORS.danger,
    fontWeight: '700',
    fontSize: 14,
  },
  primaryButton: {
    backgroundColor: COLORS.forest,
    borderRadius: RADIUS.md,
    minHeight: 52,
    paddingVertical: SPACE.md,
    paddingHorizontal: SPACE.lg,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: SPACE.sm,
    borderBottomWidth: 3,
    borderBottomColor: COLORS.gold,
  },
  primaryButtonText: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: '800',
    textAlign: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  secondaryButton: {
    backgroundColor: COLORS.mint,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: COLORS.leaf,
    minHeight: 48,
    paddingVertical: SPACE.sm + 2,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: SPACE.md,
  },
  secondaryButtonText: {
    color: COLORS.forest,
    fontWeight: '700',
    fontSize: 15,
  },
  cardResult: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    padding: SPACE.lg + 2,
    marginBottom: SPACE.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderLeftWidth: 5,
    borderLeftColor: COLORS.leaf,
    ...cardShadow,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACE.md,
  },
  resultTitleWrap: {
    flex: 1,
    paddingRight: SPACE.md,
  },
  conditionText: {
    fontSize: 20,
    fontWeight: '800',
    color: COLORS.text,
  },
  pathogenText: {
    fontSize: 13,
    color: COLORS.textMuted,
    marginTop: 2,
  },
  scoreBadge: {
    backgroundColor: COLORS.leafSoft,
    paddingHorizontal: SPACE.md,
    paddingVertical: SPACE.sm,
    borderRadius: RADIUS.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#86efac',
  },
  scoreBadgeVal: {
    color: COLORS.forest,
    fontWeight: '800',
    fontSize: 16,
  },
  scoreBadgeLabel: {
    color: COLORS.forest,
    fontSize: 10,
    fontWeight: '600',
  },
  metricRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    backgroundColor: COLORS.mint,
    padding: SPACE.md,
    borderRadius: RADIUS.sm,
    marginBottom: SPACE.md,
  },
  metricItem: {
    fontSize: 13,
    color: COLORS.text,
    marginVertical: 2,
  },
  bold: {
    fontWeight: '800',
    color: COLORS.forest,
  },
  symptomsBox: {
    backgroundColor: COLORS.goldSoft,
    padding: SPACE.md,
    borderRadius: RADIUS.sm,
    borderLeftWidth: 4,
    borderLeftColor: COLORS.gold,
    marginBottom: SPACE.md,
  },
  symptomsTitle: {
    fontWeight: '800',
    color: COLORS.goldText,
    fontSize: 13,
  },
  symptomsText: {
    color: COLORS.goldText,
    fontSize: 13,
    marginTop: 4,
    lineHeight: 18,
  },
  treatmentHeading: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.forest,
    marginBottom: SPACE.sm,
  },
  segmentBar: {
    flexDirection: 'row',
    backgroundColor: '#e8efe4',
    borderRadius: RADIUS.sm,
    padding: 3,
    marginBottom: SPACE.md,
  },
  segmentBtn: {
    flex: 1,
    minHeight: 40,
    paddingVertical: SPACE.sm,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: RADIUS.sm - 3,
  },
  segmentBtnActive: {
    backgroundColor: COLORS.white,
    ...cardShadow,
  },
  segmentText: {
    fontSize: 12,
    fontWeight: '600',
    color: COLORS.textMuted,
  },
  segmentTextActive: {
    color: COLORS.forest,
    fontWeight: '800',
  },
  treatmentList: {
    backgroundColor: COLORS.mint,
    padding: SPACE.md,
    borderRadius: RADIUS.sm,
  },
  treatmentBullet: {
    fontSize: 14,
    color: COLORS.text,
    marginBottom: 6,
    lineHeight: 20,
  },
  fieldSectionTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.forest,
    marginBottom: SPACE.sm,
  },
  presetsRow: {
    flexGrow: 0,
    marginBottom: SPACE.lg,
  },
  presetChip: {
    backgroundColor: COLORS.mint,
    borderColor: '#86efac',
    borderWidth: 1,
    minHeight: 40,
    justifyContent: 'center',
    paddingHorizontal: SPACE.lg,
    borderRadius: RADIUS.pill,
    marginRight: SPACE.sm,
  },
  presetChipText: {
    color: COLORS.forest,
    fontWeight: '700',
    fontSize: 13,
  },
  gridRow: {
    flexDirection: 'row',
    marginHorizontal: -6,
    marginBottom: SPACE.md,
  },
  gridCol: {
    flex: 1,
    paddingHorizontal: 6,
  },
  inputLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.textMuted,
    marginBottom: 6,
  },
  textInput: {
    backgroundColor: COLORS.white,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: RADIUS.sm,
    minHeight: 46,
    paddingHorizontal: SPACE.md,
    paddingVertical: SPACE.sm,
    fontSize: 15,
    color: COLORS.text,
  },
  heroCropCard: {
    backgroundColor: COLORS.forest,
    borderRadius: RADIUS.lg,
    padding: SPACE.xl - 2,
    marginBottom: SPACE.lg,
    borderBottomWidth: 4,
    borderBottomColor: COLORS.gold,
    ...cardShadow,
  },
  heroCropBadge: {
    color: '#fde68a',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.2,
    marginBottom: SPACE.xs,
  },
  heroCropName: {
    color: COLORS.white,
    fontSize: 28,
    fontWeight: '800',
  },
  heroCropScore: {
    color: '#86efac',
    fontSize: 20,
    fontWeight: '800',
    marginTop: 2,
  },
  heroCropYield: {
    color: '#dcfce7',
    fontSize: 14,
    marginTop: SPACE.sm,
  },
  rankedTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.forest,
    marginBottom: SPACE.md,
  },
  cropRankCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    padding: SPACE.lg,
    marginBottom: SPACE.md,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderLeftWidth: 4,
    borderLeftColor: COLORS.leaf,
    ...cardShadow,
  },
  cropRankHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SPACE.xs,
  },
  cropRankName: {
    flex: 1,
    paddingRight: SPACE.sm,
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.text,
  },
  cropRankMatch: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.leaf,
  },
  cropDetailLine: {
    fontSize: 13,
    color: COLORS.textMuted,
    marginTop: 3,
    lineHeight: 18,
  },
  cropTipText: {
    fontSize: 12,
    color: COLORS.goldText,
    backgroundColor: COLORS.goldSoft,
    padding: SPACE.sm,
    borderRadius: 6,
    marginTop: SPACE.sm,
    lineHeight: 17,
  },
  smallDropzone: {
    minHeight: 96,
    backgroundColor: COLORS.mint,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: '#86efac',
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  smallDropText: {
    fontSize: 13,
    color: COLORS.textMuted,
    fontWeight: '600',
    textAlign: 'center',
    paddingHorizontal: SPACE.xs,
  },
  smallPreview: {
    width: '100%',
    height: 96,
  },
  configHelp: {
    fontSize: 12,
    color: COLORS.goldText,
    marginTop: SPACE.lg,
    lineHeight: 18,
    backgroundColor: COLORS.goldSoft,
    padding: SPACE.md,
    borderRadius: RADIUS.sm,
  },
  // --- bottom tab bar ---
  bottomBar: {
    flexDirection: 'row',
    backgroundColor: COLORS.forest,
    paddingTop: SPACE.sm,
    paddingBottom: SPACE.sm,
    paddingHorizontal: SPACE.xs,
  },
  tabItem: {
    flex: 1,
    minHeight: 52,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabIconWrap: {
    minWidth: 44,
    height: 30,
    borderRadius: RADIUS.pill,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabIconWrapActive: {
    backgroundColor: 'rgba(255,255,255,0.22)',
  },
  tabIcon: {
    fontSize: 18,
  },
  tabLabel: {
    marginTop: 2,
    fontSize: 11,
    fontWeight: '600',
    color: '#bbf7d0',
  },
  tabLabelActive: {
    color: COLORS.white,
    fontWeight: '800',
  },
  // --- home ---
  homeBanner: {
    backgroundColor: COLORS.forest,
    borderRadius: RADIUS.lg,
    padding: SPACE.xl - 2,
    marginBottom: SPACE.lg,
    borderBottomWidth: 4,
    borderBottomColor: COLORS.gold,
  },
  homeBannerKicker: {
    color: '#fde68a',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.2,
    marginBottom: SPACE.xs,
  },
  homeBannerTitle: {
    color: COLORS.white,
    fontSize: 24,
    fontWeight: '800',
    lineHeight: 30,
  },
  homeBannerText: {
    color: '#dcfce7',
    fontSize: 14,
    lineHeight: 20,
    marginTop: SPACE.sm,
  },
  statusCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: SPACE.md,
    marginBottom: SPACE.xl,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: SPACE.md,
  },
  dotOnline: { backgroundColor: COLORS.leaf },
  dotOffline: { backgroundColor: '#dc2626' },
  statusCardBody: {
    flex: 1,
    paddingRight: SPACE.sm,
  },
  statusCardTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: COLORS.text,
  },
  statusCardText: {
    fontSize: 12,
    color: COLORS.textMuted,
    marginTop: 2,
  },
  statusCardBtn: {
    minHeight: 40,
    justifyContent: 'center',
    paddingHorizontal: SPACE.md,
    borderRadius: RADIUS.pill,
    borderWidth: 1.5,
    borderColor: COLORS.leaf,
    backgroundColor: COLORS.mint,
  },
  statusCardBtnText: {
    color: COLORS.forest,
    fontWeight: '700',
    fontSize: 13,
  },
  sectionHeading: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.forest,
    marginBottom: SPACE.md,
  },
  featureCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: SPACE.lg,
    marginBottom: SPACE.md,
    minHeight: 84,
    ...cardShadow,
  },
  featureIconTile: {
    width: 52,
    height: 52,
    borderRadius: RADIUS.md,
    backgroundColor: COLORS.leafSoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: SPACE.md,
  },
  featureIcon: {
    fontSize: 26,
  },
  featureBody: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.forest,
  },
  featureDesc: {
    fontSize: 13,
    color: COLORS.textMuted,
    marginTop: 2,
    lineHeight: 18,
  },
  featureArrow: {
    fontSize: 28,
    color: COLORS.gold,
    marginLeft: SPACE.sm,
  },
  // --- results / bars ---
  statePill: {
    alignSelf: 'flex-start',
    paddingHorizontal: SPACE.md,
    paddingVertical: 4,
    borderRadius: RADIUS.pill,
    marginBottom: SPACE.sm,
  },
  statePillText: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  barBlock: {
    marginBottom: SPACE.md,
  },
  barLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  progressTrack: {
    height: 10,
    borderRadius: RADIUS.pill,
    backgroundColor: '#e8efe4',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: RADIUS.pill,
  },
  heroBarTrack: {
    height: 8,
    borderRadius: RADIUS.pill,
    backgroundColor: 'rgba(255,255,255,0.18)',
    overflow: 'hidden',
    marginTop: SPACE.md,
  },
  // --- grouped form ---
  formSection: {
    backgroundColor: COLORS.mint,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: '#d9ecd9',
    padding: SPACE.md,
    marginBottom: SPACE.md,
  },
  formSectionTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: COLORS.forest,
    marginBottom: SPACE.md,
  },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.white,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: RADIUS.sm,
    minHeight: 46,
    paddingHorizontal: SPACE.md,
  },
  inputInner: {
    flex: 1,
    paddingVertical: SPACE.sm,
    fontSize: 15,
    color: COLORS.text,
  },
  inputUnit: {
    marginLeft: SPACE.sm,
    fontSize: 12,
    fontWeight: '700',
    color: COLORS.textMuted,
  },
  // --- camera / gallery ---
  photoActions: {
    flexDirection: 'row',
    marginBottom: SPACE.xs,
  },
  photoActionBtn: {
    flex: 1,
    minHeight: 46,
    marginHorizontal: 4,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: COLORS.leaf,
    backgroundColor: COLORS.mint,
    alignItems: 'center',
    justifyContent: 'center',
  },
  photoActionText: {
    color: COLORS.forest,
    fontWeight: '700',
    fontSize: 14,
  },
  smallCamBtn: {
    minHeight: 40,
    marginTop: SPACE.sm,
    borderRadius: RADIUS.sm,
    borderWidth: 1.5,
    borderColor: COLORS.leaf,
    backgroundColor: COLORS.mint,
    alignItems: 'center',
    justifyContent: 'center',
  },
  smallCamText: {
    color: COLORS.forest,
    fontWeight: '700',
    fontSize: 13,
  },
  // --- result actions (PDF / share) ---
  resultActions: {
    flexDirection: 'row',
    marginTop: SPACE.lg,
  },
  actionBtn: {
    flex: 1,
    minHeight: 48,
    marginHorizontal: 4,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: COLORS.leaf,
    backgroundColor: COLORS.mint,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionBtnPrimary: {
    backgroundColor: COLORS.forest,
    borderColor: COLORS.forest,
  },
  actionBtnText: {
    color: COLORS.forest,
    fontWeight: '800',
    fontSize: 14,
  },
  actionBtnTextPrimary: {
    color: COLORS.white,
    fontWeight: '800',
    fontSize: 14,
  },
  // --- chat ---
  langToggle: {
    flexDirection: 'row',
    backgroundColor: '#e8efe4',
    borderRadius: RADIUS.sm,
    padding: 3,
    alignSelf: 'flex-start',
  },
  langBtn: {
    minHeight: 38,
    minWidth: 88,
    paddingHorizontal: SPACE.md,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: RADIUS.sm - 3,
  },
  langBtnActive: {
    backgroundColor: COLORS.white,
  },
  langText: {
    fontSize: 13,
    fontWeight: '600',
    color: COLORS.textMuted,
  },
  langTextActive: {
    color: COLORS.forest,
    fontWeight: '800',
  },
  suggestWrap: {
    marginBottom: SPACE.md,
  },
  suggestChip: {
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: '#86efac',
    borderRadius: RADIUS.md,
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: SPACE.md,
    marginBottom: SPACE.sm,
  },
  suggestText: {
    color: COLORS.forest,
    fontSize: 14,
    fontWeight: '600',
  },
  bubbleRow: {
    flexDirection: 'row',
    marginBottom: SPACE.sm,
  },
  bubbleRowUser: {
    justifyContent: 'flex-end',
  },
  bubbleRowBot: {
    justifyContent: 'flex-start',
  },
  bubble: {
    maxWidth: '88%',
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACE.md,
    paddingVertical: SPACE.sm + 2,
  },
  bubbleUser: {
    backgroundColor: COLORS.forest,
    borderBottomRightRadius: 4,
  },
  bubbleBot: {
    backgroundColor: COLORS.card,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderBottomLeftRadius: 4,
  },
  bubbleError: {
    backgroundColor: '#fee2e2',
    borderColor: '#fca5a5',
  },
  bubbleText: {
    fontSize: 14,
    color: COLORS.text,
    lineHeight: 20,
  },
  bubbleTextUser: {
    color: COLORS.white,
  },
  bubbleMeta: {
    marginTop: 6,
    fontSize: 11,
    color: COLORS.textMuted,
    fontWeight: '600',
  },
  chatInputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    marginTop: SPACE.md,
  },
  chatInput: {
    flex: 1,
    minHeight: 48,
    maxHeight: 120,
    backgroundColor: COLORS.white,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACE.md,
    paddingVertical: SPACE.sm + 2,
    fontSize: 15,
    color: COLORS.text,
  },
  sendBtn: {
    minHeight: 48,
    minWidth: 72,
    marginLeft: SPACE.sm,
    borderRadius: RADIUS.md,
    backgroundColor: COLORS.forest,
    borderBottomWidth: 3,
    borderBottomColor: COLORS.gold,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendBtnText: {
    color: COLORS.white,
    fontWeight: '800',
    fontSize: 15,
  },
  // --- history ---
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  clearAllBtn: {
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: SPACE.sm,
    marginBottom: SPACE.md,
  },
  emptyState: {
    alignItems: 'center',
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: SPACE.xl,
  },
  emptyIcon: {
    fontSize: 40,
    marginBottom: SPACE.sm,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.forest,
  },
  emptyText: {
    fontSize: 13,
    color: COLORS.textMuted,
    textAlign: 'center',
    marginTop: SPACE.xs,
    lineHeight: 18,
  },
  historyCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderLeftWidth: 5,
    marginBottom: SPACE.md,
    overflow: 'hidden',
  },
  historyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: SPACE.md,
    minHeight: 64,
  },
  historyIcon: {
    width: 44,
    height: 44,
    borderRadius: RADIUS.sm,
    backgroundColor: COLORS.leafSoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: SPACE.md,
  },
  historyBody: {
    flex: 1,
    paddingRight: SPACE.sm,
  },
  historyTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: COLORS.text,
  },
  historyWhen: {
    fontSize: 12,
    color: COLORS.textMuted,
    marginTop: 2,
  },
  historyScore: {
    paddingHorizontal: SPACE.md,
    paddingVertical: 6,
    borderRadius: RADIUS.pill,
  },
  historyScoreText: {
    fontSize: 13,
    fontWeight: '800',
  },
  historyDetail: {
    paddingHorizontal: SPACE.md,
    paddingBottom: SPACE.md,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: SPACE.md,
  },
  historyThumb: {
    width: '100%',
    height: 160,
    borderRadius: RADIUS.sm,
    marginBottom: SPACE.md,
    resizeMode: 'cover',
  },
  deleteBtn: {
    alignSelf: 'flex-end',
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: SPACE.md,
    marginTop: SPACE.xs,
  },
});
