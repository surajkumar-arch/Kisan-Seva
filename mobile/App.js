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
  { key: 'yield', icon: '📈', label: 'Yield' },
  { key: 'mandi', icon: '💰', label: 'Mandi' },
  { key: 'vaani', icon: '🎙️', label: 'Vaani' },
  { key: 'insurance', icon: '📜', label: 'Insurance' },
  { key: 'microclimate', icon: '🌦️', label: 'Weather' },
  { key: 'carbon', icon: '🌱', label: 'Carbon' },
  { key: 'irrigation', icon: '💧', label: 'Irrigation' },
  { key: 'epidemic', icon: '☣️', label: 'Epidemic' },
  { key: 'advisor', icon: '📊', label: 'Advisor' },
  { key: 'radar', icon: '🛰️', label: 'Radar' },
  { key: 'ndvi', icon: '🔬', label: 'NDVI' },
  { key: 'drone', icon: '🚁', label: 'Drone' },
  { key: 'council', icon: '⚖️', label: 'Council' },
  { key: 'edge', icon: '⚡', label: 'Edge' },
  { key: 'satellite', icon: '🌊', label: 'Drought' },
  { key: 'tankmix', icon: '🧪', label: 'Spray Mix' },
  { key: 'audit', icon: '📋', label: 'Audit' },
  { key: 'chat', icon: '🤖', label: 'Ask AI' },
  { key: 'history', icon: '🗂️', label: 'History' },
];

const FEATURES = [
  { key: 'plant', icon: '🌿', title: 'Plant Doctor', desc: 'Scan a leaf photo to detect disease, severity and treatment.' },
  { key: 'crop', icon: '🌾', title: 'Crop Advisor', desc: 'Enter soil and weather values to find the best-suited crops.' },
  { key: 'yield', icon: '📈', title: 'Yield & Biomass Forecast', desc: 'WOFOST GDD simulation & 1,000-run Monte Carlo yield uncertainty bounds.' },
  { key: 'mandi', icon: '💰', title: 'Smart Mandi Arbitrage', desc: 'Diesel freight logistics vs APMC price spreads and warehouse storage economics.' },
  { key: 'vaani', icon: '🎙️', title: 'किसान वाणी (Kisan Vaani)', desc: 'Hindi vernacular voice intelligence with step-by-step remedy cards.' },
  { key: 'insurance', icon: '📜', title: 'PMFBY Crop Insurance Assessor', desc: 'Shortfall yield calculation against 7-year benchmarks with SHA-256 dossier.' },
  { key: 'microclimate', icon: '🌦️', title: 'Microclimate & Spray Radar', desc: 'VPD, Penman-Monteith ET0, Delta T droplet physics, and frost/heat emergency alerts.' },
  { key: 'carbon', icon: '🌱', title: 'Farm Ledger & Carbon Offset', desc: 'Double-entry bookkeeping, unit cost/quintal, and IPCC certified carbon credits.' },
  { key: 'irrigation', icon: '💧', title: 'Precision Smart Irrigation', desc: 'Crop ETc, soil water deficit modeling, and PM-KUSUM solar PV pump IoT dispatch.' },
  { key: 'epidemic', icon: '☣️', title: 'Bio-Surveillance Swarm', desc: 'Pathogen R0 dynamics, downwind corridor trajectory, and biosecurity quarantine barriers.' },
  { key: 'advisor', icon: '📊', title: 'Cost & Profit Advisor', desc: 'Fertilizer plan, cost breakdown, MSP mandi revenue, and net profit margin.' },
  { key: 'radar', icon: '🛰️', title: 'Community Outbreak Radar', desc: 'Real-time district disease alert map and epidemic warning radar.' },
  { key: 'ndvi', icon: '🔬', title: 'Synthetic NDVI Infrared', desc: 'Reconstruct NIR spectrum to spot sub-visual chlorophyll stress 7 days early.' },
  { key: 'drone', icon: '🚁', title: 'Drone Field Scanner', desc: 'Autonomous drone flyover simulation with persistent track IDs and HUD minimap.' },
  { key: 'council', icon: '⚖️', title: 'AI Agronomic Council', desc: '3 competing AI experts debate pathology vs soil vs financial risk.' },
  { key: 'edge', icon: '⚡', title: 'Zero-Internet Edge ONNX', desc: 'Offline INT8 local execution with sub-25ms latency in airplane mode.' },
  { key: 'satellite', icon: '🌊', title: 'Satellite Drought Profiler', desc: 'Synthetic NDWI moisture matrices, Penman-Monteith ET0, and drip irrigation scheduling.' },
  { key: 'tankmix', icon: '🧪', title: 'Smart Spray Tank-Mix', desc: 'Chemical antagonism check, biocide hazard alert, and ASABE droplet drift physics.' },
  { key: 'audit', icon: '📋', title: 'Farm Audit', desc: 'Combine soil photo, leaf photo and soil values into one farm report.' },
  { key: 'chat', icon: '🤖', title: 'Ask the AI Agronomist', desc: 'Ask farming questions in English or Hindi with audio readout support.' },
  { key: 'history', icon: '🗂️', title: 'Scan History', desc: 'Reopen past diagnoses, share them or export a PDF.' },
];

const ADVISOR_CROPS = [
  'Wheat', 'Rice', 'Cotton', 'Maize', 'Mustard', 'Tomato', 'Sugarcane', 'Potato', 'Soybean', 'Groundnut'
];

const MULTILINGUAL_VOICE_PROMPTS = [
  { label: '🌾 खाद कब डालें?', query: 'फसल में यूरिया और डीएपी खाद डालने का सही समय और तरीका क्या है?' },
  { label: '🐛 कीट नियंत्रण', query: 'फसल को कीटों और फफूंद से बचाने का सबसे असरदार जैविक उपाय क्या है?' },
  { label: '💰 अधिक मुनाफे वाली फसल', query: 'कम पानी और सामान्य लागत में सबसे अधिक मुनाफा देने वाली फसल कौन सी है?' },
  { label: '🍂 पत्तियों में पीलापन', query: 'पत्तियों में पीलापन आने का कारण क्या है और कौन सा पोषक तत्व देना चाहिए?' },
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

const HOST_PRESETS = [
  { label: '💻 Localhost', url: 'http://127.0.0.1:8010' },
  { label: '🤖 Android Emul', url: 'http://10.0.2.2:8010' },
  { label: '📡 Wi-Fi LAN', url: 'http://172.20.10.6:8010' },
];

export default function App() {
  const [backendUrl, setBackendUrl] = useState(
    Platform.OS === 'android' ? 'http://10.0.2.2:8010' : 'http://127.0.0.1:8010'
  );
  const [activeTab, setActiveTab] = useState('home');
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
  const [cropDistrict, setCropDistrict] = useState('Ludhiana');
  const [liveWeatherLoading, setLiveWeatherLoading] = useState(false);
  const [liveWeatherTelemetry, setLiveWeatherTelemetry] = useState(null);

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
  // Community Radar State & Handlers
  // -------------------------------------------------------------
  const [radarOutbreaks, setRadarOutbreaks] = useState([]);
  const [radarLoading, setRadarLoading] = useState(false);
  const [radarDistricts, setRadarDistricts] = useState([]);
  const [radarFilterDistrict, setRadarFilterDistrict] = useState('All');
  const [reportCondition, setReportCondition] = useState('');
  const [reportCrop, setReportCrop] = useState('Wheat');
  const [reportDistrict, setReportDistrict] = useState('Ludhiana, Punjab');
  const [reportSeverity, setReportSeverity] = useState('Moderate');
  const [submittingReport, setSubmittingReport] = useState(false);

  // -------------------------------------------------------------
  // Action & Profit Advisor State & Handlers
  // -------------------------------------------------------------
  const [advCrop, setAdvCrop] = useState('Wheat');
  const [advArea, setAdvArea] = useState('2.0');
  const [advN, setAdvN] = useState('140');
  const [advP, setAdvP] = useState('35');
  const [advK, setAdvK] = useState('120');
  const [advState, setAdvState] = useState('Punjab');
  const [advLoading, setAdvLoading] = useState(false);
  const [advResult, setAdvResult] = useState(null);
  const [advCompareLoading, setAdvCompareLoading] = useState(false);
  const [advCompareList, setAdvCompareList] = useState(null);

  const fetchRadarData = async () => {
    setRadarLoading(true);
    try {
      const [outRes, regRes] = await Promise.all([
        fetch(`${backendUrl}/api/community/outbreaks?days=14&include_demo=true`),
        fetch(`${backendUrl}/api/community/regions`),
      ]);
      if (outRes.ok) {
        const d = await outRes.json();
        setRadarOutbreaks(d.outbreaks || []);
      }
      if (regRes.ok) {
        const r = await regRes.json();
        setRadarDistricts(r.regions || []);
      }
    } catch {
      // offline fallback
    } finally {
      setRadarLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'radar') {
      fetchRadarData();
    }
  }, [activeTab, backendUrl]);

  const handleSeedRadarDemo = async () => {
    try {
      const res = await fetch(`${backendUrl}/api/community/demo/seed`, { method: 'POST' });
      if (res.ok) {
        Alert.alert('Radar Seeded 🛰️', 'Demo field outbreak alerts added successfully.');
        fetchRadarData();
      } else {
        Alert.alert('Seed Failed', 'Could not seed demo reports.');
      }
    } catch {
      Alert.alert('Error', `Cannot reach API at ${backendUrl}`);
    }
  };

  const handleQuickReportFromDiagnosis = async () => {
    if (!plantDiagnosis) return;
    try {
      const payload = {
        condition: plantDiagnosis.condition || 'Leaf Blight',
        confidence: Number(plantDiagnosis.confidence) || 0.85,
        district: reportDistrict || 'Ludhiana, Punjab',
        crop: 'Leaf/Crop',
        severity: plantDiagnosis.severity_level || 'Moderate',
      };
      const res = await fetch(`${backendUrl}/api/community/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        Alert.alert(
          'Alert Published! 🛰️',
          `Your diagnosis for "${plantDiagnosis.condition}" has been shared with the Community Outbreak Radar to notify neighboring farms.`
        );
      } else {
        const err = await res.json();
        Alert.alert('Report Rejected', err.detail || 'Could not submit report.');
      }
    } catch {
      Alert.alert('Network Error', `Could not reach ${backendUrl}`);
    }
  };

  const handleSubmitManualReport = async () => {
    if (!reportCondition.trim()) {
      Alert.alert('Missing Disease Name', 'Please enter the disease name observed in your field.');
      return;
    }
    setSubmittingReport(true);
    try {
      const payload = {
        condition: reportCondition.trim(),
        confidence: 0.90,
        district: reportDistrict,
        crop: reportCrop,
        severity: reportSeverity,
      };
      const res = await fetch(`${backendUrl}/api/community/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        Alert.alert('Field Report Submitted! 📢', 'Thank you! Your report helps prevent widespread crop epidemics.');
        setReportCondition('');
        fetchRadarData();
      } else {
        const err = await res.json();
        Alert.alert('Submission Error', err.detail || 'Failed to submit report.');
      }
    } catch {
      Alert.alert('Network Error', `Could not reach ${backendUrl}`);
    } finally {
      setSubmittingReport(false);
    }
  };

  const handleCalculateEconomics = async (selectedCrop) => {
    const cropToUse = selectedCrop || advCrop;
    setAdvLoading(true);
    setAdvResult(null);
    try {
      const payload = {
        crop: cropToUse,
        area_acres: parseFloat(advArea) || 1.0,
        nitrogen: parseFloat(advN) || parseFloat(cN) || 120.0,
        phosphorus: parseFloat(advP) || parseFloat(cP) || 35.0,
        potassium: parseFloat(advK) || parseFloat(cK) || 120.0,
        state: advState,
      };
      const res = await fetch(`${backendUrl}/api/advisor/economics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setAdvResult(data);
      } else {
        Alert.alert('Advisor Error', data.detail || 'Could not calculate economics.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setAdvLoading(false);
    }
  };

  const handleCompareCrops = async () => {
    setAdvCompareLoading(true);
    setAdvCompareList(null);
    try {
      const payload = {
        crops: ADVISOR_CROPS,
        area_acres: parseFloat(advArea) || 1.0,
        nitrogen: parseFloat(advN) || parseFloat(cN) || 120.0,
        phosphorus: parseFloat(advP) || parseFloat(cP) || 35.0,
        potassium: parseFloat(advK) || parseFloat(cK) || 120.0,
        state: advState,
      };
      const res = await fetch(`${backendUrl}/api/advisor/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setAdvCompareList(data.ranking || []);
      } else {
        Alert.alert('Comparison Error', data.detail || 'Could not rank crops.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setAdvCompareLoading(false);
    }
  };

  // -------------------------------------------------------------
  // Extraordinary Features 1-4 State & Handlers
  // -------------------------------------------------------------
  const [ndviLoading, setNdviLoading] = useState(false);
  const [ndviResult, setNdviResult] = useState(null);
  const [droneLoading, setDroneLoading] = useState(false);
  const [droneResult, setDroneResult] = useState(null);
  const [councilLoading, setCouncilLoading] = useState(false);
  const [councilResult, setCouncilResult] = useState(null);
  const [edgeLoading, setEdgeLoading] = useState(false);
  const [edgeResult, setEdgeResult] = useState(null);

  const handleRunNdvi = async () => {
    if (!plantLeafImage) {
      Alert.alert('Upload Photo', 'Please take or pick a leaf photo first in Doctor tab or select one.');
      return;
    }
    setNdviLoading(true);
    setNdviResult(null);
    try {
      const formData = new FormData();
      formData.append('image', {
        uri: plantLeafImage.uri,
        name: plantLeafImage.fileName || 'leaf.jpg',
        type: plantLeafImage.mimeType || 'image/jpeg',
      });
      const res = await fetch(`${backendUrl}/api/vision/spectral/ndvi-analyze`, {
        method: 'POST',
        headers: { Accept: 'application/json' },
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setNdviResult(data);
      } else {
        Alert.alert('NDVI Error', data.detail || 'Could not compute NDVI.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setNdviLoading(false);
    }
  };

  const handleRunDrone = async () => {
    setDroneLoading(true);
    try {
      const res = await fetch(`${backendUrl}/api/video/drone/demo-flight?plot_acres=5.0&crop_type=Wheat&duration_seconds=15`);
      const data = await res.json();
      if (res.ok) {
        setDroneResult(data);
      } else {
        Alert.alert('Drone Error', data.detail || 'Could not stream flight.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setDroneLoading(false);
    }
  };

  const handleRunCouncil = async () => {
    setCouncilLoading(true);
    try {
      const payload = {
        condition: plantDiagnosis ? plantDiagnosis.condition : 'Yellow Rust',
        crop: 'Wheat',
        severity: plantDiagnosis ? plantDiagnosis.severity_level : 'Moderate',
        farm_size_acres: parseFloat(advArea) || 2.0,
        soil_ph: 6.8,
        upcoming_rain_mm: 35.0,
      };
      const res = await fetch(`${backendUrl}/api/council/debate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setCouncilResult(data);
      } else {
        Alert.alert('Council Error', data.detail || 'Could not convene council.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setCouncilLoading(false);
    }
  };

  const handleRunEdge = async () => {
    if (!plantLeafImage) {
      Alert.alert('Select Photo', 'Pick or capture a leaf photo first to run offline INT8 edge inference.');
      return;
    }
    setEdgeLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', {
        uri: plantLeafImage.uri,
        name: plantLeafImage.fileName || 'leaf.jpg',
        type: plantLeafImage.mimeType || 'image/jpeg',
      });
      const res = await fetch(`${backendUrl}/api/edge/predict-offline`, {
        method: 'POST',
        headers: { Accept: 'application/json' },
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setEdgeResult(data);
      } else {
        Alert.alert('Edge Error', data.detail || 'Could not run offline inference.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setEdgeLoading(false);
    }
  };

  // -------------------------------------------------------------
  // Extraordinary Features 5 & 6: Satellite Drought & Spray Mix
  // -------------------------------------------------------------
  const [satCrop, setSatCrop] = useState('Wheat');
  const [satSoil, setSatSoil] = useState('Alluvial Soil');
  const [satArea, setSatArea] = useState('2.5');
  const [satStage, setSatStage] = useState('mid');
  const [satTemp, setSatTemp] = useState('32');
  const [satHum, setSatHum] = useState('48');
  const [satRain, setSatRain] = useState('8');
  const [satLoading, setSatLoading] = useState(false);
  const [satResult, setSatResult] = useState(null);

  const handleRunSatellite = async (cropChoice) => {
    setSatLoading(true);
    setSatResult(null);
    try {
      const payload = {
        crop: cropChoice || satCrop,
        soil_type: satSoil,
        temperature_c: parseFloat(satTemp) || 32.0,
        humidity_pct: parseFloat(satHum) || 50.0,
        rainfall_14d_mm: parseFloat(satRain) || 8.0,
        growth_stage: satStage,
        land_area_acres: parseFloat(satArea) || 2.5,
      };
      const res = await fetch(`${backendUrl}/api/satellite/moisture-profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setSatResult(data);
      } else {
        Alert.alert('Hydrology Error', data.detail || 'Could not evaluate drought.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setSatLoading(false);
    }
  };

  const [tmChemA, setTmChemA] = useState('Copper Oxychloride (Fungicide)');
  const [tmChemB, setTmChemB] = useState('Trichoderma viride (Bio-agent)');
  const [tmPh, setTmPh] = useState('7.8');
  const [tmWind, setTmWind] = useState('9.0');
  const [tmTemp, setTmTemp] = useState('29.0');
  const [tmHum, setTmHum] = useState('55.0');
  const [tmLoading, setTmLoading] = useState(false);
  const [tmResult, setTmResult] = useState(null);

  const handleRunTankMix = async (overrideA, overrideB) => {
    setTmLoading(true);
    setTmResult(null);
    try {
      const payload = {
        chemical_a: overrideA || tmChemA,
        chemical_b: overrideB || tmChemB,
        water_ph: parseFloat(tmPh) || 7.5,
        wind_speed_kmh: parseFloat(tmWind) || 9.0,
        temperature_c: parseFloat(tmTemp) || 29.0,
        humidity_pct: parseFloat(tmHum) || 55.0,
        tank_capacity_liters: 15.0,
      };
      const res = await fetch(`${backendUrl}/api/spray/tank-mix-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setTmResult(data);
      } else {
        Alert.alert('Spray Mix Error', data.detail || 'Could not evaluate tank mix.');
      }
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setTmLoading(false);
    }
  };

  // -------------------------------------------------------------
  // Night Innovations 1-4 State & Handlers
  // -------------------------------------------------------------
  const [yfCrop, setYfCrop] = useState('Wheat');
  const [yfArea, setYfArea] = useState('3.0');
  const [yfLoading, setYfLoading] = useState(false);
  const [yfResult, setYfResult] = useState(null);

  const handleRunYield = async (cropChoice) => {
    setYfLoading(true);
    setYfResult(null);
    try {
      const payload = {
        crop: cropChoice || yfCrop,
        soil_type: cSoil || 'Alluvial Soil',
        land_area_acres: parseFloat(yfArea) || 3.0,
        soil_nitrogen_kgha: parseFloat(cN) || 120.0,
        soil_phosphorus_kgha: parseFloat(cP) || 35.0,
        soil_potassium_kgha: parseFloat(cK) || 110.0,
        soil_ph: parseFloat(cPh) || 6.8,
        avg_temperature_c: parseFloat(cTemp) || 26.0,
        cumulative_rainfall_mm: parseFloat(cRain) || 280.0,
        disease_severity_loss_pct: plantDiagnosis ? plantDiagnosis.severity_level === 'High' ? 25.0 : 5.0 : 0.0,
        num_simulations: 1000,
      };
      const res = await fetch(`${backendUrl}/api/yield/forecast`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setYfResult(data);
      else Alert.alert('Yield Error', data.detail || 'Could not forecast yield.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setYfLoading(false);
    }
  };

  const [maCrop, setMaCrop] = useState('Wheat');
  const [maQty, setMaQty] = useState('75');
  const [maRate, setMaRate] = useState('2280');
  const [maLoading, setMaLoading] = useState(false);
  const [maResult, setMaResult] = useState(null);

  const handleRunMandi = async (cropChoice) => {
    setMaLoading(true);
    setMaResult(null);
    try {
      const payload = {
        crop: cropChoice || maCrop,
        quantity_quintals: parseFloat(maQty) || 75.0,
        local_mandi_rate_inr: parseFloat(maRate) || 2280.0,
        diesel_price_per_liter: 89.5,
        truck_mileage_km_per_liter: 4.2,
        truck_capacity_quintals: 100.0,
        warehouse_storage_months: 2,
        storage_rent_per_quintal_month: 25.0,
      };
      const res = await fetch(`${backendUrl}/api/mandi/arbitrage-calculator`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setMaResult(data);
      else Alert.alert('Mandi Error', data.detail || 'Could not evaluate arbitrage.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setMaLoading(false);
    }
  };

  const [vaaniQuery, setVaaniQuery] = useState('फसल की पत्तियों में पीलापन आ रहा है क्या खाद डालें?');
  const [vaaniLoading, setVaaniLoading] = useState(false);
  const [vaaniResult, setVaaniResult] = useState(null);

  const handleRunVaani = async (customQ) => {
    setVaaniLoading(true);
    setVaaniResult(null);
    try {
      const payload = {
        query_text: customQ || vaaniQuery,
        preferred_dialect: 'Hindi',
      };
      const res = await fetch(`${backendUrl}/api/vaani/vernacular-query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setVaaniResult(data);
      else Alert.alert('Vaani Error', data.detail || 'Could not process voice query.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setVaaniLoading(false);
    }
  };

  const [insCrop, setInsCrop] = useState('Wheat');
  const [insAcres, setInsAcres] = useState('3.0');
  const [insCause, setInsCause] = useState('Unseasonal Hailstorm & Severe Lodging');
  const [insYield, setInsYield] = useState('7.8');
  const [insLoading, setInsLoading] = useState(false);
  const [insResult, setInsResult] = useState(null);

  const handleRunInsurance = async (cropChoice) => {
    setInsLoading(true);
    setInsResult(null);
    try {
      const payload = {
        farmer_name: 'Suraj Kumar Jaiswal',
        khasra_survey_no: 'KH-842/19',
        district_state: 'Ludhiana, Punjab',
        crop: cropChoice || insCrop,
        insured_acres: parseFloat(insAcres) || 3.0,
        loss_cause: insCause,
        visual_foliar_damage_pct: 45.0,
        satellite_drought_deficit_pct: 0.0,
        assessed_actual_yield_q_acre: parseFloat(insYield) || 7.8,
        gps_lat: 30.9010,
        gps_lon: 75.8573,
      };
      const res = await fetch(`${backendUrl}/api/insurance/assess-claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setInsResult(data);
      else Alert.alert('Insurance Error', data.detail || 'Could not assess claim.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setInsLoading(false);
    }
  };

  // -------------------------------------------------------------
  // Deep-Tech Innovations 1-4 State & Handlers
  // -------------------------------------------------------------
  const [mcFarm, setMcFarm] = useState('Harit Krishi Kshetra');
  const [mcLoc, setMcLoc] = useState('Karnal, Haryana');
  const [mcCrop, setMcCrop] = useState('Wheat');
  const [mcTemp, setMcTemp] = useState('26.5');
  const [mcHum, setMcHum] = useState('62');
  const [mcWind, setMcWind] = useState('9.5');
  const [mcLoading, setMcLoading] = useState(false);
  const [mcResult, setMcResult] = useState(null);

  const handleRunMicroclimate = async () => {
    setMcLoading(true);
    setMcResult(null);
    try {
      const payload = {
        farm_name: mcFarm,
        location: mcLoc,
        crop_type: mcCrop,
        temp_c: parseFloat(mcTemp) || 26.5,
        temp_max_c: (parseFloat(mcTemp) || 26.5) + 5.0,
        temp_min_c: (parseFloat(mcTemp) || 26.5) - 12.0,
        relative_humidity: parseFloat(mcHum) || 62.0,
        wind_kmh: parseFloat(mcWind) || 9.5,
        soil_temp_c: 22.0,
        rain_probability_pct: 15.0,
      };
      const res = await fetch(`${backendUrl}/api/microclimate/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setMcResult(data);
      else Alert.alert('Microclimate Error', data.detail || 'Could not analyze microclimate.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setMcLoading(false);
    }
  };

  const [carbFarmer, setCarbFarmer] = useState('Sardar Gurpreet Singh');
  const [carbId, setCarbId] = useState('PB-LDH-098');
  const [carbHa, setCarbHa] = useState('3.5');
  const [carbLoading, setCarbLoading] = useState(false);
  const [carbResult, setCarbResult] = useState(null);

  const handleRunCarbon = async () => {
    setCarbLoading(true);
    setCarbResult(null);
    try {
      const payload = {
        farmer_name: carbFarmer,
        farm_id: carbId,
        farm_size_hectares: parseFloat(carbHa) || 3.5,
        adopted_practices: ['residue_management', 'conservation_tillage', 'solar_pump'],
      };
      const res = await fetch(`${backendUrl}/api/carbon/audit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setCarbResult(data);
      else Alert.alert('Carbon Error', data.detail || 'Could not audit carbon credits.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setCarbLoading(false);
    }
  };

  const [irrId, setIrrId] = useState('PB-BTI-014');
  const [irrCrop, setIrrCrop] = useState('cotton');
  const [irrStage, setIrrStage] = useState('flowering');
  const [irrSoil, setIrrSoil] = useState('black_cotton');
  const [irrArea, setIrrArea] = useState('4.5');
  const [irrMoist, setIrrMoist] = useState('16.5');
  const [irrLoading, setIrrLoading] = useState(false);
  const [irrResult, setIrrResult] = useState(null);

  const handleRunIrrigation = async () => {
    setIrrLoading(true);
    setIrrResult(null);
    try {
      const payload = {
        farm_id: irrId,
        crop_name: irrCrop,
        growth_stage: irrStage,
        soil_type: irrSoil,
        farm_area_acres: parseFloat(irrArea) || 4.5,
        current_soil_moisture_pct: parseFloat(irrMoist) || 16.5,
        daily_et0_mm: 5.2,
        pump_rating_hp: 5.0,
        irrigation_method: 'drip',
      };
      const res = await fetch(`${backendUrl}/api/irrigation/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setIrrResult(data);
      else Alert.alert('Irrigation Error', data.detail || 'Could not schedule irrigation.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setIrrLoading(false);
    }
  };

  const [epiPath, setEpiPath] = useState('yellow_rust');
  const [epiOrigin, setEpiOrigin] = useState('Hoshiarpur, Punjab');
  const [epiCases, setEpiCases] = useState('8');
  const [epiLoading, setEpiLoading] = useState(false);
  const [epiResult, setEpiResult] = useState(null);

  const handleRunEpidemic = async () => {
    setEpiLoading(true);
    setEpiResult(null);
    try {
      const payload = {
        pathogen_key: epiPath,
        origin_district: epiOrigin,
        active_focal_cases: parseInt(epiCases, 10) || 8,
        ambient_temp_c: 15.5,
        ambient_humidity_pct: 82.0,
        wind_speed_kmh: 14.0,
        wind_azimuth_degrees: 55.0,
        surrounding_crop_acreage: 5000.0,
      };
      const res = await fetch(`${backendUrl}/api/epidemic/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) setEpiResult(data);
      else Alert.alert('Epidemic Error', data.detail || 'Could not simulate epidemic.');
    } catch {
      Alert.alert('Connection Error', `Failed to reach ${backendUrl}`);
    } finally {
      setEpiLoading(false);
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

  const handleFetchLiveWeather = async (targetDistrict) => {
    const loc = targetDistrict || cropDistrict || 'Ludhiana';
    setLiveWeatherLoading(true);
    try {
      const res = await fetch(`${backendUrl}/api/crop/live-weather-telemetry?location=${encodeURIComponent(loc)}`);
      const data = await res.json();
      if (res.ok) {
        setLiveWeatherTelemetry(data);
        if (data.temperature_c !== undefined) setCTemp(String(data.temperature_c));
        if (data.relative_humidity_pct !== undefined) setCHum(String(data.relative_humidity_pct));
        if (data.rainfall_forecast_7d_mm !== undefined) setCRain(String(data.rainfall_forecast_7d_mm));
        Alert.alert(
          'Live Weather Connected 🌤️',
          `Station: ${data.station_name}, ${data.state}\n` +
          `Temp: ${data.temperature_c}°C | Humidity: ${data.relative_humidity_pct}%\n` +
          `7-Day Rain: ${data.rainfall_forecast_7d_mm} mm | Wind: ${data.wind_speed_kmh} km/h`
        );
      } else {
        Alert.alert('Weather Notice', data.detail || 'Could not fetch live weather.');
      }
    } catch {
      Alert.alert('Network Error', `Cannot reach API at ${backendUrl}`);
    } finally {
      setLiveWeatherLoading(false);
    }
  };

  const handlePredictWithLiveWeather = async () => {
    setCropLoading(true);
    setCropResult(null);
    try {
      const payload = {
        district: cropDistrict || 'Ludhiana',
        ph: parseFloat(cPh) || 6.5,
        nitrogen: parseFloat(cN) || 140.0,
        phosphorus: parseFloat(cP) || 35.0,
        potassium: parseFloat(cK) || 120.0,
        soil_type: cSoil || 'Alluvial Soil',
      };
      const res = await fetch(`${backendUrl}/api/crop/predict-with-live-weather`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setCropResult(data);
        if (data.weather_telemetry) {
          setLiveWeatherTelemetry(data.weather_telemetry);
        }
      } else {
        Alert.alert('Prediction Error', data.detail || 'Could not complete live crop prediction.');
      }
    } catch {
      Alert.alert('Connection Failed', `Cannot reach API at ${backendUrl}`);
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

            <View style={{ flexDirection: 'row', gap: 6, marginBottom: SPACE.md, flexWrap: 'wrap' }}>
              {HOST_PRESETS.map((hp) => (
                <TouchableOpacity
                  key={hp.url}
                  style={[
                    styles.cropChip,
                    backendUrl === hp.url && styles.cropChipActive,
                    { paddingVertical: 6, paddingHorizontal: 10, marginBottom: 0 }
                  ]}
                  onPress={() => {
                    setBackendUrl(hp.url);
                  }}
                >
                  <Text style={[styles.cropChipText, backendUrl === hp.url && styles.cropChipTextActive, { fontSize: 11 }]}>
                    {hp.label}
                  </Text>
                </TouchableOpacity>
              ))}
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
                  <TouchableOpacity
                    style={[styles.actionBtn, styles.actionBtnWarning]}
                    onPress={handleQuickReportFromDiagnosis}
                    accessibilityRole="button"
                    accessibilityLabel="Report diagnosis to community outbreak radar"
                  >
                    <Text style={styles.actionBtnTextWarning}>📢 Alert Radar</Text>
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

              {/* Live Weather Station Selection */}
              <View style={styles.formSection}>
                <Text style={styles.formSectionTitle}>🛰️ Real-Time Weather Station (Open-Meteo NWP)</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetsRow}>
                  {['Ludhiana', 'Karnal', 'Nagpur', 'Rajkot', 'Davanagere', 'Patna', 'Varanasi', 'Guntur', 'Indore'].map((dst) => (
                    <TouchableOpacity
                      key={dst}
                      style={[styles.presetChip, cropDistrict.toLowerCase().includes(dst.toLowerCase()) && styles.cropChipActive]}
                      onPress={() => {
                        setCropDistrict(dst);
                        handleFetchLiveWeather(dst);
                      }}
                    >
                      <Text style={[styles.presetChipText, cropDistrict.toLowerCase().includes(dst.toLowerCase()) && { color: COLORS.white, fontWeight: '800' }]}>
                        📍 {dst}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
                <View style={styles.gridRow}>
                  <Field label="Target District / City" value={cropDistrict} onChangeText={setCropDistrict} numeric={false} />
                </View>
                <TouchableOpacity
                  style={[styles.secondaryButton, { marginTop: 4, marginBottom: 8 }, liveWeatherLoading && styles.buttonDisabled]}
                  onPress={() => handleFetchLiveWeather(cropDistrict)}
                  disabled={liveWeatherLoading}
                >
                  {liveWeatherLoading ? (
                    <ActivityIndicator color={COLORS.forest} />
                  ) : (
                    <Text style={styles.secondaryButtonText}>🌤️ Fetch Live Weather Telemetry</Text>
                  )}
                </TouchableOpacity>

                {liveWeatherTelemetry && (
                  <View style={[styles.statusCard, { borderColor: '#38bdf8', borderWidth: 1.5, backgroundColor: '#f0f9ff', paddingVertical: 10 }]}>
                    <View style={{ flex: 1 }}>
                      <Text style={{ fontSize: 10.5, fontWeight: '800', color: '#0369a1', letterSpacing: 0.5 }}>LIVE NWP TELEMETRY CONNECTED</Text>
                      <Text style={{ fontSize: 14, fontWeight: '800', color: '#0f172a', marginVertical: 2 }}>
                        📍 {liveWeatherTelemetry.station_name}, {liveWeatherTelemetry.state}
                      </Text>
                      <Text style={{ fontSize: 12, color: '#334155' }}>
                        🌡️ {liveWeatherTelemetry.temperature_c}°C | 💧 {liveWeatherTelemetry.relative_humidity_pct}% RH | 🌧️ {liveWeatherTelemetry.rainfall_forecast_7d_mm} mm
                      </Text>
                      <Text style={{ fontSize: 11, color: '#0284c7', marginTop: 2 }}>
                        Radar: {liveWeatherTelemetry.source} ({liveWeatherTelemetry.weather_description})
                      </Text>
                    </View>
                  </View>
                )}
              </View>

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
                style={[styles.primaryButton, cropLoading && styles.buttonDisabled, { marginBottom: 10 }]}
                onPress={handlePredictWithLiveWeather}
                disabled={cropLoading}
                accessibilityRole="button"
                accessibilityState={{ disabled: cropLoading, busy: cropLoading }}
              >
                {cropLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🚀 Predict with Live Weather Telemetry</Text>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.secondaryButton, cropLoading && styles.buttonDisabled]}
                onPress={handleRecommendCrops}
                disabled={cropLoading}
              >
                <Text style={styles.secondaryButtonText}>🌾 Standard Agronomic Calculation</Text>
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
                    {cand.weather_rationale && (
                      <View style={{ backgroundColor: '#e0f2fe', borderRadius: RADIUS.sm, padding: 6, marginVertical: 4 }}>
                        <Text style={{ fontSize: 11.5, color: '#0369a1', fontWeight: '700' }}>
                          🌤️ Live Weather: {cand.weather_rationale}
                        </Text>
                        {cand.thermal_factor && (
                          <Text style={{ fontSize: 10.5, color: '#0284c7', marginTop: 2 }}>
                            Thermal factor: {cand.thermal_factor}x | Moisture: {cand.moisture_factor}x
                          </Text>
                        )}
                      </View>
                    )}
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
        {/* TAB: ECONOMIC ROI & ACTION ADVISOR                        */}
        {/* ========================================================= */}
        {activeTab === 'advisor' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>💰 Economic Profit & Action Advisor</Text>
              <Text style={styles.cardDesc}>
                Calculate fertilizer dose, input costs, MSP mandi revenue, and projected net profit margin per acre.
              </Text>

              <Text style={styles.fieldSectionTitle}>Select Target Crop:</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.presetsRow}>
                {ADVISOR_CROPS.map((c) => (
                  <TouchableOpacity
                    key={c}
                    style={[styles.cropChip, advCrop === c && styles.cropChipActive]}
                    onPress={() => {
                      setAdvCrop(c);
                      handleCalculateEconomics(c);
                    }}
                  >
                    <Text style={[styles.cropChipText, advCrop === c && styles.cropChipTextActive]}>{c}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>

              <View style={styles.fieldRow}>
                <Field label="Land Area" unit="Acres" value={advArea} onChangeText={setAdvArea} />
                <Field label="State / Region" unit="" value={advState} onChangeText={setAdvState} numeric={false} />
              </View>

              <Text style={styles.fieldSectionTitle}>Soil Test Values (N-P-K kg/ha):</Text>
              <View style={styles.fieldRow}>
                <Field label="Nitrogen (N)" unit="kg" value={advN} onChangeText={setAdvN} />
                <Field label="Phosphorus (P)" unit="kg" value={advP} onChangeText={setAdvP} />
                <Field label="Potassium (K)" unit="kg" value={advK} onChangeText={setAdvK} />
              </View>

              <View style={{ marginTop: SPACE.sm, gap: SPACE.sm }}>
                <TouchableOpacity
                  style={[styles.primaryButton, advLoading && styles.buttonDisabled]}
                  onPress={() => handleCalculateEconomics()}
                  disabled={advLoading}
                >
                  {advLoading ? (
                    <ActivityIndicator color="#ffffff" />
                  ) : (
                    <Text style={styles.primaryButtonText}>📊 Calculate Cost & Profit</Text>
                  )}
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.secondaryButton, advCompareLoading && styles.buttonDisabled]}
                  onPress={handleCompareCrops}
                  disabled={advCompareLoading}
                >
                  {advCompareLoading ? (
                    <ActivityIndicator color={COLORS.forest} />
                  ) : (
                    <Text style={styles.secondaryButtonText}>⚖️ Compare All Crops by Net Profit</Text>
                  )}
                </TouchableOpacity>
              </View>
            </View>

            {/* Advisor Result Card */}
            {advResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>{advResult.crop} Economics</Text>
                  <View style={[styles.statePill, { backgroundColor: '#dcfce7' }]}>
                    <Text style={[styles.statePillText, { color: '#166534' }]}>
                      {advResult.profit_margin_pct}% Margin
                    </Text>
                  </View>
                </View>

                {/* Key Metrics */}
                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>₹{Math.round(advResult.net_profit_margin_inr || 0).toLocaleString()}</Text>
                    <Text style={styles.statLabel}>Net Est. Profit</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: COLORS.forest }]}>₹{Math.round(advResult.projected_revenue_inr || 0).toLocaleString()}</Text>
                    <Text style={styles.statLabel}>Gross Revenue</Text>
                  </View>
                </View>

                <View style={styles.tableCard}>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Benchmark Yield</Text>
                    <Text style={styles.tableVal}>{advResult.projected_yield_quintals} Quintals ({advResult.yield_per_acre_quintals} Q/acre)</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Reference Rate / MSP</Text>
                    <Text style={styles.tableVal}>₹{advResult.msp_or_market_rate_per_quintal} / Quintal</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Total Input & Cultivation Cost</Text>
                    <Text style={[styles.tableVal, { color: '#b91c1c' }]}>₹{Math.round(advResult.total_estimated_cost_inr || 0).toLocaleString()}</Text>
                  </View>
                  <View style={[styles.tableRow, { borderBottomWidth: 0, paddingTop: 6 }]}>
                    <Text style={[styles.tableKey, { fontWeight: '800' }]}>Net Profit After All Expenses</Text>
                    <Text style={styles.tableValProfit}>₹{Math.round(advResult.net_profit_margin_inr || 0).toLocaleString()}</Text>
                  </View>
                </View>

                {advResult.fertilizer_plan && (
                  <View style={{ marginTop: SPACE.sm }}>
                    <Text style={styles.fieldSectionTitle}>Prescribed Fertilizer Schedule:</Text>
                    <View style={styles.tableCard}>
                      <View style={styles.tableRow}>
                        <Text style={styles.tableKey}>Urea (46% N)</Text>
                        <Text style={styles.tableVal}>{advResult.fertilizer_plan.urea_bags_50kg} bags (₹{advResult.fertilizer_plan.urea_cost_inr})</Text>
                      </View>
                      <View style={styles.tableRow}>
                        <Text style={styles.tableKey}>DAP (18-46-0)</Text>
                        <Text style={styles.tableVal}>{advResult.fertilizer_plan.dap_bags_50kg} bags (₹{advResult.fertilizer_plan.dap_cost_inr})</Text>
                      </View>
                      <View style={styles.tableRow}>
                        <Text style={styles.tableKey}>MOP (Potash)</Text>
                        <Text style={styles.tableVal}>{advResult.fertilizer_plan.mop_bags_50kg} bags (₹{advResult.fertilizer_plan.mop_cost_inr})</Text>
                      </View>
                      <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                        <Text style={[styles.tableKey, { fontWeight: '700' }]}>Total Fertilizer Spend</Text>
                        <Text style={[styles.tableVal, { fontWeight: '800', color: COLORS.forest }]}>₹{advResult.fertilizer_plan.total_fertilizer_cost_inr}</Text>
                      </View>
                    </View>
                  </View>
                )}
              </View>
            )}

            {/* Compare Ranking View */}
            {advCompareList && (
              <View style={styles.cardResult}>
                <Text style={styles.conditionText}>🏆 Crop Profitability Ranking ({advArea} Acres)</Text>
                <Text style={styles.cardDesc}>Ranked from highest to lowest net margin after all input costs:</Text>
                {advCompareList.slice(0, 5).map((item, idx) => (
                  <View key={idx} style={[styles.tableCard, idx === 0 && { borderColor: COLORS.leaf, borderWidth: 2 }]}>
                    <View style={styles.rowBetween}>
                      <Text style={[styles.bold, { fontSize: 16, color: COLORS.forest }]}>
                        #{idx + 1} {item.crop}
                      </Text>
                      <View style={[styles.statePill, { backgroundColor: '#dcfce7' }]}>
                        <Text style={[styles.statePillText, { color: '#166534' }]}>{item.profit_margin_pct}% margin</Text>
                      </View>
                    </View>
                    <View style={[styles.tableRow, { marginTop: 6 }]}>
                      <Text style={styles.tableKey}>Net Profit:</Text>
                      <Text style={styles.tableValProfit}>₹{Math.round(item.net_profit_margin_inr || 0).toLocaleString()}</Text>
                    </View>
                    <View style={styles.tableRow}>
                      <Text style={styles.tableKey}>Est. Revenue:</Text>
                      <Text style={styles.tableVal}>₹{Math.round(item.projected_revenue_inr || 0).toLocaleString()}</Text>
                    </View>
                    <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                      <Text style={styles.tableKey}>Input Cost:</Text>
                      <Text style={styles.tableVal}>₹{Math.round(item.total_estimated_cost_inr || 0).toLocaleString()}</Text>
                    </View>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: COMMUNITY OUTBREAK RADAR                             */}
        {/* ========================================================= */}
        {activeTab === 'radar' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🛰️ Community Outbreak Radar</Text>
              <Text style={styles.cardDesc}>
                Real-time district epidemic tracking. Anonymously aggregated alerts warn neighboring farms before pathogens spread.
              </Text>

              <View style={{ flexDirection: 'row', gap: SPACE.sm, marginVertical: SPACE.xs }}>
                <TouchableOpacity
                  style={[styles.statusCardBtn, { flex: 1, backgroundColor: COLORS.mint }]}
                  onPress={fetchRadarData}
                >
                  <Text style={[styles.statusCardBtnText, { textAlign: 'center' }]}>🔄 Refresh Radar</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.statusCardBtn, { flex: 1, backgroundColor: '#fef3c7', borderColor: '#d97706' }]}
                  onPress={handleSeedRadarDemo}
                >
                  <Text style={[styles.statusCardBtnText, { color: '#b45309', textAlign: 'center' }]}>🌱 Demo Outbreaks</Text>
                </TouchableOpacity>
              </View>

              {/* Stats overview */}
              <View style={styles.statRow}>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>{radarOutbreaks.length}</Text>
                  <Text style={styles.statLabel}>Active Hotspots</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={[styles.statValue, { color: '#dc2626' }]}>
                    {radarOutbreaks.filter((o) => (o.alert_level || o.risk_level) === 'ALERT').length}
                  </Text>
                  <Text style={styles.statLabel}>High Alert Zones</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={[styles.statValue, { color: COLORS.leaf }]}>
                    {radarDistricts.length || 10}
                  </Text>
                  <Text style={styles.statLabel}>Monitored Districts</Text>
                </View>
              </View>
            </View>

            {/* Outbreaks List */}
            <Text style={styles.sectionHeading}>🚨 Recent Field Alerts & Hotspots</Text>
            {radarLoading && <ActivityIndicator color={COLORS.leaf} style={{ marginVertical: 20 }} />}
            
            {!radarLoading && radarOutbreaks.length === 0 && (
              <View style={styles.emptyState}>
                <Text style={styles.emptyIcon}>🛡️</Text>
                <Text style={styles.emptyTitle}>No Active Outbreaks Reported</Text>
                <Text style={styles.emptyText}>Tap 'Demo Outbreaks' above to preview simulated regional alerts during judging.</Text>
              </View>
            )}

            {radarOutbreaks.map((item, idx) => {
              const level = item.alert_level || item.risk_level || (item.case_count >= 3 ? 'ALERT' : 'WATCH');
              const isAlert = level === 'ALERT';
              return (
                <View key={idx} style={[styles.historyCard, { borderLeftColor: isAlert ? '#dc2626' : '#d97706' }]}>
                  <View style={{ padding: SPACE.md }}>
                    <View style={styles.rowBetween}>
                      <Text style={[styles.bold, { fontSize: 16, color: COLORS.text }]}>
                        {item.condition || item.disease}
                      </Text>
                      <View style={[styles.statePill, isAlert ? styles.alertBadgeAlert : styles.alertBadgeWatch]}>
                        <Text style={isAlert ? styles.alertBadgeTextAlert : styles.alertBadgeTextWatch}>
                          {isAlert ? '🔴 CRITICAL ALERT' : '🟡 WATCH'}
                        </Text>
                      </View>
                    </View>

                    <Text style={{ fontSize: 13, color: COLORS.forest, fontWeight: '700', marginTop: 4 }}>
                      📍 {item.district} · {item.crop || 'Field Crop'}
                    </Text>
                    <Text style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 2 }}>
                      Reported incidents: {item.report_count || item.case_count || 1} in the last 14 days
                    </Text>

                    {item.recommended_action && (
                      <View style={{ backgroundColor: COLORS.mint, padding: 8, borderRadius: RADIUS.sm, marginTop: 8 }}>
                        <Text style={{ fontSize: 12, color: '#166534', fontWeight: '600' }}>
                          🛡️ Advisory: {item.recommended_action}
                        </Text>
                      </View>
                    )}
                  </View>
                </View>
              );
            })}

            {/* Manual Report Outbreak Form */}
            <View style={[styles.card, { marginTop: SPACE.lg }]}>
              <Text style={styles.cardTitle}>📢 Report Outbreak in Your Area</Text>
              <Text style={styles.cardDesc}>
                Seen symptoms in neighboring plots? Submit an anonymous alert to warn fellow farmers.
              </Text>
              <Field label="Observed Disease / Symptom" unit="" value={reportCondition} onChangeText={setReportCondition} numeric={false} />
              <View style={styles.fieldRow}>
                <Field label="Crop Name" unit="" value={reportCrop} onChangeText={setReportCrop} numeric={false} />
                <Field label="District, State" unit="" value={reportDistrict} onChangeText={setReportDistrict} numeric={false} />
              </View>
              <TouchableOpacity
                style={[styles.primaryButton, submittingReport && styles.buttonDisabled]}
                onPress={handleSubmitManualReport}
                disabled={submittingReport}
              >
                {submittingReport ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🚀 Submit Anonymous Alert</Text>
                )}
              </TouchableOpacity>
            </View>
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
                <Text style={styles.fieldSectionTitle}>🇮🇳 Quick Voice & Agronomy Questions:</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginBottom: 10 }}>
                  {MULTILINGUAL_VOICE_PROMPTS.map((p, idx) => (
                    <TouchableOpacity
                      key={idx}
                      style={[styles.suggestChip, { backgroundColor: '#f0fdf4', borderColor: '#86efac' }]}
                      onPress={() => sendChat(p.query)}
                    >
                      <Text style={[styles.suggestText, { color: '#166534', fontWeight: '700' }]}>{p.label}</Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
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
                  {m.role === 'assistant' && (
                    <TouchableOpacity
                      style={{ marginTop: 8, alignSelf: 'flex-start', paddingVertical: 4, paddingHorizontal: 10, backgroundColor: 'rgba(22, 163, 74, 0.12)', borderRadius: 12 }}
                      onPress={() => Alert.alert('🔊 Spoken Advisory', `Reading aloud in ${chatLanguage}:\n\n${m.text}`)}
                    >
                      <Text style={{ fontSize: 11, color: '#166534', fontWeight: '800' }}>🔊 Read Aloud</Text>
                    </TouchableOpacity>
                  )}
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
        {/* TAB: SYNTHETIC MULTISPECTRAL NDVI STUDIO                  */}
        {/* ========================================================= */}
        {activeTab === 'ndvi' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🔬 Synthetic Multispectral NDVI</Text>
              <Text style={styles.cardDesc}>
                Physics-informed Near-Infrared radiance reconstruction detects sub-visual cellular chlorophyll stress 7-10 days before lesions appear.
              </Text>
              <TouchableOpacity
                style={[styles.primaryButton, ndviLoading && styles.buttonDisabled]}
                onPress={handleRunNdvi}
                disabled={ndviLoading}
              >
                {ndviLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🛰️ Reconstruct 780nm NDVI Spectrum</Text>
                )}
              </TouchableOpacity>
              <Text style={styles.configHelp}>
                Uses leaf photo from Doctor tab or pick photo below:
              </Text>
              <TouchableOpacity style={styles.secondaryButton} onPress={() => pickPhoto('plant_leaf')}>
                <Text style={styles.secondaryButtonText}>🖼️ Choose Leaf Photo</Text>
              </TouchableOpacity>
            </View>

            {ndviResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>{ndviResult.health_status}</Text>
                  <View style={[styles.statePill, ndviResult.risk_level === 'LOW' ? styles.alertBadgeLow : styles.alertBadgeAlert]}>
                    <Text style={ndviResult.risk_level === 'LOW' ? styles.alertBadgeTextLow : styles.alertBadgeTextAlert}>
                      {ndviResult.risk_level} RISK
                    </Text>
                  </View>
                </View>
                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{ndviResult.metrics.mean_ndvi}</Text>
                    <Text style={styles.statLabel}>Mean NDVI</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{ndviResult.metrics.mean_gndvi}</Text>
                    <Text style={styles.statLabel}>Green NDVI</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: COLORS.leaf }]}>{ndviResult.metrics.synthetic_nir_reflectance_pct}%</Text>
                    <Text style={styles.statLabel}>NIR Reflectance</Text>
                  </View>
                </View>
                <View style={styles.tableCard}>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Vigorous Canopy</Text>
                    <Text style={[styles.tableVal, { color: '#16a34a' }]}>{ndviResult.metrics.vigorous_canopy_pct}%</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Early Sub-Visual Stress</Text>
                    <Text style={[styles.tableVal, { color: '#ca8a04' }]}>{ndviResult.metrics.early_stress_pct}%</Text>
                  </View>
                  <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                    <Text style={styles.tableKey}>Severe Necrosis</Text>
                    <Text style={[styles.tableVal, { color: '#dc2626' }]}>{ndviResult.metrics.severe_necrosis_pct}%</Text>
                  </View>
                </View>
                <Text style={[styles.cropDetailLine, { marginTop: 8 }]}>
                  <Text style={styles.bold}>Advisory:</Text> {ndviResult.advisory}
                </Text>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: DRONE FIELD SCANNER                                  */}
        {/* ========================================================= */}
        {activeTab === 'drone' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🚁 Drone Farm Flyover & AR HUD</Text>
              <Text style={styles.cardDesc}>
                Autonomous lawnmower flight sweep over your acreage with persistent centroid cluster tracking and rolling crop vigor monitoring.
              </Text>
              <TouchableOpacity
                style={[styles.primaryButton, droneLoading && styles.buttonDisabled]}
                onPress={handleRunDrone}
                disabled={droneLoading}
              >
                {droneLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🚀 Execute 5-Acre Drone Sweep</Text>
                )}
              </TouchableOpacity>
            </View>

            {droneResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>Flight Audit: {droneResult.crop_type}</Text>
                  <View style={[styles.statePill, { backgroundColor: '#dcfce7' }]}>
                    <Text style={[styles.statePillText, { color: '#166534' }]}>{droneResult.field_summary.field_rating}</Text>
                  </View>
                </View>
                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{droneResult.field_summary.average_health_score}/100</Text>
                    <Text style={styles.statLabel}>Avg Field Vigor</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#dc2626' }]}>{droneResult.field_summary.unique_clusters_tracked}</Text>
                    <Text style={styles.statLabel}>Tracked Clusters</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{droneResult.field_summary.affected_area_est_acres} ac</Text>
                    <Text style={styles.statLabel}>Affected Area</Text>
                  </View>
                </View>
                <Text style={styles.fieldSectionTitle}>Persistent Track IDs Detected:</Text>
                {droneResult.clusters_catalog.map((c, idx) => (
                  <View key={idx} style={styles.tableCard}>
                    <View style={styles.rowBetween}>
                      <Text style={[styles.bold, { color: COLORS.forest }]}>🎯 {c.id}: {c.name}</Text>
                      <Text style={{ fontSize: 12, fontWeight: '700', color: '#b91c1c' }}>{c.severity} Severity</Text>
                    </View>
                    <Text style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 4 }}>
                      Ground Relative Position: ({Math.round(c.x_norm * 100)}%, {Math.round(c.y_norm * 100)}%) · Conf: {Math.round(c.conf * 100)}%
                    </Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: AI AGRONOMIC COUNCIL                                 */}
        {/* ========================================================= */}
        {activeTab === 'council' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>⚖️ Multi-Agent Agronomic Debate Council</Text>
              <Text style={styles.cardDesc}>
                Self-reflective debate council: Plant Pathologist vs Soil Microbiologist vs Economic Actuary with final Arbiter Consensus.
              </Text>
              <TouchableOpacity
                style={[styles.primaryButton, councilLoading && styles.buttonDisabled]}
                onPress={handleRunCouncil}
                disabled={councilLoading}
              >
                {councilLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>⚖️ Convene Multi-Agent Debate</Text>
                )}
              </TouchableOpacity>
            </View>

            {councilResult && (
              <View>
                {councilResult.council_agents.map((ag, idx) => (
                  <View key={idx} style={[styles.historyCard, { borderLeftColor: ag.color, padding: 12 }]}>
                    <View style={styles.rowBetween}>
                      <Text style={[styles.bold, { color: COLORS.text }]}>{ag.avatar} {ag.role}</Text>
                      <Text style={{ fontSize: 11, fontWeight: '800', color: ag.color }}>{ag.stance}</Text>
                    </View>
                    <Text style={{ fontSize: 13, color: COLORS.text, marginTop: 6, lineHeight: 18 }}>{ag.statement}</Text>
                  </View>
                ))}

                <View style={[styles.cardResult, { backgroundColor: '#f0fdf4', borderColor: '#16a34a', borderWidth: 2 }]}>
                  <View style={styles.rowBetween}>
                    <Text style={[styles.bold, { color: '#14532d', fontSize: 16 }]}>
                      {councilResult.consensus_verdict.arbiter_avatar} {councilResult.consensus_verdict.arbiter_role}
                    </Text>
                    <View style={[styles.statePill, { backgroundColor: '#bbf7d0' }]}>
                      <Text style={{ color: '#14532d', fontWeight: '800', fontSize: 12 }}>
                        {councilResult.consensus_verdict.safety_integrity_score}
                      </Text>
                    </View>
                  </View>
                  <Text style={[styles.bold, { color: '#15803d', marginTop: 6 }]}>
                    Strategy: {councilResult.consensus_verdict.compromise_strategy}
                  </Text>
                  <Text style={{ fontSize: 14, color: COLORS.text, marginTop: 6, lineHeight: 20 }}>
                    {councilResult.consensus_verdict.ruling}
                  </Text>
                  <View style={[styles.tableRow, { marginTop: 10, borderBottomWidth: 0 }]}>
                    <Text style={{ fontSize: 11, fontFamily: 'monospace', color: '#16a34a' }}>
                      TOKEN: {councilResult.consensus_verdict.cryptographic_verification_token}
                    </Text>
                  </View>
                </View>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: ZERO-INTERNET EDGE ONNX                              */}
        {/* ========================================================= */}
        {activeTab === 'edge' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>⚡ Zero-Internet Edge ONNX INT8</Text>
              <Text style={styles.cardDesc}>
                100% Offline execution on rural mobile edge. Bypasses all cloud networks with sub-25ms CPU SIMD inference.
              </Text>
              <View style={[styles.statePill, { backgroundColor: '#fef3c7', alignSelf: 'flex-start', marginVertical: 8 }]}>
                <Text style={{ color: '#b45309', fontWeight: '800', fontSize: 12 }}>✈️ Airplane Mode / Air-Gapped Ready</Text>
              </View>
              <TouchableOpacity
                style={[styles.primaryButton, edgeLoading && styles.buttonDisabled]}
                onPress={handleRunEdge}
                disabled={edgeLoading}
              >
                {edgeLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>⚡ Execute Sub-25ms Local Edge ONNX</Text>
                )}
              </TouchableOpacity>
              <Text style={styles.configHelp}>
                Requires a leaf photo from Doctor tab or pick one:
              </Text>
              <TouchableOpacity style={styles.secondaryButton} onPress={() => pickPhoto('plant_leaf')}>
                <Text style={styles.secondaryButtonText}>🖼️ Pick Photo for Edge Test</Text>
              </TouchableOpacity>
            </View>

            {edgeResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>
                    {edgeResult.diagnosis.crop} · {edgeResult.diagnosis.condition}
                  </Text>
                  <View style={[styles.statePill, { backgroundColor: '#dcfce7' }]}>
                    <Text style={[styles.statePillText, { color: '#166534' }]}>{edgeResult.diagnosis.confidence_pct}% Conf</Text>
                  </View>
                </View>
                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{edgeResult.telemetry.edge_latency_ms} ms</Text>
                    <Text style={styles.statLabel}>Local Latency</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: COLORS.leaf }]}>{edgeResult.telemetry.speedup_vs_cloud}</Text>
                    <Text style={styles.statLabel}>Speedup vs Cloud</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{edgeResult.telemetry.model_size_mb} MB</Text>
                    <Text style={styles.statLabel}>INT8 Quantized</Text>
                  </View>
                </View>
                <View style={styles.tableCard}>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Runtime Engine</Text>
                    <Text style={styles.tableVal}>{edgeResult.runtime_engine}</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Network Packets</Text>
                    <Text style={[styles.tableVal, { color: '#16a34a' }]}>0 bytes (Pure Offline)</Text>
                  </View>
                  <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                    <Text style={styles.tableKey}>Memory Compression</Text>
                    <Text style={styles.tableVal}>{edgeResult.telemetry.memory_saved_pct} Saved</Text>
                  </View>
                </View>
              </View>
            )}
          </View>
        {/* ========================================================= */}
        {/* TAB: SATELLITE DROUGHT PROFILER                           */}
        {/* ========================================================= */}
        {activeTab === 'satellite' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🌊 Satellite Hydrology & Drought Profiler</Text>
              <Text style={styles.cardDesc}>
                Synthetic NDWI moisture matrices, FAO-56 Penman-Monteith ET0, and root-zone water balance to prescribe exact drip irrigation.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="Cultivated Crop" unit="" value={satCrop} onChangeText={setSatCrop} numeric={false} />
                <Field label="Soil Taxonomy" unit="" value={satSoil} onChangeText={setSatSoil} numeric={false} />
              </View>
              <View style={styles.fieldRow}>
                <Field label="Field Area" unit="Acres" value={satArea} onChangeText={setSatArea} />
                <Field label="14-Day Rain" unit="mm" value={satRain} onChangeText={setSatRain} />
              </View>
              <View style={styles.fieldRow}>
                <Field label="Temperature" unit="°C" value={satTemp} onChangeText={setSatTemp} />
                <Field label="Humidity" unit="%" value={satHum} onChangeText={setSatHum} />
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, satLoading && styles.buttonDisabled]}
                onPress={() => handleRunSatellite()}
                disabled={satLoading}
              >
                {satLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🛰️ Run Drought Risk Assessment</Text>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={() => handleRunSatellite('Wheat')}
              >
                <Text style={styles.secondaryButtonText}>⚡ 1-Tap Wheat Drought Audit</Text>
              </TouchableOpacity>
            </View>

            {satResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>{satResult.crop} Hydrology</Text>
                  <View style={[styles.statePill, { backgroundColor: '#fee2e2' }]}>
                    <Text style={{ color: '#991b1b', fontWeight: '800', fontSize: 12 }}>
                      {satResult.hydrology_metrics.drought_badge}
                    </Text>
                  </View>
                </View>

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#dc2626' }]}>
                      {satResult.hydrology_metrics.drought_risk_index_score}/100
                    </Text>
                    <Text style={styles.statLabel}>Drought Risk</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: COLORS.forest }]}>
                      {satResult.hydrology_metrics.root_zone_soil_moisture_pct}%
                    </Text>
                    <Text style={styles.statLabel}>Root Moisture</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>
                      {satResult.hydrology_metrics.water_deficit_14d_mm} mm
                    </Text>
                    <Text style={styles.statLabel}>14d Deficit</Text>
                  </View>
                </View>

                <Text style={styles.fieldSectionTitle}>Dynamic Drip Irrigation Prescription:</Text>
                <View style={styles.tableCard}>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Target Water Depth</Text>
                    <Text style={styles.tableVal}>{satResult.irrigation_prescription.recommended_water_depth_mm} mm</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Volume / Acre</Text>
                    <Text style={styles.tableVal}>{satResult.irrigation_prescription.water_volume_liters_per_acre.toLocaleString()} L</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Total Field Need</Text>
                    <Text style={[styles.tableVal, { color: '#16a34a' }]}>{satResult.irrigation_prescription.total_field_volume_liters.toLocaleString()} L</Text>
                  </View>
                  <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                    <Text style={styles.tableKey}>Recommended Drip Run Time</Text>
                    <Text style={styles.tableValProfit}>{satResult.irrigation_prescription.recommended_drip_duration_hours} Hours</Text>
                  </View>
                </View>
                <Text style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 4 }}>
                  💡 {satResult.irrigation_prescription.recommended_timing}
                </Text>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: SMART SPRAY TANK-MIX COMPATIBILITY                   */}
        {/* ========================================================= */}
        {activeTab === 'tankmix' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🧪 Smart Spray Tank-Mix & Drift Studio</Text>
              <Text style={styles.cardDesc}>
                Prevent tank flocculation, chemical burns, biocide inactivation, and off-target spray drift.
              </Text>

              <Text style={styles.inputLabel}>Product A (Fungicide / Insecticide):</Text>
              <TextInput style={styles.textInput} value={tmChemA} onChangeText={setTmChemA} />

              <Text style={[styles.inputLabel, { marginTop: 10 }]}>Product B (Partner in Tank):</Text>
              <TextInput style={styles.textInput} value={tmChemB} onChangeText={setTmChemB} />

              <View style={styles.fieldRow}>
                <Field label="Water pH" unit="" value={tmPh} onChangeText={setTmPh} />
                <Field label="Wind Speed" unit="km/h" value={tmWind} onChangeText={setTmWind} />
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, tmLoading && styles.buttonDisabled]}
                onPress={() => handleRunTankMix()}
                disabled={tmLoading}
              >
                {tmLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🧪 Check Tank Compatibility</Text>
                )}
              </TouchableOpacity>

              <View style={{ flexDirection: 'row', gap: 8, marginTop: 8 }}>
                <TouchableOpacity
                  style={[styles.secondaryButton, { flex: 1, backgroundColor: '#fef2f2', borderColor: '#ef4444' }]}
                  onPress={() => {
                    setTmChemA('Copper Oxychloride (Fungicide)');
                    setTmChemB('Trichoderma viride (Bio-agent)');
                    handleRunTankMix('Copper Oxychloride (Fungicide)', 'Trichoderma viride (Bio-agent)');
                  }}
                >
                  <Text style={[styles.secondaryButtonText, { color: '#b91c1c' }]}>🚨 Biocide Conflict</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.secondaryButton, { flex: 1, backgroundColor: '#f0fdf4', borderColor: '#16a34a' }]}
                  onPress={() => {
                    setTmChemA('Mancozeb 75% WP (Fungicide)');
                    setTmChemB('Imidacloprid 17.8% SL (Insecticide)');
                    handleRunTankMix('Mancozeb 75% WP (Fungicide)', 'Imidacloprid 17.8% SL (Insecticide)');
                  }}
                >
                  <Text style={[styles.secondaryButtonText, { color: '#166534' }]}>✅ Safe Tank Mix</Text>
                </TouchableOpacity>
              </View>
            </View>

            {tmResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>
                    {tmResult.tank_compatibility.overall_safety_rating}
                  </Text>
                  <View
                    style={[
                      styles.statePill,
                      {
                        backgroundColor: tmResult.tank_compatibility.is_compatible ? '#dcfce7' : '#fee2e2',
                      },
                    ]}
                  >
                    <Text
                      style={{
                        color: tmResult.tank_compatibility.is_compatible ? '#166534' : '#991b1b',
                        fontWeight: '800',
                        fontSize: 12,
                      }}
                    >
                      {tmResult.tank_compatibility.safety_badge}
                    </Text>
                  </View>
                </View>

                {tmResult.tank_compatibility.conflicts && tmResult.tank_compatibility.conflicts.map((c, idx) => (
                  <View key={idx} style={[styles.tableCard, { backgroundColor: '#fef2f2', borderColor: '#fca5a5', marginTop: 8 }]}>
                    <Text style={{ color: '#b91c1c', fontWeight: '700', fontSize: 13 }}>⛔ {c}</Text>
                  </View>
                ))}

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#d97706' }]}>
                      {tmResult.spray_physics_and_drift.drift_risk_pct}%
                    </Text>
                    <Text style={styles.statLabel}>Drift Danger</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>
                      {tmResult.compliance_and_safety.mandatory_phi_days} d
                    </Text>
                    <Text style={styles.statLabel}>Harvest PHI</Text>
                  </View>
                </View>

                <View style={styles.tableCard}>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Recommended Nozzle</Text>
                    <Text style={styles.tableVal}>{tmResult.spray_physics_and_drift.recommended_nozzle.split('(')[0]}</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Water pH Advisory</Text>
                    <Text style={[styles.tableVal, { color: COLORS.forest }]}>{tmResult.water_quality_buffering.ph_status}</Text>
                  </View>
                  <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                    <Text style={styles.tableKey}>Spray Window</Text>
                    <Text style={{ fontSize: 12, fontWeight: '700', color: COLORS.text, flex: 1, textAlign: 'right' }}>
                      {tmResult.spray_physics_and_drift.spray_window_advisory}
                    </Text>
                  </View>
                </View>
              </View>
            )}
          </View>
        {/* ========================================================= */}
        {/* TAB: PREDICTIVE CROP YIELD & BIOMASS FORECASTING          */}
        {/* ========================================================= */}
        {activeTab === 'yield' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>📈 Predictive Crop Yield & Biomass</Text>
              <Text style={styles.cardDesc}>
                WOFOST GDD thermal growth model & 1,000-run Monte Carlo simulation for harvest prediction and downside risk.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="Target Crop" unit="" value={yfCrop} onChangeText={setYfCrop} numeric={false} />
                <Field label="Cultivated Area" unit="Acres" value={yfArea} onChangeText={setYfArea} />
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, yfLoading && styles.buttonDisabled]}
                onPress={() => handleRunYield()}
                disabled={yfLoading}
              >
                {yfLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>📈 Run 1,000-Run Yield Forecast</Text>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={() => handleRunYield('Wheat')}
              >
                <Text style={styles.secondaryButtonText}>⚡ 1-Tap Wheat Forecast Demo</Text>
              </TouchableOpacity>
            </View>

            {yfResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>{yfResult.crop} Yield Forecast</Text>
                  <View style={[styles.statePill, { backgroundColor: '#dcfce7' }]}>
                    <Text style={{ color: '#166534', fontWeight: '800', fontSize: 12 }}>
                      {yfResult.yield_forecast.yield_badge}
                    </Text>
                  </View>
                </View>

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: COLORS.forest }]}>
                      {yfResult.yield_forecast.expected_yield_quintals_acre} Q
                    </Text>
                    <Text style={styles.statLabel}>Per Acre</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#15803d' }]}>
                      {yfResult.yield_forecast.total_harvest_expected_quintals} Q
                    </Text>
                    <Text style={styles.statLabel}>Total Field</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#d97706' }]}>
                      {yfResult.monte_carlo_risk_bounds.p10_pessimistic_quintals_acre} Q
                    </Text>
                    <Text style={styles.statLabel}>Downside P10</Text>
                  </View>
                </View>

                <Text style={styles.fieldSectionTitle}>Monte Carlo Value-at-Risk Bounds:</Text>
                <View style={styles.tableCard}>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>P10 (Pessimistic Scenario)</Text>
                    <Text style={styles.tableVal}>{yfResult.monte_carlo_risk_bounds.p10_pessimistic_quintals_acre} Q/Acre</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>P50 (Median Baseline)</Text>
                    <Text style={styles.tableVal}>{yfResult.monte_carlo_risk_bounds.p50_median_quintals_acre} Q/Acre</Text>
                  </View>
                  <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                    <Text style={styles.tableKey}>P90 (Optimistic High)</Text>
                    <Text style={styles.tableValProfit}>{yfResult.monte_carlo_risk_bounds.p90_optimistic_quintals_acre} Q/Acre</Text>
                  </View>
                </View>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: SMART MANDI REAL-TIME ARBITRAGE                      */}
        {/* ========================================================= */}
        {activeTab === 'mandi' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>💰 Smart Mandi Real-Time Arbitrage</Text>
              <Text style={styles.cardDesc}>
                Logistics freight calculator vs APMC Mandi price spreads and warehouse holding returns.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="Crop" unit="" value={maCrop} onChangeText={setMaCrop} numeric={false} />
                <Field label="Harvest Quantity" unit="Quintals" value={maQty} onChangeText={setMaQty} />
              </View>
              <Field label="Current Local Price" unit="₹/q" value={maRate} onChangeText={setMaRate} />

              <TouchableOpacity
                style={[styles.primaryButton, maLoading && styles.buttonDisabled]}
                onPress={() => handleRunMandi()}
                disabled={maLoading}
              >
                {maLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🚚 Calculate Mandi Arbitrage</Text>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={() => handleRunMandi('Wheat')}
              >
                <Text style={styles.secondaryButtonText}>⚡ 1-Tap Mandi Demo</Text>
              </TouchableOpacity>
            </View>

            {maResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>Arbitrage Strategy</Text>
                  <View style={[styles.statePill, { backgroundColor: '#dcfce7' }]}>
                    <Text style={{ color: '#166534', fontWeight: '800', fontSize: 12 }}>
                      {maResult.arbitrage_verdict.badge}
                    </Text>
                  </View>
                </View>

                <View style={[styles.tableCard, { backgroundColor: '#f0fdf4', borderColor: '#86efac', marginTop: 8 }]}>
                  <Text style={{ color: '#166534', fontWeight: '700', fontSize: 13 }}>
                    💡 {maResult.arbitrage_verdict.action_strategy}
                  </Text>
                </View>

                <Text style={styles.fieldSectionTitle}>Top Mandi Hubs (Sorted by Net Payout):</Text>
                {maResult.mandi_arbitrage_matrix.slice(0, 4).map((m, idx) => (
                  <View key={idx} style={styles.tableCard}>
                    <View style={styles.rowBetween}>
                      <Text style={[styles.bold, { color: COLORS.forest }]}>{m.mandi_name}</Text>
                      <Text style={styles.tableValProfit}>₹{m.net_farmer_payout_inr.toLocaleString()}</Text>
                    </View>
                    <Text style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 4 }}>
                      {m.distance_km} km · Rate: ₹{m.spot_rate_per_quintal}/q · Freight: ₹{m.transport_cost_per_quintal}/q
                    </Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: KISAN VAANI VERNACULAR AGRO-VOICE                    */}
        {/* ========================================================= */}
        {activeTab === 'vaani' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🎙️ किसान वाणी (Kisan Vaani)</Text>
              <Text style={styles.cardDesc}>
                ग्रामीण भारत के किसान भाइयों के लिए हिंदी व क्षेत्रीय भाषाओं में प्राकृतिक परामर्श।
              </Text>

              <Text style={styles.inputLabel}>अपना सवाल लिखें या नीचे से चुनें:</Text>
              <TextInput
                style={[styles.textInput, { minHeight: 70, textAlignVertical: 'top' }]}
                multiline
                value={vaaniQuery}
                onChangeText={setVaaniQuery}
              />

              <View style={{ flexDirection: 'row', gap: 6, flexWrap: 'wrap', marginVertical: 8 }}>
                {[
                  { l: '🍂 पीलापन', q: 'पत्तियों में पीलापन आ रहा है क्या खाद डालें' },
                  { l: '🐛 कीट नियंत्रण', q: 'फसल में कीड़ा और इल्ली लग गई है कौन सी दवा डालें' },
                  { l: '🌾 खाद समय', q: 'यूरिया और डीएपी खाद कब और कैसे डालना चाहिए' },
                  { l: '🌧️ बारिश पूर्व छिड़काव', q: 'कल बारिश होने वाली है क्या आज दवा छिड़कें' },
                ].map((item, idx) => (
                  <TouchableOpacity
                    key={idx}
                    style={[styles.cropChip, { paddingVertical: 6, paddingHorizontal: 10, marginBottom: 0 }]}
                    onPress={() => {
                      setVaaniQuery(item.q);
                      handleRunVaani(item.q);
                    }}
                  >
                    <Text style={[styles.cropChipText, { fontSize: 11 }]}>{item.l}</Text>
                  </TouchableOpacity>
                ))}
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, vaaniLoading && styles.buttonDisabled]}
                onPress={() => handleRunVaani()}
                disabled={vaaniLoading}
              >
                {vaaniLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🎙️ किसान वाणी समाधान प्राप्त करें</Text>
                )}
              </TouchableOpacity>
            </View>

            {vaaniResult && (
              <View style={styles.cardResult}>
                <Text style={styles.conditionText}>{vaaniResult.vernacular_card.title_hi}</Text>
                <View style={[styles.tableCard, { backgroundColor: '#f0fdf4', borderColor: '#86efac', marginTop: 8 }]}>
                  <Text style={{ color: '#166534', fontSize: 14, lineHeight: 20 }}>
                    🔊 {vaaniResult.vernacular_card.spoken_summary_hi}
                  </Text>
                </View>

                <Text style={styles.fieldSectionTitle}>चरणबद्ध उपचार कदम (Action Steps):</Text>
                {vaaniResult.vernacular_card.prescribed_steps_hi.map((step, idx) => (
                  <View key={idx} style={[styles.tableCard, { paddingVertical: 8, marginVertical: 3 }]}>
                    <Text style={{ color: COLORS.text, fontSize: 13, fontWeight: '600' }}>{step}</Text>
                  </View>
                ))}

                <View style={[styles.tableCard, { backgroundColor: '#fef3c7', borderColor: '#fcd34d', marginTop: 6 }]}>
                  <Text style={{ color: '#b45309', fontSize: 12, fontWeight: '700' }}>
                    {vaaniResult.vernacular_card.safety_warning_hi}
                  </Text>
                </View>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: PMFBY CROP INSURANCE CLAIM ASSESSOR                  */}
        {/* ========================================================= */}
        {activeTab === 'insurance' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>📜 PMFBY Crop Insurance Assessor</Text>
              <Text style={styles.cardDesc}>
                Pradhan Mantri Fasal Bima Yojana statutory claim shortfall assessment with SHA-256 evidence fingerprint.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="Insured Crop" unit="" value={insCrop} onChangeText={setInsCrop} numeric={false} />
                <Field label="Insured Area" unit="Acres" value={insAcres} onChangeText={setInsAcres} />
              </View>
              <Field label="Post-Calamity Yield" unit="Q/Acre" value={insYield} onChangeText={setInsYield} />

              <TouchableOpacity
                style={[styles.primaryButton, insLoading && styles.buttonDisabled]}
                onPress={() => handleRunInsurance()}
                disabled={insLoading}
              >
                {insLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>📜 Assess PMFBY Insurance Claim</Text>
                )}
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={() => handleRunInsurance('Wheat')}
              >
                <Text style={styles.secondaryButtonText}>⚡ 1-Tap Claim Demo (Wheat Hailstorm)</Text>
              </TouchableOpacity>
            </View>

            {insResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>{insResult.actuarial_loss_assessment.claim_badge}</Text>
                </View>
                <Text style={{ fontSize: 13, color: COLORS.forest, fontWeight: '800', marginTop: 4 }}>
                  Registration ID: {insResult.claim_registration_id}
                </Text>

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#dc2626' }]}>
                      {insResult.actuarial_loss_assessment.crop_shortfall_percentage}%
                    </Text>
                    <Text style={styles.statLabel}>Crop Shortfall</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>
                      ₹{insResult.settlement_financials.total_sum_insured_inr.toLocaleString()}
                    </Text>
                    <Text style={styles.statLabel}>Sum Insured</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#16a34a' }]}>
                      ₹{insResult.settlement_financials.approved_indemnity_claim_payout_inr.toLocaleString()}
                    </Text>
                    <Text style={styles.statLabel}>Approved Claim</Text>
                  </View>
                </View>

                <View style={styles.tableCard}>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>7-Yr District Average</Text>
                    <Text style={styles.tableVal}>{insResult.actuarial_loss_assessment['7yr_historical_average_yield_q_acre'] || 21.0} Q/Acre</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Statutory Threshold Yield</Text>
                    <Text style={styles.tableVal}>{insResult.actuarial_loss_assessment.statutory_threshold_yield_q_acre} Q/Acre</Text>
                  </View>
                  <View style={styles.tableRow}>
                    <Text style={styles.tableKey}>Assessed Post-Calamity Yield</Text>
                    <Text style={[styles.tableVal, { color: '#b91c1c' }]}>{insResult.actuarial_loss_assessment.actual_assessed_post_disaster_yield_q_acre} Q/Acre</Text>
                  </View>
                  <View style={[styles.tableRow, { borderBottomWidth: 0 }]}>
                    <Text style={styles.tableKey}>Farmer Paid Premium</Text>
                    <Text style={styles.tableVal}>₹{insResult.settlement_financials.farmer_premium_already_paid_inr.toLocaleString()}</Text>
                  </View>
                </View>

                <Text style={{ fontSize: 11, fontFamily: 'monospace', color: COLORS.textMuted, marginTop: 6 }}>
                  HASH: {insResult.statutory_verification.evidence_cryptographic_sha256.slice(0, 32)}...
                </Text>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: MICROCLIMATE & SPRAY EARLY WARNING                   */}
        {/* ========================================================= */}
        {activeTab === 'microclimate' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🌦️ Microclimate & Spray Early Warning</Text>
              <Text style={styles.cardDesc}>
                Penman-Monteith ET0, Vapor Pressure Deficit (VPD), Delta-T droplet evaporation, and nocturnal frost / diurnal heatwave radar.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="Location / Tehsil" unit="" value={mcLoc} onChangeText={setMcLoc} numeric={false} />
                <Field label="Crop Variety" unit="" value={mcCrop} onChangeText={setMcCrop} numeric={false} />
              </View>
              <View style={styles.fieldRow}>
                <Field label="Ambient Temp" unit="°C" value={mcTemp} onChangeText={setMcTemp} />
                <Field label="Humidity" unit="%" value={mcHum} onChangeText={setMcHum} />
              </View>
              <Field label="Wind Speed" unit="km/h" value={mcWind} onChangeText={setMcWind} />

              <TouchableOpacity
                style={[styles.primaryButton, mcLoading && styles.buttonDisabled]}
                onPress={() => handleRunMicroclimate()}
                disabled={mcLoading}
              >
                {mcLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>📡 Compute Microclimate Radar</Text>
                )}
              </TouchableOpacity>
            </View>

            {mcResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>Spray Status: {mcResult.spray_window.status}</Text>
                  <View
                    style={[
                      styles.statePill,
                      {
                        backgroundColor: mcResult.spray_window.can_spray ? '#dcfce7' : '#fee2e2',
                      },
                    ]}
                  >
                    <Text
                      style={{
                        color: mcResult.spray_window.can_spray ? '#166534' : '#991b1b',
                        fontWeight: '800',
                        fontSize: 12,
                      }}
                    >
                      {mcResult.spray_window.can_spray ? 'SAFE TO SPRAY' : 'SPRAY HAZARD'}
                    </Text>
                  </View>
                </View>

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{mcResult.microclimate_metrics.vapor_pressure_deficit_kpa} kPa</Text>
                    <Text style={styles.statLabel}>VPD Deficit</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{mcResult.microclimate_metrics.reference_et0_mm_day} mm</Text>
                    <Text style={styles.statLabel}>Daily ET0</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#d97706' }]}>{mcResult.spray_window.delta_t_c} °C</Text>
                    <Text style={styles.statLabel}>Delta T</Text>
                  </View>
                </View>

                {mcResult.early_warning_alerts && mcResult.early_warning_alerts.length > 0 && (
                  mcResult.early_warning_alerts.map((al, idx) => (
                    <View key={idx} style={[styles.tableCard, { backgroundColor: '#fef2f2', borderColor: '#fca5a5', marginTop: 8 }]}>
                      <Text style={{ color: '#b91c1c', fontWeight: '800', fontSize: 13 }}>🚨 {al.type} ({al.severity})</Text>
                      <Text style={{ color: '#7f1d1d', fontSize: 12, marginTop: 4 }}>{al.message}</Text>
                      <Text style={{ color: '#15803d', fontSize: 12, marginTop: 4, fontWeight: '700' }}>👉 {al.mitigation}</Text>
                    </View>
                  ))
                )}

                <View style={[styles.tableCard, { marginTop: 8 }]}>
                  <Text style={{ fontWeight: '700', color: COLORS.forest }}>Agronomic Advisory:</Text>
                  <Text style={{ fontSize: 13, color: COLORS.text, marginTop: 4 }}>{mcResult.spray_window.advisory}</Text>
                </View>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: FARM LEDGER & CARBON CREDIT OFFSET                   */}
        {/* ========================================================= */}
        {activeTab === 'carbon' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>🌱 Farm Ledger & Carbon Credits</Text>
              <Text style={styles.cardDesc}>
                Double-entry farm economics, cost per quintal, and IPCC-aligned verified carbon credits (VCM) with SHA-256 integrity hash.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="Farmer Name" unit="" value={carbFarmer} onChangeText={setCarbFarmer} numeric={false} />
                <Field label="Farm ID" unit="" value={carbId} onChangeText={setCarbId} numeric={false} />
              </View>
              <Field label="Holding Area" unit="Hectares" value={carbHa} onChangeText={setCarbHa} />

              <TouchableOpacity
                style={[styles.primaryButton, carbLoading && styles.buttonDisabled]}
                onPress={() => handleRunCarbon()}
                disabled={carbLoading}
              >
                {carbLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🌱 Audit Carbon Sequestration</Text>
                )}
              </TouchableOpacity>
            </View>

            {carbResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>Audit ID: {carbResult.verification_certificate.certificate_id}</Text>
                  <View style={[styles.statePill, { backgroundColor: '#dcfce7' }]}>
                    <Text style={{ color: '#166534', fontWeight: '800', fontSize: 12 }}>VERIFIED</Text>
                  </View>
                </View>

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#16a34a' }]}>{carbResult.total_co2e_sequestered_tons} t</Text>
                    <Text style={styles.statLabel}>CO2e Offset</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>₹{carbResult.vcm_spot_price_per_ton_inr}</Text>
                    <Text style={styles.statLabel}>Price/Ton</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#15803d' }]}>₹{carbResult.estimated_carbon_dividend_inr.toLocaleString()}</Text>
                    <Text style={styles.statLabel}>Annual Payout</Text>
                  </View>
                </View>

                <Text style={styles.fieldSectionTitle}>Verified Practices:</Text>
                {carbResult.practices_breakdown && carbResult.practices_breakdown.map((p, idx) => (
                  <View key={idx} style={[styles.tableCard, { marginVertical: 4 }]}>
                    <View style={styles.rowBetween}>
                      <Text style={[styles.bold, { color: COLORS.forest, flex: 1 }]}>🌿 {p.title}</Text>
                      <Text style={{ fontWeight: '800', color: '#15803d' }}>₹{p.valuation_inr.toLocaleString()}</Text>
                    </View>
                    <Text style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 4 }}>{p.description}</Text>
                  </View>
                ))}

                <Text style={{ fontSize: 11, fontFamily: 'monospace', color: COLORS.textMuted, marginTop: 8 }}>
                  HASH: {carbResult.verification_certificate.integrity_hash.slice(0, 32)}...
                </Text>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: PRECISION SMART IRRIGATION & SOLAR DIGITAL TWIN      */}
        {/* ========================================================= */}
        {activeTab === 'irrigation' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>💧 Precision Smart Irrigation</Text>
              <Text style={styles.cardDesc}>
                Crop ETc, soil water deficit balance, PM-KUSUM solar PV pump dispatch, and automated IoT MQTT telemetry packets.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="Pump / Zone ID" unit="" value={irrId} onChangeText={setIrrId} numeric={false} />
                <Field label="Crop Variety" unit="" value={irrCrop} onChangeText={setIrrCrop} numeric={false} />
              </View>
              <View style={styles.fieldRow}>
                <Field label="Field Area" unit="Acres" value={irrArea} onChangeText={setIrrArea} />
                <Field label="Soil Moisture" unit="%" value={irrMoist} onChangeText={setIrrMoist} />
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, irrLoading && styles.buttonDisabled]}
                onPress={() => handleRunIrrigation()}
                disabled={irrLoading}
              >
                {irrLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🌊 Calculate Irrigation Schedule</Text>
                )}
              </TouchableOpacity>
            </View>

            {irrResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>Status: {irrResult.irrigation_urgency} IRRIGATION</Text>
                  <View style={[styles.statePill, { backgroundColor: '#dbeafe' }]}>
                    <Text style={{ color: '#1e40af', fontWeight: '800', fontSize: 12 }}>{irrResult.water_budget.application_efficiency_pct}% EFF</Text>
                  </View>
                </View>

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{irrResult.water_budget.gross_water_needed_liters.toLocaleString()} L</Text>
                    <Text style={styles.statLabel}>Water Needed</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#0284c7' }]}>{irrResult.solar_pump_dispatch.runtime_minutes} M</Text>
                    <Text style={styles.statLabel}>Pump Runtime</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#16a34a' }]}>{irrResult.water_budget.liters_saved_vs_flood.toLocaleString()} L</Text>
                    <Text style={styles.statLabel}>Water Saved</Text>
                  </View>
                </View>

                <View style={[styles.tableCard, { backgroundColor: '#eff6ff', borderColor: '#93c5fd', marginTop: 8 }]}>
                  <Text style={{ color: '#1e40af', fontWeight: '800' }}>☀️ PM-KUSUM Solar Synchronization:</Text>
                  <Text style={{ fontSize: 13, color: '#1e3a8a', marginTop: 4 }}>
                    Window: {irrResult.solar_pump_dispatch.solar_window} ({irrResult.solar_pump_dispatch.solar_power_source})
                  </Text>
                </View>

                <View style={[styles.tableCard, { backgroundColor: '#1e293b', marginTop: 8 }]}>
                  <Text style={{ color: '#38bdf8', fontFamily: 'monospace', fontSize: 12 }}>
                    MQTT: {irrResult.iot_mqtt_control_packet.topic}
                  </Text>
                  <Text style={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: 11, marginTop: 4 }}>
                    CMD: {irrResult.iot_mqtt_control_packet.command} · Flow: {irrResult.iot_mqtt_control_packet.parameters.pump_flow_lpm} LPM
                  </Text>
                </View>
              </View>
            )}
          </View>
        )}

        {/* ========================================================= */}
        {/* TAB: BIO-SURVEILLANCE & EPIDEMIC SWARM RADAR              */}
        {/* ========================================================= */}
        {activeTab === 'epidemic' && (
          <View>
            <View style={styles.card}>
              <Text style={styles.cardTitle}>☣️ Bio-Surveillance & Epidemic Radar</Text>
              <Text style={styles.cardDesc}>
                Regional pathogen cluster surveillance, aerodynamic spore dispersal corridor modeling, and epidemiological R0 transmission factor.
              </Text>

              <View style={styles.fieldRow}>
                <Field label="District Epicenter" unit="" value={epiOrigin} onChangeText={setEpiOrigin} numeric={false} />
                <Field label="Infected Foci" unit="Plots" value={epiCases} onChangeText={setEpiCases} />
              </View>

              <TouchableOpacity
                style={[styles.primaryButton, epiLoading && styles.buttonDisabled]}
                onPress={() => handleRunEpidemic()}
                disabled={epiLoading}
              >
                {epiLoading ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.primaryButtonText}>🚨 Simulate Outbreak Corridor</Text>
                )}
              </TouchableOpacity>
            </View>

            {epiResult && (
              <View style={styles.cardResult}>
                <View style={styles.rowBetween}>
                  <Text style={styles.conditionText}>{epiResult.epidemiological_metrics.threat_level}</Text>
                </View>

                <View style={styles.statRow}>
                  <View style={styles.statCard}>
                    <Text style={[styles.statValue, { color: '#dc2626' }]}>{epiResult.epidemiological_metrics.effective_reproduction_rt}</Text>
                    <Text style={styles.statLabel}>Reproduction Rt</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{epiResult.epidemiological_metrics.daily_dispersal_velocity_km} km/d</Text>
                    <Text style={styles.statLabel}>Dispersal Speed</Text>
                  </View>
                  <View style={styles.statCard}>
                    <Text style={styles.statValue}>{epiResult.epidemiological_metrics.wind_corridor_heading}</Text>
                    <Text style={styles.statLabel}>Downwind</Text>
                  </View>
                </View>

                <Text style={styles.fieldSectionTitle}>Downwind Spatial Trajectory:</Text>
                {epiResult.downwind_trajectory_forecast && epiResult.downwind_trajectory_forecast.map((step, idx) => (
                  <View key={idx} style={[styles.tableCard, { marginVertical: 4 }]}>
                    <View style={styles.rowBetween}>
                      <Text style={[styles.bold, { color: COLORS.forest }]}>⏱️ {step.forecast_window}</Text>
                      <Text style={{ fontWeight: '700', color: '#b91c1c' }}>{step.projected_active_foci} farms ({step.estimated_acreage_threatened} ac)</Text>
                    </View>
                    <Text style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 4 }}>
                      Corridor Reach: {step.corridor_reach_km} km downwind ({step.vector_status})
                    </Text>
                  </View>
                ))}

                <View style={[styles.tableCard, { backgroundColor: '#fef2f2', borderColor: '#fca5a5', marginTop: 8 }]}>
                  <Text style={{ color: '#b91c1c', fontWeight: '800' }}>🛡️ Prophylactic Bio-Barrier Protocol:</Text>
                  <Text style={{ fontSize: 12, color: '#7f1d1d', marginTop: 4 }}>
                    Quarantine Buffer: {epiResult.prophylactic_barrier_protocol.quarantine_buffer_radius_km} km radius
                  </Text>
                  <Text style={{ fontSize: 12, color: '#15803d', marginTop: 4, fontWeight: '700' }}>
                    {epiResult.prophylactic_barrier_protocol.advisory}
                  </Text>
                </View>
              </View>
            )}
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
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.bottomBarScroll}
        >
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
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------
// Design tokens - single source of truth for the visual language
// ---------------------------------------------------------------
const COLORS = {
  forest: '#064e3b', // rich deep emerald
  forestDeep: '#022c22', // midnight emerald
  leaf: '#10b981', // vibrant emerald
  leafDark: '#047857',
  leafSoft: '#ecfdf5',
  mint: '#f0fdf4',
  bg: '#f8fafc', // ultra-clean modern app canvas
  card: '#ffffff',
  border: '#e2e8f0', // crisp slate border
  borderFocus: '#10b981',
  text: '#0f172a', // high-contrast slate-900
  textMuted: '#64748b', // slate-500
  gold: '#f59e0b', // warm vibrant amber
  goldSoft: '#fffbeb',
  goldText: '#b45309',
  danger: '#ef4444',
  dangerSoft: '#fef2f2',
  white: '#ffffff',
};

const RADIUS = { sm: 10, md: 14, lg: 18, xl: 24, pill: 999 };
const SPACE = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 };

const cardShadow = {
  shadowColor: '#0f172a',
  shadowOffset: { width: 0, height: 4 },
  shadowOpacity: 0.07,
  shadowRadius: 14,
  elevation: 3,
};

const buttonShadow = {
  shadowColor: '#059669',
  shadowOffset: { width: 0, height: 4 },
  shadowOpacity: 0.22,
  shadowRadius: 8,
  elevation: 4,
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
    paddingTop: SPACE.md + (Platform.OS === 'android' ? StatusBar.currentHeight || 0 : 4),
    paddingBottom: SPACE.lg + 2,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.08)',
  },
  headerTitle: {
    fontSize: 25,
    fontWeight: '900',
    color: COLORS.white,
    letterSpacing: 0.4,
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#a7f3d0',
    fontWeight: '600',
    marginTop: 3,
    letterSpacing: 0.2,
  },
  statusBadge: {
    minHeight: 32,
    justifyContent: 'center',
    paddingHorizontal: SPACE.md,
    paddingVertical: 5,
    borderRadius: RADIUS.pill,
    borderWidth: 1.5,
  },
  badgeOnline: {
    backgroundColor: 'rgba(16, 185, 129, 0.22)',
    borderColor: '#34d399',
  },
  badgeOffline: {
    backgroundColor: 'rgba(239, 68, 68, 0.22)',
    borderColor: '#f87171',
  },
  statusText: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.3,
  },
  container: {
    padding: SPACE.lg,
    backgroundColor: COLORS.bg,
    paddingBottom: 60,
    flexGrow: 1,
  },
  card: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    padding: SPACE.lg + 4,
    marginBottom: SPACE.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
    ...cardShadow,
  },
  cardTitle: {
    fontSize: 19,
    fontWeight: '800',
    color: COLORS.text,
    marginBottom: SPACE.xs,
    letterSpacing: -0.2,
  },
  cardDesc: {
    fontSize: 13.5,
    color: COLORS.textMuted,
    marginBottom: SPACE.lg,
    lineHeight: 20,
  },
  imageDropzone: {
    minHeight: 210,
    backgroundColor: COLORS.leafSoft,
    borderRadius: RADIUS.md,
    borderWidth: 2,
    borderColor: '#a7f3d0',
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: SPACE.md,
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: 230,
    resizeMode: 'cover',
  },
  placeholderBox: {
    alignItems: 'center',
    paddingHorizontal: SPACE.lg,
  },
  placeholderIcon: {
    fontSize: 44,
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
    fontWeight: '800',
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
    marginTop: SPACE.sm + 2,
    borderBottomWidth: 3,
    borderBottomColor: COLORS.gold,
    ...buttonShadow,
  },
  primaryButtonText: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: '800',
    textAlign: 'center',
    letterSpacing: 0.3,
  },
  buttonDisabled: {
    opacity: 0.65,
  },
  secondaryButton: {
    backgroundColor: COLORS.leafSoft,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: '#6ee7b7',
    minHeight: 48,
    paddingVertical: SPACE.sm + 2,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: SPACE.md,
  },
  secondaryButtonText: {
    color: COLORS.forest,
    fontWeight: '800',
    fontSize: 14.5,
  },
  cardResult: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    padding: SPACE.lg + 4,
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
    fontSize: 21,
    fontWeight: '900',
    color: COLORS.text,
    letterSpacing: -0.3,
  },
  pathogenText: {
    fontSize: 13.5,
    color: COLORS.textMuted,
    marginTop: 3,
    fontWeight: '500',
  },
  scoreBadge: {
    backgroundColor: COLORS.leafSoft,
    paddingHorizontal: SPACE.md + 2,
    paddingVertical: SPACE.sm,
    borderRadius: RADIUS.md,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#a7f3d0',
  },
  scoreBadgeVal: {
    color: COLORS.forest,
    fontWeight: '900',
    fontSize: 17,
  },
  scoreBadgeLabel: {
    color: COLORS.forest,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  metricRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    backgroundColor: '#f1f5f9',
    padding: SPACE.md,
    borderRadius: RADIUS.sm + 2,
    marginBottom: SPACE.md,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  metricItem: {
    fontSize: 13.5,
    color: COLORS.text,
    marginVertical: 2,
  },
  bold: {
    fontWeight: '800',
    color: COLORS.forest,
  },
  symptomsBox: {
    backgroundColor: COLORS.goldSoft,
    padding: SPACE.md + 2,
    borderRadius: RADIUS.sm + 2,
    borderLeftWidth: 4,
    borderLeftColor: COLORS.gold,
    marginBottom: SPACE.md,
    borderWidth: 1,
    borderColor: '#fef3c7',
  },
  symptomsTitle: {
    fontWeight: '800',
    color: COLORS.goldText,
    fontSize: 13.5,
    letterSpacing: 0.2,
  },
  symptomsText: {
    color: '#78350f',
    fontSize: 13,
    marginTop: 5,
    lineHeight: 19,
  },
  treatmentHeading: {
    fontSize: 16,
    fontWeight: '800',
    color: COLORS.text,
    marginBottom: SPACE.sm + 2,
  },
  segmentBar: {
    flexDirection: 'row',
    backgroundColor: '#e2e8f0',
    borderRadius: RADIUS.sm + 2,
    padding: 3,
    marginBottom: SPACE.md,
  },
  segmentBtn: {
    flex: 1,
    minHeight: 40,
    paddingVertical: SPACE.sm,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: RADIUS.sm,
  },
  segmentBtnActive: {
    backgroundColor: COLORS.white,
    ...cardShadow,
  },
  segmentText: {
    fontSize: 12.5,
    fontWeight: '600',
    color: COLORS.textMuted,
  },
  segmentTextActive: {
    color: COLORS.forest,
    fontWeight: '800',
  },
  treatmentList: {
    backgroundColor: COLORS.leafSoft,
    padding: SPACE.md + 2,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: '#d1fae5',
  },
  treatmentBullet: {
    fontSize: 14,
    color: '#064e3b',
    marginBottom: 7,
    lineHeight: 21,
    fontWeight: '500',
  },
  fieldSectionTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: COLORS.text,
    marginBottom: SPACE.sm + 2,
    letterSpacing: -0.1,
  },
  presetsRow: {
    flexGrow: 0,
    marginBottom: SPACE.lg,
  },
  presetChip: {
    backgroundColor: COLORS.white,
    borderColor: '#cbd5e1',
    borderWidth: 1.5,
    minHeight: 40,
    justifyContent: 'center',
    paddingHorizontal: SPACE.lg,
    borderRadius: RADIUS.pill,
    marginRight: SPACE.sm,
    ...cardShadow,
  },
  presetChipText: {
    color: COLORS.text,
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
    fontSize: 12.5,
    fontWeight: '700',
    color: '#475569',
    marginBottom: 6,
  },
  textInput: {
    backgroundColor: COLORS.white,
    borderWidth: 1.5,
    borderColor: '#cbd5e1',
    borderRadius: RADIUS.sm + 2,
    minHeight: 48,
    paddingHorizontal: SPACE.md,
    paddingVertical: SPACE.sm + 2,
    fontSize: 15,
    color: COLORS.text,
  },
  heroCropCard: {
    backgroundColor: COLORS.forest,
    borderRadius: RADIUS.lg,
    padding: SPACE.xl,
    marginBottom: SPACE.lg,
    borderBottomWidth: 4,
    borderBottomColor: COLORS.gold,
    ...buttonShadow,
  },
  heroCropBadge: {
    color: '#fef08a',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.3,
    marginBottom: SPACE.xs,
    textTransform: 'uppercase',
  },
  heroCropName: {
    color: COLORS.white,
    fontSize: 30,
    fontWeight: '900',
    letterSpacing: -0.4,
  },
  heroCropScore: {
    color: '#6ee7b7',
    fontSize: 22,
    fontWeight: '900',
    marginTop: 2,
  },
  heroCropYield: {
    color: '#dcfce7',
    fontSize: 14,
    marginTop: SPACE.sm,
    fontWeight: '500',
  },
  rankedTitle: {
    fontSize: 17,
    fontWeight: '800',
    color: COLORS.text,
    marginBottom: SPACE.md,
  },
  cropRankCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    padding: SPACE.lg,
    marginBottom: SPACE.md,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderLeftWidth: 5,
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
    fontSize: 16.5,
    fontWeight: '800',
    color: COLORS.text,
  },
  cropRankMatch: {
    fontSize: 16,
    fontWeight: '900',
    color: COLORS.leafDark,
  },
  cropDetailLine: {
    fontSize: 13,
    color: COLORS.textMuted,
    marginTop: 3,
    lineHeight: 19,
  },
  cropTipText: {
    fontSize: 12.5,
    color: COLORS.goldText,
    backgroundColor: COLORS.goldSoft,
    padding: SPACE.sm + 2,
    borderRadius: RADIUS.sm,
    marginTop: SPACE.sm,
    lineHeight: 18,
    borderWidth: 1,
    borderColor: '#fef3c7',
  },
  smallDropzone: {
    minHeight: 104,
    backgroundColor: COLORS.leafSoft,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: '#a7f3d0',
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
    height: 104,
  },
  configHelp: {
    fontSize: 12.5,
    color: COLORS.goldText,
    marginTop: SPACE.lg,
    lineHeight: 19,
    backgroundColor: COLORS.goldSoft,
    padding: SPACE.md,
    borderRadius: RADIUS.sm + 2,
    borderWidth: 1,
    borderColor: '#fef3c7',
  },
  // --- bottom tab bar ---
  bottomBar: {
    flexDirection: 'row',
    backgroundColor: COLORS.forestDeep,
    paddingTop: SPACE.sm + 2,
    paddingBottom: Platform.OS === 'ios' ? SPACE.lg : SPACE.sm + 4,
    paddingHorizontal: SPACE.xs,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  tabItem: {
    minWidth: 64,
    minHeight: 52,
    paddingHorizontal: 4,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabIconWrap: {
    minWidth: 46,
    height: 32,
    borderRadius: RADIUS.pill,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabIconWrapActive: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.35)',
  },
  tabIcon: {
    fontSize: 19,
  },
  tabLabel: {
    marginTop: 3,
    fontSize: 10.5,
    fontWeight: '600',
    color: '#94a3b8',
    letterSpacing: 0.1,
  },
  tabLabelActive: {
    color: '#ffffff',
    fontWeight: '800',
  },
  // --- home ---
  homeBanner: {
    backgroundColor: COLORS.forest,
    borderRadius: RADIUS.xl,
    padding: SPACE.xl,
    marginBottom: SPACE.lg,
    borderBottomWidth: 4,
    borderBottomColor: COLORS.gold,
    ...buttonShadow,
  },
  homeBannerKicker: {
    color: '#fef08a',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1.4,
    marginBottom: SPACE.xs,
    textTransform: 'uppercase',
  },
  homeBannerTitle: {
    color: COLORS.white,
    fontSize: 26,
    fontWeight: '900',
    lineHeight: 32,
    letterSpacing: -0.3,
  },
  homeBannerText: {
    color: '#dcfce7',
    fontSize: 14,
    lineHeight: 21,
    marginTop: SPACE.sm,
    fontWeight: '500',
  },
  statusCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: SPACE.md + 2,
    marginBottom: SPACE.xl,
    ...cardShadow,
  },
  statusDot: {
    width: 14,
    height: 14,
    borderRadius: 7,
    marginRight: SPACE.md,
  },
  dotOnline: { backgroundColor: COLORS.leaf },
  dotOffline: { backgroundColor: COLORS.danger },
  statusCardBody: {
    flex: 1,
    paddingRight: SPACE.sm,
  },
  statusCardTitle: {
    fontSize: 14.5,
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
    backgroundColor: COLORS.leafSoft,
  },
  statusCardBtnText: {
    color: COLORS.forest,
    fontWeight: '800',
    fontSize: 13,
  },
  sectionHeading: {
    fontSize: 17,
    fontWeight: '800',
    color: COLORS.text,
    marginBottom: SPACE.md,
    letterSpacing: -0.2,
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
    minHeight: 88,
    ...cardShadow,
  },
  featureIconTile: {
    width: 54,
    height: 54,
    borderRadius: RADIUS.md + 2,
    backgroundColor: COLORS.leafSoft,
    borderWidth: 1,
    borderColor: '#d1fae5',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: SPACE.md,
  },
  featureIcon: {
    fontSize: 27,
  },
  featureBody: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 16.5,
    fontWeight: '800',
    color: COLORS.text,
    letterSpacing: -0.2,
  },
  featureDesc: {
    fontSize: 13,
    color: COLORS.textMuted,
    marginTop: 3,
    lineHeight: 19,
  },
  featureArrow: {
    fontSize: 26,
    color: COLORS.gold,
    marginLeft: SPACE.sm,
    fontWeight: '800',
  },
  // --- results / bars ---
  statePill: {
    alignSelf: 'flex-start',
    paddingHorizontal: SPACE.md,
    paddingVertical: 5,
    borderRadius: RADIUS.pill,
    marginBottom: SPACE.sm,
  },
  statePillText: {
    fontSize: 11.5,
    fontWeight: '900',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
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
    backgroundColor: '#e2e8f0',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: RADIUS.pill,
  },
  heroBarTrack: {
    height: 8,
    borderRadius: RADIUS.pill,
    backgroundColor: 'rgba(255,255,255,0.22)',
    overflow: 'hidden',
    marginTop: SPACE.md,
  },
  // --- grouped form ---
  formSection: {
    backgroundColor: COLORS.leafSoft,
    borderRadius: RADIUS.lg,
    borderWidth: 1,
    borderColor: '#d1fae5',
    padding: SPACE.lg,
    marginBottom: SPACE.md,
  },
  formSectionTitle: {
    fontSize: 14.5,
    fontWeight: '800',
    color: COLORS.forest,
    marginBottom: SPACE.md,
  },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.white,
    borderWidth: 1.5,
    borderColor: '#cbd5e1',
    borderRadius: RADIUS.sm + 2,
    minHeight: 48,
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
    minHeight: 48,
    marginHorizontal: 4,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: '#6ee7b7',
    backgroundColor: COLORS.leafSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  photoActionText: {
    color: COLORS.forest,
    fontWeight: '800',
    fontSize: 14,
  },
  smallCamBtn: {
    minHeight: 40,
    marginTop: SPACE.sm,
    borderRadius: RADIUS.sm + 2,
    borderWidth: 1.5,
    borderColor: '#6ee7b7',
    backgroundColor: COLORS.leafSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  smallCamText: {
    color: COLORS.forest,
    fontWeight: '800',
    fontSize: 13,
  },
  // --- result actions (PDF / share) ---
  resultActions: {
    flexDirection: 'row',
    marginTop: SPACE.lg,
  },
  actionBtn: {
    flex: 1,
    minHeight: 50,
    marginHorizontal: 4,
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: '#6ee7b7',
    backgroundColor: COLORS.leafSoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  actionBtnPrimary: {
    backgroundColor: COLORS.forest,
    borderColor: COLORS.forest,
    ...buttonShadow,
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
    backgroundColor: '#e2e8f0',
    borderRadius: RADIUS.sm + 2,
    padding: 3,
    alignSelf: 'flex-start',
  },
  langBtn: {
    minHeight: 38,
    minWidth: 88,
    paddingHorizontal: SPACE.md,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: RADIUS.sm,
  },
  langBtnActive: {
    backgroundColor: COLORS.white,
    ...cardShadow,
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
    borderWidth: 1.5,
    borderColor: '#a7f3d0',
    borderRadius: RADIUS.md,
    minHeight: 46,
    justifyContent: 'center',
    paddingHorizontal: SPACE.md,
    marginBottom: SPACE.sm,
    ...cardShadow,
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
    borderRadius: RADIUS.lg,
    paddingHorizontal: SPACE.md + 2,
    paddingVertical: SPACE.sm + 4,
    ...cardShadow,
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
    lineHeight: 21,
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
    borderColor: '#cbd5e1',
    borderRadius: RADIUS.md,
    paddingHorizontal: SPACE.md,
    paddingVertical: SPACE.sm + 2,
    fontSize: 15,
    color: COLORS.text,
  },
  sendBtn: {
    minHeight: 48,
    minWidth: 74,
    marginLeft: SPACE.sm,
    borderRadius: RADIUS.md,
    backgroundColor: COLORS.forest,
    borderBottomWidth: 3,
    borderBottomColor: COLORS.gold,
    alignItems: 'center',
    justifyContent: 'center',
    ...buttonShadow,
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
    ...cardShadow,
  },
  emptyIcon: {
    fontSize: 42,
    marginBottom: SPACE.sm,
  },
  emptyTitle: {
    fontSize: 17,
    fontWeight: '800',
    color: COLORS.text,
  },
  emptyText: {
    fontSize: 13.5,
    color: COLORS.textMuted,
    textAlign: 'center',
    marginTop: SPACE.xs,
    lineHeight: 19,
  },
  historyCard: {
    backgroundColor: COLORS.card,
    borderRadius: RADIUS.md,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderLeftWidth: 5,
    marginBottom: SPACE.md,
    overflow: 'hidden',
    ...cardShadow,
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
    fontSize: 15.5,
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
  // --- bottom bar scroll & items ---
  bottomBarScroll: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: SPACE.xs,
  },
  // --- extra action buttons ---
  actionBtnWarning: {
    backgroundColor: '#fffbeb',
    borderColor: '#f59e0b',
    borderWidth: 1.5,
  },
  actionBtnTextWarning: {
    color: '#b45309',
    fontWeight: '800',
    fontSize: 13.5,
  },
  // --- advisor & radar cards & tables ---
  statRow: {
    flexDirection: 'row',
    gap: SPACE.sm,
    marginVertical: SPACE.md,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#f8fafc',
    borderRadius: RADIUS.md,
    borderWidth: 1.5,
    borderColor: '#e2e8f0',
    padding: SPACE.sm + 2,
    alignItems: 'center',
    ...cardShadow,
  },
  statValue: {
    fontSize: 19,
    fontWeight: '900',
    color: COLORS.forest,
    letterSpacing: -0.3,
  },
  statLabel: {
    fontSize: 11,
    color: '#64748b',
    marginTop: 3,
    fontWeight: '700',
    textAlign: 'center',
    textTransform: 'uppercase',
    letterSpacing: 0.3,
  },
  rowBetween: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  cropChip: {
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: RADIUS.pill,
    backgroundColor: COLORS.card,
    borderWidth: 1.5,
    borderColor: '#cbd5e1',
    marginRight: 8,
    marginBottom: 8,
    ...cardShadow,
  },
  cropChipActive: {
    backgroundColor: COLORS.forest,
    borderColor: COLORS.forest,
  },
  cropChipText: {
    fontSize: 13,
    fontWeight: '700',
    color: COLORS.text,
  },
  cropChipTextActive: {
    color: COLORS.white,
    fontWeight: '800',
  },
  tableCard: {
    backgroundColor: '#f8fafc',
    borderRadius: RADIUS.md,
    padding: SPACE.md + 2,
    marginVertical: SPACE.sm,
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  tableRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 7,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  tableKey: {
    fontSize: 13,
    color: '#64748b',
    fontWeight: '600',
  },
  tableVal: {
    fontSize: 13.5,
    color: COLORS.text,
    fontWeight: '800',
  },
  tableValProfit: {
    fontSize: 15.5,
    color: '#047857',
    fontWeight: '900',
  },
  alertBadgeAlert: {
    backgroundColor: '#fef2f2',
    borderColor: '#ef4444',
    borderWidth: 1.5,
  },
  alertBadgeWatch: {
    backgroundColor: '#fffbeb',
    borderColor: '#f59e0b',
    borderWidth: 1.5,
  },
  alertBadgeTextAlert: {
    color: '#b91c1c',
    fontWeight: '800',
    fontSize: 11.5,
  },
  alertBadgeTextWatch: {
    color: '#b45309',
    fontWeight: '800',
    fontSize: 11.5,
  },
});

