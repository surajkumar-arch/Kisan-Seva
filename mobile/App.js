import React, { useState, useEffect } from 'react';
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
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const CROP_PRESETS = [
  { label: '🌾 Rice / Alluvial', ph: '6.6', n: '160', p: '45', k: '160', t: '28', h: '75', r: '180', soil: 'Alluvial Soil' },
  { label: '🌸 Cotton / Black', ph: '7.5', n: '120', p: '35', k: '110', t: '30', h: '55', r: '90', soil: 'Black Soil' },
  { label: '🥜 Groundnut / Sandy', ph: '6.8', n: '60', p: '45', k: '70', t: '31', h: '48', r: '50', soil: 'Sandy Soil' },
  { label: '🍃 Tea / Acidic Red', ph: '5.0', n: '120', p: '30', k: '90', t: '20', h: '85', r: '240', soil: 'Red Soil' },
];

export default function App() {
  const [backendUrl, setBackendUrl] = useState(
    'http://172.20.10.6:8010'
  );
  const [activeTab, setActiveTab] = useState('plant'); // 'plant', 'crop', 'audit', 'config'
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
      <StatusBar barStyle="light-content" backgroundColor="#064e3b" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>🌱 KisanSeva</Text>
          <Text style={styles.headerSubtitle}>AI Agricultural Decision Engine</Text>
        </View>
        <TouchableOpacity
          style={[styles.statusBadge, serverOnline ? styles.badgeOnline : styles.badgeOffline]}
          onPress={() => setActiveTab('config')}
        >
          <Text style={styles.statusText}>{serverOnline ? '● Online' : '● Offline'}</Text>
        </TouchableOpacity>
      </View>

      {/* Navigation Tabs */}
      <View style={styles.navBar}>
        <TouchableOpacity
          style={[styles.navItem, activeTab === 'plant' && styles.navItemActive]}
          onPress={() => setActiveTab('plant')}
        >
          <Text style={[styles.navText, activeTab === 'plant' && styles.navTextActive]}>🌿 Plant Doctor</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.navItem, activeTab === 'crop' && styles.navItemActive]}
          onPress={() => setActiveTab('crop')}
        >
          <Text style={[styles.navText, activeTab === 'crop' && styles.navTextActive]}>🌾 Crop Advisor</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.navItem, activeTab === 'audit' && styles.navItemActive]}
          onPress={() => setActiveTab('audit')}
        >
          <Text style={[styles.navText, activeTab === 'audit' && styles.navTextActive]}>🔬 Farm Audit</Text>
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={styles.container}>
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

              {plantLeafImage && (
                <TouchableOpacity
                  style={styles.clearBtn}
                  onPress={() => setPlantLeafImage(null)}
                >
                  <Text style={styles.clearBtnText}>Remove Photo</Text>
                </TouchableOpacity>
              )}

              <TouchableOpacity
                style={styles.primaryButton}
                onPress={handleDiagnosePlant}
                disabled={plantLoading}
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
              <View style={styles.cardResult}>
                <View style={styles.resultHeader}>
                  <View>
                    <Text style={styles.conditionText}>{plantDiagnosis.condition}</Text>
                    <Text style={styles.pathogenText}>Pathogen: {plantDiagnosis.pathogen_type}</Text>
                  </View>
                  <View style={styles.scoreBadge}>
                    <Text style={styles.scoreBadgeVal}>{plantDiagnosis.plant_health_score}/100</Text>
                    <Text style={styles.scoreBadgeLabel}>Health Score</Text>
                  </View>
                </View>

                <View style={styles.metricRow}>
                  <Text style={styles.metricItem}>
                    <b>Confidence:</b> {(plantDiagnosis.confidence * 100).toFixed(1)}%
                  </Text>
                  <Text style={styles.metricItem}>
                    <b>Severity:</b> {plantDiagnosis.severity_level}
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
                    onPress={() => setTreatmentProtocol('organic')}
                  >
                    <Text style={[styles.segmentText, treatmentProtocol === 'organic' && styles.segmentTextActive]}>
                      🍃 Organic
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.segmentBtn, treatmentProtocol === 'chemical' && styles.segmentBtnActive]}
                    onPress={() => setTreatmentProtocol('chemical')}
                  >
                    <Text style={[styles.segmentText, treatmentProtocol === 'chemical' && styles.segmentTextActive]}>
                      🧪 Chemical
                    </Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.segmentBtn, treatmentProtocol === 'preventive' && styles.segmentBtnActive]}
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

              {/* Inputs */}
              <View style={styles.gridRow}>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Soil pH (0-14):</Text>
                  <TextInput style={styles.textInput} value={cPh} onChangeText={setCPh} keyboardType="numeric" />
                </View>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Nitrogen (N) kg/ha:</Text>
                  <TextInput style={styles.textInput} value={cN} onChangeText={setCN} keyboardType="numeric" />
                </View>
              </View>

              <View style={styles.gridRow}>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Phosphorus (P) kg/ha:</Text>
                  <TextInput style={styles.textInput} value={cP} onChangeText={setCP} keyboardType="numeric" />
                </View>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Potassium (K) kg/ha:</Text>
                  <TextInput style={styles.textInput} value={cK} onChangeText={setCK} keyboardType="numeric" />
                </View>
              </View>

              <View style={styles.gridRow}>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Temperature (°C):</Text>
                  <TextInput style={styles.textInput} value={cTemp} onChangeText={setCTemp} keyboardType="numeric" />
                </View>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Humidity (%):</Text>
                  <TextInput style={styles.textInput} value={cHum} onChangeText={setCHum} keyboardType="numeric" />
                </View>
              </View>

              <View style={styles.gridRow}>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Rainfall (mm):</Text>
                  <TextInput style={styles.textInput} value={cRain} onChangeText={setCRain} keyboardType="numeric" />
                </View>
                <View style={styles.gridCol}>
                  <Text style={styles.inputLabel}>Soil Type:</Text>
                  <TextInput style={styles.textInput} value={cSoil} onChangeText={setCSoil} />
                </View>
              </View>

              <TouchableOpacity
                style={styles.primaryButton}
                onPress={handleRecommendCrops}
                disabled={cropLoading}
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
                    <Text style={styles.cropDetailLine}>
                      <b>Cycle:</b> {cand.growing_duration_days} | <b>Water:</b> {cand.water_requirement}
                    </Text>
                    <Text style={styles.cropDetailLine}>
                      <b>Expected Yield:</b> {cand.projected_yield}
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
                </View>
              </View>

              <TouchableOpacity
                style={styles.primaryButton}
                onPress={handleRunAudit}
                disabled={auditLoading}
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
                <View style={styles.metricRow}>
                  <Text style={styles.metricItem}>
                    <b>Farm Health Score:</b> {auditReport.summary?.overall_farm_health_score}/100
                  </Text>
                  <Text style={styles.metricItem}>
                    <b>Vigor Index (FVI):</b> {auditReport.summary?.farm_vigor_index}/100
                  </Text>
                </View>
                <Text style={styles.cropDetailLine}>
                  <b>Top Recommended Crop:</b> {auditReport.summary?.primary_crop}
                </Text>
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
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#064e3b',
  },
  header: {
    backgroundColor: '#064e3b',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 15,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '900',
    color: '#ffffff',
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#a7f3d0',
    fontWeight: '500',
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  badgeOnline: {
    backgroundColor: '#10b981',
  },
  badgeOffline: {
    backgroundColor: '#ef4444',
  },
  statusText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '700',
  },
  navBar: {
    flexDirection: 'row',
    backgroundColor: '#047857',
    borderBottomWidth: 1,
    borderBottomColor: '#059669',
  },
  navItem: {
    flex: 1,
    paddingVertical: 12,
    alignItems: 'center',
  },
  navItemActive: {
    borderBottomWidth: 3,
    borderBottomColor: '#34d399',
    backgroundColor: '#065f46',
  },
  navText: {
    color: '#d1fae5',
    fontWeight: '600',
    fontSize: 13,
  },
  navTextActive: {
    color: '#ffffff',
    fontWeight: '800',
  },
  container: {
    padding: 16,
    backgroundColor: '#f0fdf4',
    paddingBottom: 40,
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#064e3b',
    marginBottom: 4,
  },
  cardDesc: {
    fontSize: 13,
    color: '#4b5563',
    marginBottom: 14,
    lineHeight: 18,
  },
  imageDropzone: {
    height: 180,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#d1d5db',
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholderBox: {
    alignItems: 'center',
  },
  placeholderIcon: {
    fontSize: 40,
    marginBottom: 8,
  },
  placeholderText: {
    color: '#6b7280',
    fontSize: 14,
    fontWeight: '600',
  },
  clearBtn: {
    alignSelf: 'center',
    marginBottom: 12,
  },
  clearBtnText: {
    color: '#ef4444',
    fontWeight: '700',
    fontSize: 13,
  },
  primaryButton: {
    backgroundColor: '#059669',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 6,
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '800',
  },
  secondaryButton: {
    backgroundColor: '#e5e7eb',
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
    marginTop: 10,
  },
  secondaryButtonText: {
    color: '#374151',
    fontWeight: '700',
    fontSize: 14,
  },
  cardResult: {
    backgroundColor: '#ffffff',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 5,
    borderLeftColor: '#059669',
    elevation: 3,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  conditionText: {
    fontSize: 20,
    fontWeight: '900',
    color: '#111827',
  },
  pathogenText: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  scoreBadge: {
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#a7f3d0',
  },
  scoreBadgeVal: {
    color: '#065f46',
    fontWeight: '900',
    fontSize: 16,
  },
  scoreBadgeLabel: {
    color: '#065f46',
    fontSize: 10,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#f9fafb',
    padding: 10,
    borderRadius: 8,
    marginBottom: 10,
  },
  metricItem: {
    fontSize: 13,
    color: '#374151',
  },
  symptomsBox: {
    backgroundColor: '#fffbeb',
    padding: 10,
    borderRadius: 8,
    marginBottom: 12,
  },
  symptomsTitle: {
    fontWeight: '800',
    color: '#92400e',
    fontSize: 13,
  },
  symptomsText: {
    color: '#78350f',
    fontSize: 12,
    marginTop: 2,
    lineHeight: 16,
  },
  treatmentHeading: {
    fontSize: 15,
    fontWeight: '800',
    color: '#111827',
    marginBottom: 8,
  },
  segmentBar: {
    flexDirection: 'row',
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    padding: 3,
    marginBottom: 10,
  },
  segmentBtn: {
    flex: 1,
    paddingVertical: 8,
    alignItems: 'center',
    borderRadius: 6,
  },
  segmentBtnActive: {
    backgroundColor: '#ffffff',
    elevation: 1,
  },
  segmentText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6b7280',
  },
  segmentTextActive: {
    color: '#059669',
    fontWeight: '800',
  },
  treatmentList: {
    backgroundColor: '#f9fafb',
    padding: 12,
    borderRadius: 8,
  },
  treatmentBullet: {
    fontSize: 13,
    color: '#374151',
    marginBottom: 6,
    lineHeight: 18,
  },
  fieldSectionTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#374151',
    marginBottom: 6,
  },
  presetsRow: {
    flexDirection: 'row',
    marginBottom: 14,
  },
  presetChip: {
    backgroundColor: '#ecfdf5',
    borderColor: '#a7f3d0',
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    marginRight: 8,
  },
  presetChipText: {
    color: '#065f46',
    fontWeight: '700',
    fontSize: 12,
  },
  gridRow: {
    flexDirection: 'row',
    marginHorizontal: -6,
    marginBottom: 10,
  },
  gridCol: {
    flex: 1,
    paddingHorizontal: 6,
  },
  inputLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: '#4b5563',
    marginBottom: 4,
  },
  textInput: {
    backgroundColor: '#f9fafb',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 14,
    color: '#111827',
  },
  heroCropCard: {
    backgroundColor: '#065f46',
    borderRadius: 14,
    padding: 18,
    marginBottom: 14,
    elevation: 3,
  },
  heroCropBadge: {
    color: '#a7f3d0',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
    marginBottom: 4,
  },
  heroCropName: {
    color: '#ffffff',
    fontSize: 26,
    fontWeight: '900',
  },
  heroCropScore: {
    color: '#34d399',
    fontSize: 20,
    fontWeight: '800',
    marginTop: 2,
  },
  heroCropYield: {
    color: '#d1fae5',
    fontSize: 13,
    marginTop: 6,
  },
  rankedTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: '#064e3b',
    marginBottom: 10,
  },
  cropRankCard: {
    backgroundColor: '#ffffff',
    borderRadius: 10,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#10b981',
    elevation: 2,
  },
  cropRankHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  cropRankName: {
    fontSize: 16,
    fontWeight: '800',
    color: '#111827',
  },
  cropRankMatch: {
    fontSize: 15,
    fontWeight: '800',
    color: '#059669',
  },
  cropDetailLine: {
    fontSize: 12,
    color: '#4b5563',
    marginTop: 2,
  },
  cropTipText: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 6,
    fontStyle: 'italic',
  },
  smallDropzone: {
    height: 70,
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d1d5db',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  smallDropText: {
    fontSize: 12,
    color: '#6b7280',
    fontWeight: '600',
  },
  smallPreview: {
    width: '100%',
    height: '100%',
  },
  configHelp: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 12,
    lineHeight: 16,
  },
});
