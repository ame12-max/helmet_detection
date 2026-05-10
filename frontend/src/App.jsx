// App.jsx  ← THE ONLY REACT FILE YOU NEED
// =========================================
// Shows annotated frame returned by Flask
// so you can SEE the bounding boxes in the UI.

import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const API = 'http://localhost:5000';

function App() {
  const [selectedFile, setSelectedFile]   = useState(null);
  const [previewUrl,   setPreviewUrl]     = useState(null);   // raw preview
  const [annotatedUrl, setAnnotatedUrl]   = useState(null);   // result from backend
  const [violations,   setViolations]     = useState([]);
  const [stats,        setStats]          = useState({ total: 0, weekly: 0, platesDetected: 0 });
  const [loading,      setLoading]        = useState(false);
  const [uploadResult, setUploadResult]   = useState(null);
  const [searchTerm,   setSearchTerm]     = useState('');
  const [filterType,   setFilterType]     = useState('all');
  const [isLive,       setIsLive]         = useState(false);
  const [notification, setNotification]   = useState(null);

  const videoRef     = useRef(null);
  const streamRef    = useRef(null);
  const intervalRef  = useRef(null);
  const isLiveRef    = useRef(false);   // ✅ ref avoids stale closure in setInterval

  // ── Helpers ────────────────────────────────────────────────────────────────
  const notify = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3500);
  };

  // ── Fetch violations & stats ───────────────────────────────────────────────
  const fetchViolations = useCallback(async () => {
    try {
      const [vRes, sRes] = await Promise.all([
        axios.get(`${API}/violations`),
        axios.get(`${API}/stats`),
      ]);
      setViolations(vRes.data);
      setStats(sRes.data);
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    fetchViolations();
    const id = setInterval(fetchViolations, 10000);
    return () => clearInterval(id);
  }, [fetchViolations]);

  // ── File upload ────────────────────────────────────────────────────────────
  const handleFileChange = e => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnnotatedUrl(null);
    setUploadResult(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    const fd = new FormData();
    fd.append('image', selectedFile);
    try {
      const res = await axios.post(`${API}/upload`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      });
      setUploadResult(res.data);
      // Show annotated frame returned by Flask
      if (res.data.annotated_b64) {
        setAnnotatedUrl(`data:image/jpeg;base64,${res.data.annotated_b64}`);
      }
      notify(`Found ${res.data.violations} violation(s)!`,
             res.data.violations > 0 ? 'warning' : 'success');
      fetchViolations();
    } catch {
      setUploadResult({ error: 'Upload failed. Is Flask running on port 5000?' });
      notify('Upload failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  // ── Live stream ─────────────────────────────────────────────────────────────
  const captureFrame = useCallback(async () => {
    if (!isLiveRef.current) return;
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(async blob => {
      if (!blob) return;
      const fd = new FormData();
      fd.append('image', blob, 'frame.jpg');
      try {
        const res = await axios.post(`${API}/upload`, fd, { timeout: 10000 });
        if (res.data.violations > 0) {
          // Show annotated frame from live stream
          if (res.data.annotated_b64) {
            setAnnotatedUrl(`data:image/jpeg;base64,${res.data.annotated_b64}`);
          }
          notify(`⚠️ ${res.data.violations} person(s) without helmet!`, 'warning');
          fetchViolations();
        }
      } catch { /* silent during live */ }
    }, 'image/jpeg', 0.85);
  }, [fetchViolations]);

  const startLive = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play().then(() => {
          isLiveRef.current = true;
          setIsLive(true);
          notify('Live stream active!', 'success');
          captureFrame();
          intervalRef.current = setInterval(captureFrame, 5000);
        });
      };
    } catch (err) {
      notify(err.name === 'NotAllowedError'
        ? 'Camera access denied' : `Camera error: ${err.message}`, 'error');
    }
  };

  const stopLive = () => {
    isLiveRef.current = false;
    setIsLive(false);
    clearInterval(intervalRef.current);
    intervalRef.current = null;
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.onloadedmetadata = null;
    }
    notify('Live stream stopped', 'info');
  };

  useEffect(() => () => {
    isLiveRef.current = false;
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
  }, []);

  // ── Filtered table ─────────────────────────────────────────────────────────
  const filtered = violations.filter(v => {
    const matchSearch = v.plate?.toLowerCase().includes(searchTerm.toLowerCase())
                     || String(v.id).includes(searchTerm);
    const matchFilter = filterType === 'all'
      || (filterType === 'detected' && v.plate && v.plate !== 'UNKNOWN')
      || (filterType === 'unknown'  && (!v.plate || v.plate === 'UNKNOWN'));
    return matchSearch && matchFilter;
  });

  // ── Toast colour ───────────────────────────────────────────────────────────
  const toastColor = {
    success: 'bg-green-500', error: 'bg-red-500',
    warning: 'bg-yellow-500', info: 'bg-blue-500',
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">

      {/* Toast */}
      {notification && (
        <div className="fixed top-4 right-4 z-50 animate-slide-in">
          <div className={`px-6 py-3 rounded-lg shadow-lg text-white ${toastColor[notification.type]}`}>
            {notification.message}
          </div>
        </div>
      )}

      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              🚔 Helmet Violation System
            </h1>
            <p className="text-sm text-gray-500 mt-1">AI-powered detection &amp; license plate recognition</p>
          </div>
          {!isLive
            ? <button onClick={startLive}
                className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg flex items-center gap-2 shadow">
                <span className="animate-pulse">●</span> Live Stream
              </button>
            : <button onClick={stopLive}
                className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg shadow">
                ⬛ Stop Stream
              </button>
          }
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {[
            { label: 'Total Violations', value: stats.total,          icon: '🚨' },
            { label: 'Last 7 Days',      value: stats.weekly,         icon: '📅' },
            { label: 'Plates Detected',  value: stats.platesDetected, icon: '🚗' },
          ].map(s => (
            <div key={s.label} className="bg-white rounded-xl shadow p-6 hover:shadow-md transition">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-500 text-sm">{s.label}</p>
                  <p className="text-3xl font-bold text-gray-800">{s.value}</p>
                </div>
                <span className="text-4xl">{s.icon}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Live video panel — always in DOM so ref is valid */}
        <div className={`bg-white rounded-xl shadow p-6 mb-8 ${isLive ? '' : 'hidden'}`}>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <span className="animate-pulse text-red-500">●</span> Live Surveillance
            <span className="ml-auto text-sm font-normal text-gray-400">Auto-captures every 5s</span>
          </h2>
          <video ref={videoRef} autoPlay playsInline muted
            className="w-full rounded-lg border border-gray-200 bg-black"
            style={{ maxHeight: 420 }} />
        </div>

        {/* Upload + result */}
        <div className="bg-white rounded-xl shadow p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
          <div className="flex flex-wrap gap-3 mb-4">
            <label className="cursor-pointer bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg shadow">
              📁 Choose File
              <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
            </label>
            <button onClick={handleUpload} disabled={!selectedFile || loading}
              className={`px-6 py-2 rounded-lg font-medium shadow transition ${
                !selectedFile || loading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}>
              {loading ? '⏳ Analyzing…' : '🔍 Upload & Detect'}
            </button>
          </div>

          {/* Side-by-side: original + annotated */}
          {(previewUrl || annotatedUrl) && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
              {previewUrl && (
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Original</p>
                  <img src={previewUrl} alt="Original"
                    className="w-full rounded-lg border border-gray-200 object-contain max-h-72" />
                </div>
              )}
              {annotatedUrl && (
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">Detected (annotated)</p>
                  <img src={annotatedUrl} alt="Annotated"
                    className="w-full rounded-lg border border-gray-200 object-contain max-h-72" />
                </div>
              )}
            </div>
          )}

          {/* Result card */}
          {uploadResult && (
            <div className={`mt-4 p-4 rounded-lg ${
              uploadResult.error
                ? 'bg-red-50 border border-red-200'
                : uploadResult.violations > 0
                  ? 'bg-red-50 border border-red-200'
                  : 'bg-green-50 border border-green-200'
            }`}>
              {uploadResult.error
                ? <p className="text-red-700">⚠️ {uploadResult.error}</p>
                : <>
                    <p className="font-semibold text-lg">
                      {uploadResult.violations > 0 ? '🚨' : '✅'} Violations: {uploadResult.violations}
                    </p>
                    <p className="mt-1">
                      🚗 Plate: <span className="font-mono font-bold text-blue-600">
                        {uploadResult.plate || 'Not detected'}
                      </span>
                    </p>
                    <p className="mt-1 text-sm text-gray-500">
                      Status: {uploadResult.helmet_status}
                    </p>
                  </>
              }
            </div>
          )}
        </div>

        {/* Search & filter */}
        <div className="bg-white rounded-xl shadow p-4 mb-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <input type="text" placeholder="🔍 Search by plate or ID…"
              value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400 focus:border-transparent" />
            <select value={filterType} onChange={e => setFilterType(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-400">
              <option value="all">All Violations</option>
              <option value="detected">With Plate Detected</option>
              <option value="unknown">Unknown Plate</option>
            </select>
          </div>
        </div>

        {/* Violation log table */}
        <div className="bg-white rounded-xl shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
            <h2 className="text-xl font-semibold">📋 Violation Log</h2>
            <p className="text-sm text-gray-500 mt-1">
              {filtered.length} of {violations.length} records
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {['ID', 'Timestamp', 'Plate Number', 'Evidence'].map(h => (
                    <th key={h} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filtered.length === 0
                  ? <tr><td colSpan="4" className="px-6 py-10 text-center text-gray-400">
                      No violations yet. Upload an image or start the live stream.
                    </td></tr>
                  : filtered.map(v => (
                    <tr key={v.id} className="hover:bg-gray-50 transition">
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">#{v.id}</td>
                      <td className="px-6 py-4 text-sm text-gray-500">
                        {new Date(v.time).toLocaleString()}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`text-sm font-mono font-bold px-2 py-1 rounded ${
                          v.plate && v.plate !== 'UNKNOWN'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {v.plate || 'UNKNOWN'}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm">
                        <a href={`${API}/${v.image}`} target="_blank" rel="noreferrer"
                          className="text-blue-600 hover:underline">
                          🔍 View Evidence
                        </a>
                      </td>
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>
        </div>
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12 py-6 text-center text-gray-400 text-sm">
        🚔 Helmet Violation Detection System — Group 3
      </footer>

      <style>{`
        @keyframes slide-in { from{transform:translateX(110%);opacity:0} to{transform:translateX(0);opacity:1} }
        .animate-slide-in { animation: slide-in .3s ease-out; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        .animate-pulse { animation: pulse 1.5s ease-in-out infinite; }
      `}</style>
    </div>
  );
}

export default App;
