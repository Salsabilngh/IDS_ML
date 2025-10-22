import React, { useState, useEffect, useRef, useCallback } from "react";
import { io } from "socket.io-client"; // ← import recommandé

const SOCKET_URL = "http://localhost:5000/ids";

const AppStyles = () => (
  <style jsx="true">{`
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body { margin:0; padding:0; width:100%; height:100%; background:#151d29ff; }
    .container { font-family:'Inter',sans-serif; background:#0d1117; color:#c9d1d9; min-height:100vh; padding:40px 20px; display:flex; flex-direction:column; align-items:center; }
    h1 { color:#f7f7f7ff; font-weight:800; margin-bottom:5px; font-size:2rem; text-align:center; }
    h2 { color:#eceff3ff; font-weight:600; margin-top:20px; margin-bottom:15px; font-size:1.25rem; }
    #top-section { display:flex; flex-wrap:wrap; gap:20px; margin-bottom:30px; width:100%; max-width:1000px; justify-content:center; }
    #control-panel-column, #status-display-column, #counter-column {
      flex:1 1 300px; background:#161b22; padding:20px; border-radius:12px; border:1px solid #30363d; box-shadow:0 4px 10px rgba(0,0,0,.5); text-align:center;
    }
    @media (min-width: 1000px) { #top-section { display:grid; grid-template-columns: repeat(3,1fr); gap:20px; } }
    #control-panel { display:flex; flex-direction:column; gap:10px; }
    .control-button { padding:12px 20px; border-radius:8px; font-weight:600; cursor:pointer; transition:background-color .3s, transform .1s, opacity .3s; border:none; color:#fff; }
    #start-button { background-color:#2ea44f; } #start-button:hover:not(:disabled){ background-color:#2c974b; }
    #stop-button { background-color:#f85149; } #stop-button:hover:not(:disabled){ background-color:#d73a49; }
    .control-button:disabled { background-color:#30363d; cursor:not-allowed; opacity:.6; }
    #control-status { margin-top:15px; font-size:.9em; color:#8b949e; padding:8px; background:#21262d; border-radius:6px; }
    #status-display,#attack-counter { padding:30px 10px; border-radius:10px; font-size:1.8rem; font-weight:800; text-shadow:0 0 5px rgba(0,0,0,.5); min-height:100px; display:flex; align-items:center; justify-content:center; transition:background-color .5s, color .5s; }
    #attack-counter { background:#30363d; color:#ff7b72; font-size:2.5rem; border:2px solid #ff7b7230; }
    .NORMAL { background:#238636; color:#fff; border:2px solid #2ea44f; }
    .ATTACK { background:#f85149; color:#fff; animation:pulse-red 1.5s infinite; border:2px solid #d73a49; }
    @keyframes pulse-red { 0%{box-shadow:0 0 0 0 rgba(248,81,73,.7);} 70%{box-shadow:0 0 0 15px rgba(248,81,73,0);} 100%{box-shadow:0 0 0 0 rgba(248,81,73,0);} }

    #upload-card, #log-container {
      width:100%; max-width:1000px; background:#161b22; border:1px solid #30363d; border-radius:12px;
    }
    #upload-card { padding:20px; margin:10px 0 10px; text-align:center; }
    #upload-preview { margin-top:12px; text-align:left; border-top:1px dashed #30363d; padding-top:12px; font-size:.9em; }
    #log-container { max-height:400px; overflow-y:auto; margin-top:10px; padding:15px; font-size:.9em; }
    .log-entry { border-bottom:1px solid #21262d; padding:8px 0; display:flex; align-items:center; }
    .log-entry:last-child { border-bottom:none; }
    .log-attack-text { color:#ff7b72; font-weight:bold; margin-left:10px; }
    .log-normal-text { color:#56d364; margin-left:10px; }
    .small { font-size:.85em; color:#6e7681; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:6px 8px; border-bottom:1px solid #21262d; }
    th { color:#9eb1ff; text-align:left; }
  `}</style>
);

export default function App() {
  // --- State (simulation)
  const [status, setStatus] = useState('AWAITING CONNECTION...');
  const [statusClass, setStatusClass] = useState('NORMAL');
  const [controlStatus, setControlStatus] = useState('Status: Waiting for connection...');
  const [isRunning, setIsRunning] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [log, setLog] = useState([]);
  const [attackCount, setAttackCount] = useState(0);
  const socketRef = useRef(null);
  const MAX_LOG = 50;

  // --- State (upload)
  const [pcapFile, setPcapFile] = useState(null);
  const [uploadMsg, setUploadMsg] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [preview, setPreview] = useState([]); // ← aperçu des flows

  const updateControlUI = useCallback((data) => {
    const { state, message } = data;
    setControlStatus('Status: ' + message);
    if (state === 'RUNNING') {
      setIsRunning(true);
    } else if (state === 'STOPPED') {
      setIsRunning(false);
      setStatus('SIMULATION STOPPED');
      setStatusClass('NORMAL');
    } else if (state === 'STOPPING') {
      setIsRunning(false);
    }
  }, []);

  // Socket.IO
  useEffect(() => {
    const socket = io(SOCKET_URL, { transports: ['websocket', 'polling'], forceNew: true });
    socketRef.current = socket;

    const idsUpdateHandler = (data) => {
      const newStatus = data.status;
      setStatus(newStatus === "ATTACK" ? "INTRUSION DETECTED" : "SYSTEM SECURE");
      setStatusClass(newStatus);
      if (newStatus === 'ATTACK') setAttackCount(c => c + 1);

      const logTime = new Date((data.timestamp || Date.now()/1000) * 1000).toLocaleTimeString();
      const statusHTML = newStatus === 'ATTACK'
        ? <span className="log-attack-text">🚨 THREAT ALERT — {newStatus} (p={data.prob_attack?.toFixed?.(2) ?? "?"})</span>
        : <span className="log-normal-text">✅ Status OK — {newStatus} (p={data.prob_attack?.toFixed?.(2) ?? "?"})</span>;

      const extra = [];
      if (data.src && data.dst) extra.push(` ${data.src} → ${data.dst}`);
      if (data.proto) extra.push(` [${data.proto}:${data.sport ?? "?"}->${data.dport ?? "?"}]`);

      const entry = { id: data.id || Date.now(), time: logTime, html: <>{statusHTML}<span style={{color:'#8b949e'}}>{extra.join(' ')}</span></> };
      setLog(prev => [entry, ...prev.slice(0, MAX_LOG - 1)]);
    };

    socket.on('connect', () => {
      setIsConnected(true);
      setStatus('READY');
      setStatusClass('NORMAL');
      setControlStatus('Status: Connected. Ready to start.');
    });
    socket.on('disconnect', () => {
      setIsConnected(false);
      updateControlUI({ state: 'STOPPED', message: 'CONNECTION LOST' });
      setStatus('DISCONNECTED');
      setStatusClass('ATTACK');
    });
    socket.on('status_control', updateControlUI);
    socket.on('ids_update', idsUpdateHandler);

    return () => {
      socket.off('ids_update', idsUpdateHandler);
      socket.off('status_control', updateControlUI);
      socket.off('connect'); socket.off('disconnect');
      if (socket.connected) socket.disconnect();
    };
  }, [updateControlUI]);

  // Simulation controls
  const handleStart = () => { if (isConnected && !isRunning) socketRef.current.emit('start_simulation'); };
  const handleStop  = () => { if (isConnected && isRunning)  socketRef.current.emit('stop_simulation'); };
  const handleResetCounter = () => setAttackCount(0);

  // Upload handlers
  const handlePcapChange = (e) => {
    setPcapFile(e.target.files?.[0] || null);
    setUploadMsg("");
    setPreview([]);
  };

  const uploadAndAnalyze = async () => {
    if (!pcapFile) { setUploadMsg("Choisis un fichier .pcap ou .pcapng"); return; }
    if (!/\.(pcap|pcapng)$/i.test(pcapFile.name)) {
      setUploadMsg("Extension non autorisée. (.pcap/.pcapng)"); return;
    }
    setUploadMsg(""); setIsUploading(true); setPreview([]);
    try {
      const form = new FormData();
      form.append("file", pcapFile);
      // URL absolue = pas besoin de proxy
      const res = await fetch("http://localhost:5000/upload_pcap", { method: "POST", body: form });
      let data = null;
      try { data = await res.json(); } catch {}
      if (!res.ok || !data?.ok) {
        const msg = data?.error || `Upload échoué (HTTP ${res.status})`;
        setUploadMsg("Erreur: " + msg);
        return;
      }
      setUploadMsg(`Analyse OK: ${data.n_flows} flows. CSV: ${data.csv_path}`);
      setPreview(data.preview || []);
    } catch (e) {
      setUploadMsg("Erreur réseau: " + e.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="container">
      <AppStyles />
      <h1>Intrusion Detection System based ML - Salsabil Naghmouchi</h1>
      <p className="text-gray-500 mb-6"><p id="modelPowered" className="subtitle">
  Powered By : XGBoost + Isolation Forest + Autoencoder
</p>
</p>

      <div id="top-section">
        <div id="control-panel-column">
          <h2>Simulation Control</h2>
          <div id="control-panel">
            <button id="start-button" className="control-button" onClick={handleStart} disabled={isRunning || !isConnected}>START STREAM</button>
            <button id="stop-button"  className="control-button" onClick={handleStop}  disabled={!isRunning || !isConnected}>STOP STREAM</button>
            <div id="control-status">{controlStatus}</div>
          </div>
        </div>
        <div id="status-display-column">
          <h2>Live Status</h2>
          <div id="status-display" className={statusClass}>{status}</div>
        </div>
        <div id="counter-column">
          <h2>Total Threats Detected</h2>
          <div id="attack-counter">{attackCount}</div>
          <button className="control-button" onClick={handleResetCounter} style={{marginTop:10, backgroundColor:'#30363d'}}>RESET COUNTER</button>
        </div>
      </div>

      {/* --- Upload PCAP --- */}
      <div id="upload-card">
        <h2>Analyze PCAP</h2>
        <input type="file" accept=".pcap,.pcapng" onChange={handlePcapChange} style={{ marginRight: 10 }} />
        <button className="control-button" onClick={uploadAndAnalyze} disabled={isUploading} style={{ backgroundColor:'#238636' }}>
          {isUploading ? "Uploading..." : "Upload & Analyze"}
        </button>
        <div style={{ marginTop: 10, color: uploadMsg.startsWith("Erreur") ? '#ff7b72' : '#8b949e' }}>{uploadMsg}</div>
        <div className="small">Les résultats s’affichent aussi en direct dans l’Activity Log.</div>

        {/* Aperçu des 20 premiers flows */}
        {preview.length > 0 && (
          <div id="upload-preview">
            <h3 style={{margin:'6px 0 8px'}}>Preview (top 20 flows)</h3>
            <table>
              <thead>
                <tr>
                  <th>#</th><th>src</th><th>dst</th><th>proto</th><th>sport</th><th>dport</th><th>status</th><th>p</th>
                </tr>
              </thead>
              <tbody>
                {preview.map((r, i) => (
                  <tr key={i}>
                    <td>{i+1}</td>
                    <td>{r.src}</td>
                    <td>{r.dst}</td>
                    <td>{r.proto}</td>
                    <td>{r.sport}</td>
                    <td>{r.dport}</td>
                    <td style={{fontWeight:700, color: r.status === "ATTACK" ? "#ff7b72" : "#56d364"}}>{r.status}</td>
                    <td>{typeof r.prob_attack === "number" ? r.prob_attack.toFixed(2) : r.prob_attack}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <h2>Activity Log</h2>
      <div id="log-container">
        {log.length === 0 && !isRunning ? (
          <div style={{ padding: 10, color: '#777' }}>Log empty. Press START STREAM or upload a PCAP to begin.</div>
        ) : (
          log.map((entry, idx) => (
            <div key={`${entry.id}-${idx}`} className="log-entry">
              <span style={{color:'#fcfbf5ff', fontWeight:'bold'}}>[{entry.time}]</span>
              <span style={{margin:'0 10px', color:'#8b949e'}}>Sample #{entry.id}:</span>
              {entry.html}
            </div>
          ))
        )}
      </div>

      <p style={{marginTop:30, fontSize:'.8em', color:'#777'}}>Architecture: Flask-SocketIO Backend | Frontend: React.js</p>
    </div>
  );
}
