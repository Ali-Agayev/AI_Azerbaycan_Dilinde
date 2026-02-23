import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Sparkles, Trash2, Video, Upload, Play, Download, Clock } from 'lucide-react';
import axios from 'axios';
import './App.css';

/**
 * İsmayıl AI — Əsas İstifadəçi İnterfeysi
 * ✅ Söhbət Tab: Custom Transformer ilə çat
 * 🎬 Video Düzəlt Tab: Stable Diffusion (Kaggle T4 GPU) ilə video düzəltmə
 */
function App() {
  // ── Aktiv Tab ────────────────────────────────────────────────────────────
  const [aktifTab, aktifTabiYenile] = useState('chat'); // 'chat' | 'video'

  // ── Söhbət State-ləri ────────────────────────────────────────────────────
  const [mesajlar, mesajlariYenile] = useState([{
    id: 1, rol: 'assistant',
    metn: 'Salam! Mən İsmayılam. Tamamilə sıfırdan sizin komputerinizdə yaradılmış süni intellekt modeliyəm. Sizə necə kömək edə bilərəm?'
  }]);
  const [girisMetni, girisMetniniYenile] = useState('');
  const [yuklenir, yuklenirVeziyyeti] = useState(false);
  const mesajlarSonuRef = useRef(null);

  // ── Video Düzəltmə State-ləri ────────────────────────────────────────────
  const [secilmisVideo, secilmisVideoYenile] = useState(null);       // File object
  const [videoOnizleme, videoOnizlemeYenile] = useState(null);       // blob URL
  const [videoPrompt, videoPromptYenile] = useState('');             // Prompt mətni
  const [isId, isIdYenile] = useState(null);                         // job_id
  const [isVeziyyeti, isVeziyyetiYenile] = useState(null);           // status obj
  const [videoYuklenir, videoYuklenirYenile] = useState(false);      // Backend sorğusu
  const [surukleme, suruklemeYenile] = useState(false);              // Drag & drop
  const pollingRef = useRef(null);

  // ── Söhbət Funksiyaları ──────────────────────────────────────────────────
  const ashagiSurushdur = () => mesajlarSonuRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { ashagiSurushdur(); }, [mesajlar]);

  const mesajiGonder = async (e) => {
    if (e) e.preventDefault();
    if (!girisMetni.trim() || yuklenir) return;
    const yeniUserMesaj = { id: Date.now(), rol: 'user', metn: girisMetni };
    mesajlariYenile(k => [...k, yeniUserMesaj]);
    const gonderilenMetn = girisMetni;
    girisMetniniYenile('');
    yuklenirVeziyyeti(true);
    try {
      const cavab = await axios.post('https://aiazerbaycandilinde-production.up.railway.app/v1/chat/completions', {
        messages: [{ role: 'user', content: gonderilenMetn }]
      });
      const aiCavabi = cavab.data.choices[0].message.content;
      mesajlariYenile(k => [...k, { id: Date.now() + 1, rol: 'assistant', metn: aiCavabi }]);
    } catch {
      mesajlariYenile(k => [...k, {
        id: Date.now() + 1, rol: 'assistant',
        metn: 'Bağışlayın, backend serveri ilə əlaqə qura bilmədim.'
      }]);
    } finally {
      yuklenirVeziyyeti(false);
    }
  };

  const chatiniTemizle = () => {
    if (window.confirm('Bütün söhbəti silmək istədiyinizə əminsiniz?')) {
      mesajlariYenile([{ id: 1, rol: 'assistant', metn: 'Söhbət təmizləndi. Necə kömək edə bilərəm?' }]);
    }
  };

  // ── Video Funksiyaları ───────────────────────────────────────────────────
  const videoSec = (fayl) => {
    if (!fayl) return;
    if (!fayl.type.startsWith('video/')) {
      alert('Zəhmət olmasa yalnız video fayl seçin (mp4, avi, mov).');
      return;
    }
    secilmisVideoYenile(fayl);
    // Önizlmə URL-i yarat
    if (videoOnizleme) URL.revokeObjectURL(videoOnizleme);
    videoOnizlemeYenile(URL.createObjectURL(fayl));
    // Əvvəlki nəticəni sıfırla
    isIdYenile(null);
    isVeziyyetiYenile(null);
    if (pollingRef.current) clearInterval(pollingRef.current);
  };

  const suruklemeilesle = (e) => {
    e.preventDefault();
    suruklemeYenile(true);
  };

  const suruklemebitirildi = (e) => {
    e.preventDefault();
    suruklemeYenile(false);
    const fayl = e.dataTransfer.files?.[0];
    if (fayl) videoSec(fayl);
  };

  // Status polling — hər 5 saniyədən bir vəziyyəti yoxlayır
  const pollingBaslat = (jobId) => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = setInterval(async () => {
      try {
        const r = await axios.get(`https://aiazerbaycandilinde-production.up.railway.app/video/status/${jobId}`);
        isVeziyyetiYenile(r.data);
        if (r.data.status === 'done' || r.data.status === 'error') {
          clearInterval(pollingRef.current);
        }
      } catch { /* şəbəkə xətası — növbəti polling-ə qədər gözlə */ }
    }, 5000);
  };

  const videoDuzelт = async () => {
    if (!secilmisVideo || !videoPrompt.trim()) return;
    videoYuklenirYenile(true);
    const formData = new FormData();
    formData.append('video', secilmisVideo);
    formData.append('prompt', videoPrompt);
    try {
      const r = await axios.post('https://aiazerbaycandilinde-production.up.railway.app/video/edit', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const jobId = r.data.job_id;
      isIdYenile(jobId);
      isVeziyyetiYenile({ status: 'pending', job_id: jobId });
      pollingBaslat(jobId);
    } catch (xata) {
      alert('Xəta baş verdi: ' + (xata.response?.data?.detail || xata.message));
    } finally {
      videoYuklenirYenile(false);
    }
  };

  // Cleanup
  useEffect(() => () => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    if (videoOnizleme) URL.revokeObjectURL(videoOnizleme);
  }, []);

  // ── Status Etiketləri ────────────────────────────────────────────────────
  const statusRengi = {
    pending: '#f59e0b', processing: '#3b82f6', done: '#10b981', error: '#ef4444'
  };
  const statusMetni = {
    pending: '⏳ Kaggle Notebook-da emalı gözləyir...',
    processing: '⚙️ Frame-lər emal edilir...',
    done: '✅ Video hazırdır!',
    error: '❌ Xəta baş verdi'
  };

  // ════════════════════════════════════════════════════════════════════════
  return (
    <div className="app-container">
      {/* Yan Panel */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Sparkles size={24} color="#10a37f" />
            <span>İsmayıl AI</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${aktifTab === 'chat' ? 'active' : ''}`}
            onClick={() => aktifTabiYenile('chat')}
          >
            <Bot size={18} />
            <span>Söhbət</span>
          </button>
          <button
            className={`nav-item ${aktifTab === 'video' ? 'active' : ''}`}
            onClick={() => aktifTabiYenile('video')}
          >
            <Video size={18} />
            <span>Video Düzəlt</span>
          </button>
        </nav>

        {aktifTab === 'chat' && (
          <div className="sidebar-footer">
            <button className="clear-btn" onClick={chatiniTemizle}>
              <Trash2 size={18} />
              <span>Söhbəti Təmizlə</span>
            </button>
          </div>
        )}
      </aside>

      {/* ── SÖHBƏT TAB ──────────────────────────────────────────────────── */}
      {aktifTab === 'chat' && (
        <main className="chat-main">
          <header className="chat-header">
            <h2>İsmayıl (Custom Transformer)</h2>
            <div className="header-status">
              <span className="status-dot"></span>
              Server: Online
            </div>
          </header>

          <div className="messages-container">
            {mesajlar.map((m) => (
              <div key={m.id} className={`message-wrapper ${m.rol}`}>
                <div className="message-icon">
                  {m.rol === 'assistant' ? <Bot size={20} /> : <User size={20} />}
                </div>
                <div className="message-content">
                  <div className="message-sender">{m.rol === 'assistant' ? 'İsmayıl' : 'Siz'}</div>
                  <div className="message-text">{m.metn}</div>
                </div>
              </div>
            ))}
            {yuklenir && (
              <div className="message-wrapper assistant loading">
                <div className="message-icon"><Bot size={20} /></div>
                <div className="message-content">
                  <div className="message-sender">İsmayıl</div>
                  <div className="typing-indicator"><span></span><span></span><span></span></div>
                </div>
              </div>
            )}
            <div ref={mesajlarSonuRef} />
          </div>

          <footer className="chat-footer">
            <form className="input-container" onSubmit={mesajiGonder}>
              <input
                type="text"
                placeholder="İsmayıldan bir şey soruşun..."
                value={girisMetni}
                onChange={(e) => girisMetniniYenile(e.target.value)}
                disabled={yuklenir}
              />
              <button type="submit" className="send-btn" disabled={!girisMetni.trim() || yuklenir}>
                <Send size={20} />
              </button>
            </form>
            <p className="disclaimer">İsmayıl sıfırdan kodlanmış bir AI modelidir, səhvlər edə bilər.</p>
          </footer>
        </main>
      )}

      {/* ── VİDEO DÜZƏLT TAB ────────────────────────────────────────────── */}
      {aktifTab === 'video' && (
        <main className="chat-main video-main">
          <header className="chat-header">
            <h2>🎬 Video Düzəlt (Stable Diffusion)</h2>
            <div className="header-status">
              <span className="status-dot" style={{ background: '#3b82f6' }}></span>
              Kaggle T4 GPU
            </div>
          </header>

          <div className="video-workspace">

            {/* ── Addım 1: Video Yüklə ── */}
            <div className="video-section">
              <h3 className="section-title">
                <span className="step-badge">1</span>
                Video Seçin
              </h3>

              <div
                className={`drop-zone ${surukleme ? 'dragging' : ''} ${secilmisVideo ? 'has-file' : ''}`}
                onDragOver={suruklemeilesle}
                onDragLeave={() => suruklemeYenile(false)}
                onDrop={suruklemebitirildi}
                onClick={() => document.getElementById('video-input').click()}
              >
                <input
                  id="video-input"
                  type="file"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={(e) => videoSec(e.target.files?.[0])}
                />
                {secilmisVideo ? (
                  <div className="file-selected">
                    <Play size={32} color="#10a37f" />
                    <div>
                      <p className="file-name">{secilmisVideo.name}</p>
                      <p className="file-size">{(secilmisVideo.size / 1e6).toFixed(1)} MB</p>
                    </div>
                  </div>
                ) : (
                  <div className="drop-placeholder">
                    <Upload size={40} color="#555" />
                    <p>Videonu buraya sürükleyin</p>
                    <span>və ya klikləyin (mp4, avi, mov)</span>
                  </div>
                )}
              </div>

              {/* Video önizləməsi */}
              {videoOnizleme && (
                <video
                  className="video-preview"
                  src={videoOnizleme}
                  controls
                  muted
                />
              )}
            </div>

            {/* ── Addım 2: Prompt ── */}
            <div className="video-section">
              <h3 className="section-title">
                <span className="step-badge">2</span>
                Prompt Yazın
              </h3>
              <textarea
                className="prompt-input"
                placeholder="Videonu necə görmək istəyirsiniz?&#10;Məsələn: oil painting style, impressionist, colorful brushstrokes&#10;Məsələn: anime style, bold outlines, vibrant colors&#10;Məsələn: cyberpunk neon city, dark, futuristic"
                value={videoPrompt}
                onChange={(e) => videoPromptYenile(e.target.value)}
                rows={4}
              />

              {/* Hazır promptlar */}
              <div className="prompt-chips">
                {[
                  'oil painting style, impressionist',
                  'anime style, vibrant colors',
                  'cyberpunk neon, dark futuristic',
                  'watercolor painting, soft colors',
                  'sketch style, pencil drawing'
                ].map((p) => (
                  <button
                    key={p}
                    className="prompt-chip"
                    onClick={() => videoPromptYenile(p)}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            {/* ── Addım 3: Düzəlt ── */}
            <div className="video-section">
              <h3 className="section-title">
                <span className="step-badge">3</span>
                Emal Edin
              </h3>

              <button
                className="edit-btn"
                onClick={videoDuzelт}
                disabled={!secilmisVideo || !videoPrompt.trim() || videoYuklenir || isVeziyyeti?.status === 'pending' || isVeziyyeti?.status === 'processing'}
              >
                {videoYuklenir ? (
                  <><span className="spinner-mini"></span> Göndərilir...</>
                ) : (
                  <><Sparkles size={20} /> Düzəlt (Kaggle GPU)</>
                )}
              </button>

              {/* Kaggle Notebook təlimatı silindi — artıq 100% avtomatikdir */}

              {/* Status göstəricisi */}
              {isVeziyyeti && (
                <div className="status-card" style={{ borderColor: statusRengi[isVeziyyeti.status] }}>
                  <div className="status-header">
                    {isVeziyyeti.status !== 'done' && isVeziyyeti.status !== 'error' && (
                      <span className="status-spinner"></span>
                    )}
                    <span style={{ color: statusRengi[isVeziyyeti.status] }}>
                      {statusMetni[isVeziyyeti.status] || isVeziyyeti.status}
                    </span>
                  </div>

                  {isVeziyyeti.status === 'error' && isVeziyyeti.error && (
                    <div style={{ marginTop: '10px', fontSize: '13px', color: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', padding: '10px', borderRadius: '6px', whiteSpace: 'pre-wrap' }}>
                      <strong>XƏTA DETALI:</strong><br />
                      {isVeziyyeti.error}
                    </div>
                  )}

                  {isVeziyyeti.status === 'done' && isVeziyyeti.video_url && (
                    <a
                      href={`https://aiazerbaycandilinde-production.up.railway.app${isVeziyyeti.video_url}`}
                      target="_blank" rel="noreferrer"
                      className="download-btn"
                    >
                      <Download size={18} /> Nəticəni Yüklə
                    </a>
                  )}
                </div>
              )}
            </div>
          </div>
        </main>
      )}
    </div>
  );
}

export default App;
