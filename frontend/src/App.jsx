import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Sparkles, Trash2, Github } from 'lucide-react';
import axios from 'axios';
import './App.css';

/**
 * İsmayıl AI - Əsas İstifadəçi İnterfeysi
 * Bu komponent çat pəncərəsini, mesajların göstərilməsini və backend ilə əlaqəni idarə edir.
 */
function App() {
  // --- Vəziyyət (State) Dəyişənləri ---

  // Mesajların siyahısı (Hər mesajda 'id', 'rol' (user/assistant) və 'metn' var)
  const [mesajlar, mesajlariYenile] = useState([
    {
      id: 1,
      rol: 'assistant',
      metn: 'Salam! Mən İsmayılam. Tamamilə sıfırdan sizin komputerinizdə yaradılmış süni intellekt modeliyəm. Sizə necə kömək edə bilərəm?'
    }
  ]);

  // İstifadəçinin daxil etdiyi cari mətn
  const [girisMetni, girisMetniniYenile] = useState('');

  // Cavabın gözlənilməsi vəziyyəti (Yüklənir işarəsi üçün)
  const [yuklenir, yuklenirVeziyyeti] = useState(false);

  // Çat pəncərəsinin avtomatik aşağı sürüşməsi üçün referans
  const mesajlarSonuRef = useRef(null);

  // --- Funksiyalar ---

  // Yeni mesaj gəldikdə ekranı avtomatik ən aşağıya sürüşdürən funksiya
  const ashagiSurushdur = () => {
    mesajlarSonuRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // mesajlar siyahısı hər dəfə dəyişəndə aşağı sürüşdürürük
  useEffect(() => {
    ashagiSurushdur();
  }, [mesajlar]);

  /**
   * Mesaj göndərmə funksiyası.
   * İstifadəçi düyməni basdıqda və ya Enter vurduqda işə düşür.
   */
  const mesajiGonder = async (e) => {
    if (e) e.preventDefault();

    // Əgər giriş boşdursa və ya artıq sürətli sorğu gedirsə, dayandırırıq
    if (!girisMetni.trim() || yuklenir) return;

    // İstifadəçinin mesajını siyahıya əlavə edirik
    const yeniUserMesaj = {
      id: Date.now(),
      rol: 'user',
      metn: girisMetni
    };

    mesajlariYenile(kohneMesajlar => [...kohneMesajlar, yeniUserMesaj]);
    const gonderilecekMetn = girisMetni;
    girisMetniniYenile(''); // Giriş sahəsini təmizləyirik
    yuklenirVeziyyeti(true); // Yüklənir animasiyasını aktiv edirik

    try {
      // Backend API-yə sorğu göndəririk
      // Qeyd: Backend OpenAI formatında mesaj siyahısı gözləyir
      const cavab = await axios.post('http://localhost:8000/v1/chat/completions', {
        messages: [{ role: 'user', content: gonderilecekMetn }]
      });

      // AI-nin cavabını alırıq
      const aiCavabi = cavab.data.choices[0].message.content;

      // AI cavabını çat siyahısına əlavə edirik
      mesajlariYenile(kohneMesajlar => [...kohneMesajlar, {
        id: Date.now() + 1,
        rol: 'assistant',
        metn: aiCavabi
      }]);
    } catch (xata) {
      console.error("Backend ilə əlaqə xətası:", xata);
      // Xəta halında istifadəçiyə məlumat veririk
      mesajlariYenile(kohneMesajlar => [...kohneMesajlar, {
        id: Date.now() + 1,
        rol: 'assistant',
        metn: "Bağışlayın, backend serveri ilə əlaqə qura bilmədim. Zəhmət olmasa serverin açıq olduğundan əmin olun."
      }]);
    } finally {
      yuklenirVeziyyeti(false); // Yüklənir animasiyasını söndürürük
    }
  };

  // Çat tarixçəsini təmizləyən funksiya
  const chatiniTemizle = () => {
    if (window.confirm("Bütün söhbəti silmək istədiyinizə əminsiniz?")) {
      mesajlariYenile([{
        id: 1,
        rol: 'assistant',
        metn: 'Söhbət təmizləndi. Necə kömək edə bilərəm?'
      }]);
    }
  };

  // --- Render (İnterfeys) ---
  return (
    <div className="app-container">
      {/* Yan Panel (Sidebar) */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Sparkles size={24} color="#10a37f" />
            <span>İsmayıl AI</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button className="nav-item active">
            <Bot size={18} />
            <span>Yeni Söhbət</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <button className="clear-btn" onClick={chatiniTemizle}>
            <Trash2 size={18} />
            <span>Söhbəti Təmizlə</span>
          </button>
        </div>
      </aside>

      {/* Əsas Çat Sahəsi */}
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
                <div className="message-sender">
                  {m.rol === 'assistant' ? 'İsmayıl' : 'Siz'}
                </div>
                <div className="message-text">{m.metn}</div>
              </div>
            </div>
          ))}
          {/* AI cavab yazarkən göstərilən "yazır" animasiyası */}
          {yuklenir && (
            <div className="message-wrapper assistant loading">
              <div className="message-icon"><Bot size={20} /></div>
              <div className="message-content">
                <div className="message-sender">İsmayıl</div>
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={mesajlarSonuRef} />
        </div>

        {/* Mesaj Giriş Sahəsi (Input) */}
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
          <p className="disclaimer">
            İsmayıl sıfırdan kodlanmış bir AI modelidir, səhvlər edə bilər.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
