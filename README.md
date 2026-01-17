# İsmayıl AI - Sıfırdan Transformer Projesi 🇦🇿🤖

Bu layihə, heç bir hazır AI kitabxanasından (Ollama, OpenAI və s.) asılı olmayan, birbaşa PyTorch ilə sıfırdan kodlanmış bir Transformer (Dil Modeli) arxitekturasıdır.

## 🌟 Xüsusiyyətlər
- **Tam Müstəqil:** Bütün neyron şəbəkə qatları (`Attention`, `FeedForward`, `TransformerBlock`) əllə yazılmışdır.
- **Azərbaycan Dilində Kod:** Kodun hər bir sətiri Azərbaycan dilində dəyişən adları və geniş şərhlərlə sənədləşdirilib.
- **Lokal və Təhlükəsiz:** Bütün hesablamalar yerli (local) icra olunur, məlumatlarınız heç yerə göndərilmir.
- **Modern İnterfeys:** React + Vite ilə hazırlanmış, ChatGPT üslubunda qaranlıq rejimli (dark mode) UI.

## 🛠 Texnologiyalar
- **Backend:** Python, PyTorch, FastAPI, Uvicorn.
- **Frontend:** React, Vite, CSS3, Lucide Icons, Axios.
- **AI Core:** Custom Transformer (Karakter səviyyəli Tokenizer).

## 🚀 Quraşdırma

### 1. Backend-i işə salın:
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python train.py  # Modeli öyrətmək üçün
python main.py   # Serveri başlatmaq üçün
```

### 2. Frontend-i işə salın:
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Model Haqqında
Bu model "Attention is All You Need" məqaləsindəki orijinal Transformer arxitekturasına əsaslanır. Karakter səviyyəsində tokenizasiya istifadə edir və kiçik ölçülü olduğu üçün fərdi komputerlərdə rahatlıqla öyrədilə bilir.

## 📜 Lisenziya
Bu layihə açıq mənbəlidir. İstədiyiniz kimi paylaşıb inkişaf etdirə bilərsiniz.

---
**İsmayıl AI** - Azərbaycanın rəqəmsal gələcəyi üçün güclü bir başlanğıc! 🚀🌻
