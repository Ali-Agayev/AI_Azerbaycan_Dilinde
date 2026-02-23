from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import os
from pathlib import Path

# Torch opsionaldir — Railway-də lazim deyil (GPU Kaggle-dadır)
try:
    import torch
    TORCH_VAR = True
except ImportError:
    TORCH_VAR = False

# Model və tokenizer opsionaldir — yalnız lokal dev-də işləyir
try:
    from model import IsmayilModeli
    from tokenizer import CharTokenizator
    MODEL_VAR = True
except ImportError:
    MODEL_VAR = False

from kaggle_client import is_gondər, is_veziyyeti, is_siyahisi

# FastAPI tətbiqini yaradırıq
ismayil_server = FastAPI(title="İsmayılın Şəxsi AI Serveri")

# CORS tənzimləmələri
# Bütün mənbələrə icazə veririk ki, Vercel rahat qoşulsun
ismayil_server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qlobal dəyişənlər — Model və Tokenizatoru burada saxlayırıq
cihaz = ('cuda' if (TORCH_VAR and torch.cuda.is_available()) else 'cpu') if TORCH_VAR else 'cpu'
tokenizator = None
ismayil_modeli = None

def ai_ni_bashlat():
    """
    Modeli və Tokenizatoru yaddaşdan yükləyən funksiya.
    Torch və ya model faylı yoxdursa False qaytarır (Railway-də normal haldir).
    """  
    if not TORCH_VAR or not MODEL_VAR:
        return False  # Railway-də torch/model yoxdur — chat endpoint disabled
    global tokenizator, ismayil_modeli
    if ismayil_modeli is None:
        # Faylların varlığını yoxlayırıq
        if not os.path.exists('tokenizer.json') or not os.path.exists('ismayil_model.pth'):
            return False
            
        # Tokenizatoru yükləyirik
        tokenizator = CharTokenizator.yukle('tokenizer.json')
        # Model memarlığını yaradırıq
        ismayil_modeli = IsmayilModeli(tokenizator.luget_olcusu)
        # Öyrənilmiş çəkiləri (weights) modelə yükləyirik
        ismayil_modeli.load_state_dict(torch.load('ismayil_model.pth', map_location=cihaz))
        ismayil_modeli.to(cihaz)
        ismayil_modeli.eval() # Modeli yalnız cavab vermə (inference) rejiminə salırıq
        print("İsmayıl AI uğurla işə düşdü və suallarınızı gözləyir!")
    return True

# API istəkləri üçün məlumat strukturları (Pydantic modelləri)
class Mesaj(BaseModel):
    role: str # 'user' və ya 'assistant'
    content: str # Mesajın mətni

class ChatIsteyi(BaseModel):
    messages: List[Mesaj]

@ismayil_server.get("/")
async def ana_sehife():
    # Serverin aktiv olub-olmadığını yoxlamaq üçün kiçik endpoint
    return {"status": "online", "model": "İsmayıl Custom Transformer"}

@ismayil_server.post("/v1/chat/completions")
async def chat_cavabi(istek: ChatIsteyi):
    """
    Bu əsas hissədir. İstifadəçidən gələn mesajı AI-yə göndərir və cavab alır.
    """
    # AI-nin hazır olub-olmadığını yoxlayırıq
    if not ai_ni_bashlat():
        # Railway-də lokal model yoxdursa "bağışlayın" mock cavabı qaytarırıq (xəta verməkdən yaxşıdır)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Salam! Mən hazırda 'Video Düzəlt' rejimində, yüngül (Railway) serverdə işləyirəm. Öz 'Custom Transformer' beynim (PyTorch) bu serverə yüklənməyib. Mənlə real söhbət etmək üçün məni öz kompüterinizdə (Anaconda ilə) çalışdırın və ya 'Video Düzəlt' bölməsindən videomuzu hazırlayaq! 🎬"
                    }
                }
            ]
        }
        
    try:
        # İstifadəçinin son göndərdiyi sualı götürürük
        son_mesaj = istek.messages[-1].content
        
        # Mətni rəqəmlərə (tokenlərə) çeviririk
        giriş_idləri = torch.tensor([tokenizator.kodlasdir(son_mesaj)], dtype=torch.long, device=cihaz)
        
        # AI-dən yeni simvollar generasya etməsini istəyirik
        with torch.no_grad():
            # generate() yerinə yeni_metn_yarat() istifadə edirik (model.py-da adını dəyişmişik)
            butun_idlar = ismayil_modeli.yeni_metn_yarat(giriş_idləri, maksimum_yeni_simvol=100)
            # Yalnız AI-nin yaratdığı hissəni kəsib götürürük (giriş mətnini çıxarıq)
            cavab_idlari = butun_idlar[0][len(giriş_idləri[0]):].tolist()
            # Rəqəmləri yenidən başa düşülən mətnə çeviririk
            cavab_metni = tokenizator.de_kodlasdir(cavab_idlari)
            
        # OpenAI formatına uyğun cavab qaytarırıq (Frontend bunu gözləyir)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": cavab_metni
                    }
                }
            ]
        }
    except Exception as xata:
        raise HTTPException(status_code=500, detail=str(xata))

# ═══════════════════════════════════════════════════════════
# 🎬 VİDEO DÜZƏLTMƏ ENDPOINT-LƏRİ (Kaggle + Stable Diffusion)
# ═══════════════════════════════════════════════════════════

@ismayil_server.post("/video/edit")
async def video_duzelт(
    video: UploadFile = File(..., description="Düzəldiləcək video fayl (mp4, avi, mov)"),
    prompt: str = Form(..., description="Necə görünsün? Məs: 'oil painting style, colorful'")
):
    """
    Video faylı və prompt qəbul edib Kaggle-a emal üçün göndərir.
    Dərhal iş ID-si qaytarır — status-u /video/status/{job_id} ilə izləyin.
    """
    # Fayl növünü yoxlayırıq
    icaze_verilmis = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
    if video.content_type and video.content_type not in icaze_verilmis:
        # Content-type həmişə dəqiq olmur, buna görə adını da yoxlayırıq
        ad = (video.filename or "").lower()
        if not any(ad.endswith(x) for x in [".mp4", ".avi", ".mov", ".mkv"]):
            raise HTTPException(
                status_code=400,
                detail="Yalnız video faylları (mp4, avi, mov, mkv) qəbul edilir."
            )
    
    # Video məzmununu oxuyuruq
    video_bytes = await video.read()
    
    if len(video_bytes) == 0:
        raise HTTPException(status_code=400, detail="Boş fayl göndərildi.")
    
    # Maksimum ölçü yoxlanışı (500 MB)
    max_olchu = 500 * 1024 * 1024
    if len(video_bytes) > max_olchu:
        raise HTTPException(
            status_code=413,
            detail=f"Fayl həddən böyükdür. Maksimum: 500 MB, Göndərilən: {len(video_bytes)//1024//1024} MB"
        )
    
    # Kaggle-a iş göndəririk
    is_id = is_gondər(
        video_bytes=video_bytes,
        prompt=prompt,
        fayl_adi=video.filename or "video.mp4"
    )
    
    return {
        "success": True,
        "job_id": is_id,
        "message": "Video emal üçün qəbul edildi!",
        "kaggle_telimat": (
            f"📋 Kaggle Notebook-a gedin → video_jobs/{is_id}/ qovluğundakı video + prompt.txt fayllarını "
            f"Kaggle Dataset-ə yükləyin → video_edit_worker.py notebook-unu işlədin → "
            f"nəticə /kaggle/working/output.mp4-da olacaq"
        ),
        "status_url": f"/video/status/{is_id}"
    }


@ismayil_server.get("/video/status/{is_id}")
async def video_status(is_id: str):
    """İş vəziyyətini yoxlayır: pending | processing | done | error"""
    return is_veziyyeti(is_id)


@ismayil_server.get("/video/download/{is_id}")
async def video_yukle(is_id: str):
    """Tamamlanmış çıxış videosunu yükləməyə imkan verir."""
    melumat = is_veziyyeti(is_id)
    
    if melumat["status"] == "not_found":
        raise HTTPException(status_code=404, detail="İş tapılmadı")
    
    if melumat["status"] != "done":
        raise HTTPException(
            status_code=202,
            detail=f"Video hələ hazır deyil. Cari vəziyyət: {melumat['status']}"
        )
    
    video_yolu = Path("video_jobs") / is_id / "output.mp4"
    if not video_yolu.exists():
        raise HTTPException(status_code=404, detail="Çıxış faylı tapılmadı")
    
    return FileResponse(
        path=str(video_yolu),
        media_type="video/mp4",
        filename=f"ishayil_ai_video_{is_id}.mp4"
    )


@ismayil_server.get("/video/jobs")
async def butun_isler():
    """Bütün video emal işlərinin siyahısı."""
    return {"jobs": is_siyahisi()}


if __name__ == "__main__":
    # Railway və digər serverlər PORT env variable istifadə edir
    # Lokal istifadə üçün default 8000
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Bütün interfeysldən qulaq as
    print(f"🚀 İsmayıl AI server: http://{host}:{port}")
    print(f"   Kaggle API: {'ENV VARS' if os.environ.get('KAGGLE_KEY') else '~/.kaggle/kaggle.json'}")
    uvicorn.run(ismayil_server, host=host, port=port)
